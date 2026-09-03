"""
eval_fast.py — Fast metrics for CT report generation.

Computes (in seconds-to-minutes):
  - Text metrics: BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr  (via shared/metrics)
  - Clinical efficacy: weighted Precision/Recall/F1 from CT-RATE 18 findings
                       using the official RadBert classifier
  - CRG Score (Hamamci et al., MIDL 2025) from the same RadBert predictions

Slow LLM-based metrics (GREEN, Llama-Score) are deliberately excluded — see
`eval_slow_green.py` and `eval_slow_llama.py` for those.

Input JSONL format (one record per line):
    {"volume_id": "...", "generated_text": "...", "reference_text": "..."}

Usage:
    python eval_fast.py \\
        --pred predictions.jsonl \\
        --labels-csv .../valid_predicted_labels.csv \\
        --radbert .../RadBertClassifier.pth \\
        --out metrics_fast.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# shared/metrics is VENDORED under core_code/vendor/ (2026-07-16). It used to point at
# "<an external repo, now vendored here as core_code/vendor>", which was renamed to AdaRAG-CT-RAW — the import
# then silently resolved to nothing and report-gen eval could not run at all. Cross-repo paths rot; vendored
# copies do not. See core_code/vendor/README.md.
SHARED_ROOT = str(Path(__file__).resolve().parents[2] / "vendor")
if SHARED_ROOT not in sys.path:
    sys.path.insert(0, SHARED_ROOT)

# Local CRG implementation (verified against paper Table 2)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crg_score import crg_score, crg_score_with_components

DEFAULT_LABELS_CSV = "./data/labels/valid_predicted_labels.csv"
DEFAULT_RADBERT    = "./checkpoints/RadBertClassifier.pth"


def make_flat_summary(payload: Dict[str, Any]) -> Dict[str, float | int]:
    """Canonical flat headline metrics while preserving nested details."""

    summary: Dict[str, float | int] = {}
    if "n_predictions" in payload:
        summary["n_predictions"] = int(payload["n_predictions"])
        summary["validation_metric_samples"] = int(payload["n_predictions"])
    text = payload.get("text") or {}
    if isinstance(text, dict):
        for key in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "bleu_sacre", "rouge_l", "meteor", "cider"):
            if key in text and text[key] is not None:
                summary[key] = float(text[key])
    clinical = payload.get("clinical") or {}
    if isinstance(clinical, dict):
        for source_key, out_key in (
            ("precision", "clinical_precision"),
            ("recall", "clinical_recall"),
            ("f1", "clinical_f1"),
            ("micro_f1", "clinical_micro_f1"),
            ("macro_f1", "clinical_macro_f1"),
            ("crg", "clinical_crg"),
        ):
            if source_key in clinical and clinical[source_key] is not None:
                summary[out_key] = float(clinical[source_key])
        if "clinical_crg" in summary:
            summary["crg"] = summary["clinical_crg"]
        if "support" in clinical and clinical["support"] is not None:
            summary["clinical_support"] = int(clinical["support"])
        confusion = clinical.get("confusion") or {}
        if isinstance(confusion, dict):
            for source_key, out_key in (
                ("TP", "clinical_tp"),
                ("FP", "clinical_fp"),
                ("FN", "clinical_fn"),
                ("TN", "clinical_tn"),
            ):
                if source_key in confusion and confusion[source_key] is not None:
                    summary[out_key] = int(confusion[source_key])
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, help="Predictions JSONL file (volume_id, generated_text, reference_text)")
    p.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    p.add_argument("--radbert", default=DEFAULT_RADBERT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--no-text", action="store_true", help="Skip text metrics (BLEU/ROUGE/METEOR/CIDEr)")
    p.add_argument("--no-clinical", action="store_true", help="Skip RadBert classifier metrics (F1/CRG)")
    p.add_argument(
        "--clinical-labels-out",
        default="",
        help=(
            "Output JSONL path for per-sample RadBERT labels. "
            "Default: clinical_labels.jsonl next to --out."
        ),
    )
    p.add_argument(
        "--no-save-clinical-labels",
        action="store_true",
        help="Do not save per-sample RadBERT y_true/y_pred labels.",
    )
    return p.parse_args()


def load_predictions(jsonl_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Returns (volume_ids, hypotheses, references)."""
    vols, hyp, ref = [], [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vols.append(r["volume_id"])
            hyp.append(r["generated_text"])
            ref.append(r["reference_text"])
    if not vols:
        raise ValueError(f"No predictions in {jsonl_path}")
    return vols, hyp, ref


def lookup_labels(volume_ids: List[str], labels_csv: str) -> np.ndarray:
    """Returns ground-truth labels of shape (N, 18); raises if any volume not found."""
    from shared.metrics.medical.clinical_efficacy import (
        load_clinical_labels_csv, CLINICAL_FINDINGS,
    )
    label_map = load_clinical_labels_csv(labels_csv)
    out = []
    missing = []
    for v in volume_ids:
        # Normalise to match how load_clinical_labels_csv keyed them.
        key = v
        if key not in label_map:
            # Try common variants — strip suffix or match with .nii.gz
            alt = v.replace(".nii.gz", "")
            if alt in label_map:
                key = alt
            else:
                missing.append(v)
                continue
        out.append(label_map[key])
    if missing:
        raise KeyError(f"Volume IDs not found in labels CSV (showing first 5): {missing[:5]} (total {len(missing)})")
    arr = np.asarray(out, dtype=np.int32)
    assert arr.shape == (len(volume_ids), len(CLINICAL_FINDINGS)), arr.shape
    return arr


def compute_text_metrics(hyp: List[str], ref: List[str]) -> Dict[str, float]:
    from shared.metrics.text.text_metrics import compute_all_text_metrics
    return compute_all_text_metrics(hypotheses=hyp, references=ref)


def compute_clinical_and_crg(
    hyp: List[str],
    y_true: np.ndarray,
    radbert_ckpt: str,
    device: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Calls RadBert ONCE to extract predicted labels, then computes both
    weighted P/R/F1 (per-finding + averaged) and the global CRG."""
    from shared.metrics.medical.clinical_efficacy import (
        extract_findings, _compute_clinical_metrics_from_arrays, CLINICAL_FINDINGS,
    )
    y_pred = extract_findings(hyp, radbert_ckpt, device)
    assert y_pred.shape == y_true.shape, f"shape mismatch y_pred={y_pred.shape} y_true={y_true.shape}"

    cl = _compute_clinical_metrics_from_arrays(y_true, y_pred)

    # CRG uses dataset-level confusion-matrix totals (sum across volumes and 18 labels)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    crg = crg_score_with_components(TP, FP, FN, TN)

    return {
        "precision": cl["precision"],
        "recall":    cl["recall"],
        "f1":        cl["f1"],
        "per_finding": cl["detailed"],
        "crg":       crg["crg"],
        "confusion": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
        "crg_components": {k: v for k, v in crg.items() if k != "crg"},
    }, y_pred


def default_clinical_labels_path(metrics_path: str) -> Path:
    return Path(metrics_path).parent / "clinical_labels.jsonl"


def save_clinical_labels(
    *,
    jsonl_path: Path,
    volume_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    from shared.metrics.medical.clinical_efficacy import CLINICAL_FINDINGS

    if y_true.shape != y_pred.shape:
        raise ValueError(f"clinical label shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")
    if y_true.shape[0] != len(volume_ids):
        raise ValueError(f"volume/label length mismatch: volumes={len(volume_ids)} labels={y_true.shape[0]}")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as f:
        for volume_id, true_row, pred_row in zip(volume_ids, y_true, y_pred, strict=True):
            f.write(
                json.dumps(
                    {
                        "volume_id": volume_id,
                        "y_true": true_row.astype(int).tolist(),
                        "y_pred": pred_row.astype(int).tolist(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    npz_path = jsonl_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        volume_ids=np.asarray(volume_ids, dtype=str),
        y_true=y_true.astype(np.int8),
        y_pred=y_pred.astype(np.int8),
        findings=np.asarray(CLINICAL_FINDINGS, dtype=str),
    )
    return {
        "jsonl": str(jsonl_path),
        "npz": str(npz_path),
        "count": int(len(volume_ids)),
        "findings": list(CLINICAL_FINDINGS),
    }


def main():
    args = parse_args()
    volume_ids, hyp, ref = load_predictions(args.pred)
    print(f"Loaded {len(volume_ids)} predictions from {args.pred}")

    out: Dict[str, object] = {
        "n_predictions": len(volume_ids),
        "input_file": args.pred,
    }

    if not args.no_text:
        print("\nComputing text metrics ...")
        out["text"] = compute_text_metrics(hyp, ref)
        for k, v in out["text"].items():
            print(f"  {k:12s} = {v:.4f}")

    if not args.no_clinical:
        print(f"\nLoading ground-truth labels from {args.labels_csv} ...")
        y_true = lookup_labels(volume_ids, args.labels_csv)
        print(f"Running RadBert classifier on {len(hyp)} hypotheses ...")
        clinical, y_pred = compute_clinical_and_crg(hyp, y_true, args.radbert, args.device)
        out["clinical"] = clinical
        c = out["clinical"]
        print(f"\n  Clinical F1     = {c['f1']:.4f}")
        print(f"  Clinical Prec   = {c['precision']:.4f}")
        print(f"  Clinical Recall = {c['recall']:.4f}")
        print(f"  CRG Score       = {c['crg']:.4f}")
        print(f"  Confusion (TP/FP/FN/TN): "
              f"{c['confusion']['TP']}/{c['confusion']['FP']}/"
              f"{c['confusion']['FN']}/{c['confusion']['TN']}")
        if not args.no_save_clinical_labels:
            labels_out = Path(args.clinical_labels_out) if args.clinical_labels_out else default_clinical_labels_path(args.out)
            out["clinical_labels"] = save_clinical_labels(
                jsonl_path=labels_out,
                volume_ids=volume_ids,
                y_true=y_true,
                y_pred=y_pred,
            )
            print(f"  Clinical labels  = {out['clinical_labels']['jsonl']}")

    out["summary"] = make_flat_summary(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
