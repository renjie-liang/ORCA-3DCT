"""
btb3d_to_metrics_jsonl.py — Convert BTB3D `multigpu_llava_eval.py` output
into the standard JSONL accepted by `eval_fast.py` / `eval_slow_*.py`.

BTB3D writes:
    {"image": "valid_x_a_1.nii_embedded.npz",
     "conversations_out": [{"id": ..., "question": ..., "answer": "..."}]}

This script:
  1. Strips the `.nii_embedded.npz` suffix to get the canonical volume id
     (e.g. "valid_1_a_1").
  2. Joins on the CT-RATE radiology_text_reports CSV to attach ground-truth.
     By default the BTB3D report-gen target is `Findings_EN + Impressions_EN`
     (matches the field order in the CT-RATE training split).
  3. Writes one record per (volume, conversation-with-non-empty-answer) pair:
        {"volume_id": ..., "generated_text": ..., "reference_text": ...,
         "question": ...}

Usage:
    python btb3d_to_metrics_jsonl.py \\
        --btb3d-jsonl predictions_btb3d_format.jsonl \\
        --reports-csv .../validation_reports.csv \\
        --out predictions_metrics_format.jsonl
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--btb3d-jsonl", required=True)
    p.add_argument("--reports-csv", required=True,
                   help="CT-RATE validation_reports.csv (or train_reports.csv)")
    p.add_argument("--out", required=True)
    p.add_argument("--reference-fields", default="Findings_EN,Impressions_EN",
                   help="Comma-separated CSV columns to concatenate as the reference report")
    p.add_argument("--report-task-only", action="store_true", default=True,
                   help="Keep only the report-generation conversation turn (default: True)")
    p.add_argument("--keep-empty-answers", action="store_true",
                   help="Keep records where answer is empty (default: drop)")
    return p.parse_args()


def normalise_volume_id(image_field: str) -> str:
    """`valid_1_a_1.nii_embedded.npz` -> `valid_1_a_1`.

    Also handles `.npz`, `.nii.gz` and basename-vs-path inputs.
    """
    name = Path(image_field).name
    name = re.sub(r"\.nii_embedded\.npz$", "", name)
    name = re.sub(r"\.nii\.gz$", "", name)
    name = re.sub(r"\.npz$", "", name)
    return name


def load_reports(csv_path: str, fields: List[str]) -> Dict[str, str]:
    """Returns volume_id (without .nii.gz) -> concatenated reference text."""
    out: Dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vol = normalise_volume_id(row["VolumeName"])
            parts = []
            for col in fields:
                v = (row.get(col) or "").strip()
                if v:
                    parts.append(v)
            out[vol] = " ".join(parts)
    return out


def convert_btb3d_jsonl(
    btb3d_jsonl: str,
    reports_csv: str,
    out_jsonl: str,
    reference_fields: List[str],
    report_task_only: bool = True,
    keep_empty_answers: bool = False,
) -> Dict[str, int]:
    """Convert BTB3D raw output JSONL into the canonical evaluation JSONL.

    Returns summary counts for lightweight logging / wrapper use.
    """
    reports = load_reports(reports_csv, reference_fields)
    print(f"Loaded {len(reports)} ground-truth reports from {reports_csv}")

    n_in = n_out = n_missing = n_empty = 0
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(btb3d_jsonl) as fin, open(out_jsonl, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            vol = normalise_volume_id(rec["image"])
            ref = reports.get(vol)
            if ref is None:
                n_missing += 1
                continue

            for conv in rec.get("conversations_out", []):
                ans = (conv.get("answer") or "").strip()
                if not ans and not keep_empty_answers:
                    n_empty += 1
                    continue
                question = (conv.get("question") or "").strip()
                if report_task_only and "report" not in question.lower():
                    continue

                fout.write(json.dumps({
                    "volume_id": vol,
                    "generated_text": ans,
                    "reference_text": ref,
                    "question": question,
                    "source_image": rec["image"],
                }) + "\n")
                n_out += 1

    print(f"Read {n_in} BTB3D records | wrote {n_out} metric records | "
          f"missing-gt {n_missing} | dropped-empty {n_empty}")
    if n_out == 0:
        raise SystemExit("No output records produced — check filters and field names.")
    print(f"Wrote {out_jsonl}")
    return {
        "n_in": n_in,
        "n_out": n_out,
        "n_missing": n_missing,
        "n_empty": n_empty,
    }


def main():
    args = parse_args()
    fields = [s.strip() for s in args.reference_fields.split(",") if s.strip()]
    convert_btb3d_jsonl(
        btb3d_jsonl=args.btb3d_jsonl,
        reports_csv=args.reports_csv,
        out_jsonl=args.out,
        reference_fields=fields,
        report_task_only=args.report_task_only,
        keep_empty_answers=args.keep_empty_answers,
    )


if __name__ == "__main__":
    main()
