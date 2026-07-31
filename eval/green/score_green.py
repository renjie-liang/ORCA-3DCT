"""Compute the GREEN report-generation metric for one report-gen run.

Reads a predictions.jsonl (one JSON object per line with keys
`generated_text` and `reference_text`), runs the StanfordAIMI GREEN
radiology-report scorer, and writes an aggregate + per-sample result.

Fail-fast: no silent defaults. Missing keys / files raise immediately.

Example
-------
python -m eval.green.score_green \
    --pred results/from_collabrator/runs/reportgen_orcafull_b216/reportgen_orcafull_b216__s2/evaluations/step_006032/predictions.jsonl \
    --out  results/green/orcafull_b216_s2_step006032
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.green.green import GREEN


def load_pairs(pred_path: Path):
    refs, hyps, vids = [], [], []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            refs.append(d["reference_text"])
            hyps.append(d["generated_text"])
            vids.append(d["volume_id"])
    return vids, refs, hyps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, type=Path,
                   help="predictions.jsonl for one report-gen eval step")
    p.add_argument("--out", required=True, type=Path,
                   help="output directory for green_metrics.json + per_sample.csv")
    p.add_argument("--model", default="./checkpoints/GREEN-RadLlama2-7b",
                   help="local path (or HF repo id) of the GREEN scorer")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    vids, refs, hyps = load_pairs(args.pred)
    print(f"Loaded {len(refs)} report pairs from {args.pred}")

    # compute_summary_flag=False: skip mpnet-based representative-sentence
    # clustering (mpnet not cached); GREEN mean/std is unaffected.
    scorer = GREEN(args.model, output_dir=str(args.out), compute_summary_flag=False)
    mean, std, green_scores, summary, results_df = scorer(refs, hyps)

    results_df.insert(0, "volume_id", vids)
    per_sample_csv = args.out / "per_sample.csv"
    results_df.to_csv(per_sample_csv, index=False)

    # recover the eval step (epoch) from the predictions path, e.g. step_006032
    step = next((part for part in args.pred.parts if part.startswith("step_")), None)
    metrics = {
        "pred_path": str(args.pred),
        "step": step,
        "epoch_selection": "best CRG on stage-2",
        "model": args.model,
        "n_samples": len(refs),
        "green_mean": float(mean),
        "green_std": float(std),
    }
    with open(args.out / "green_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.out / "green_summary.txt", "w") as f:
        f.write(summary)

    print(f"GREEN mean={mean:.4f} std={std:.4f}  ->  {args.out/'green_metrics.json'}")


if __name__ == "__main__":
    main()
