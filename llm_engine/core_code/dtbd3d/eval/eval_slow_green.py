"""
eval_slow_green.py — Standalone GREEN score (Ostmeier et al., 2024).

GREEN runs a fine-tuned LLM over (reference, hypothesis) pairs and is the most
expensive metric in the pipeline (typically minutes-to-an-hour for 1k samples
on a single GPU). Run this AFTER predictions are saved, never inside the
training loop.

Input JSONL format (one record per line):
    {"volume_id": "...", "generated_text": "...", "reference_text": "..."}

Usage:
    python eval_slow_green.py \\
        --pred predictions.jsonl \\
        --green-model StanfordAIMI/GREEN-radllama2-7b \\
        --out metrics_green.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

SHARED_ROOT = "<an external repo, now vendored here as core_code/vendor>"
if SHARED_ROOT not in sys.path:
    sys.path.insert(0, SHARED_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True)
    p.add_argument("--green-model", default="StanfordAIMI/GREEN-radllama2-7b",
                   help="HuggingFace model name or local path")
    p.add_argument("--output-dir", default="/tmp/green_work",
                   help="Working directory for GREEN's intermediate files")
    p.add_argument("--out", required=True)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def load_predictions(jsonl_path: str) -> Tuple[List[str], List[str], List[str]]:
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
    return vols, hyp, ref


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    volume_ids, hyp, ref = load_predictions(args.pred)
    print(f"Loaded {len(volume_ids)} predictions from {args.pred}")

    from shared.metrics.medical.green_score.green import GREEN
    print(f"Initialising GREEN with model: {args.green_model}")
    scorer = GREEN(model_name=args.green_model, output_dir=args.output_dir, cpu=args.cpu)

    print(f"Scoring {len(hyp)} (reference, hypothesis) pairs ...")
    result = scorer(refs=ref, hyps=hyp)

    # The GREEN class returns a tuple/dict depending on version; serialise robustly.
    if hasattr(result, "_asdict"):
        result = result._asdict()
    if not isinstance(result, dict):
        result = {"green_raw": result}

    out = {
        "n_predictions": len(volume_ids),
        "input_file": args.pred,
        "green_model": args.green_model,
        "green": result,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
