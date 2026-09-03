"""
eval_slow_llama.py — Standalone Llama-Score (LLM-as-judge).

Uses a (large) Llama model to score generated reports against references on a
0-10 scale. Expensive to run; not part of the training loop. Default model is
Meta-Llama-3-70B-Instruct via vLLM tensor parallelism.

Input JSONL format (one record per line):
    {"volume_id": "...", "generated_text": "...", "reference_text": "..."}

Usage:
    python eval_slow_llama.py \\
        --pred predictions.jsonl \\
        --model ./checkpoints/Meta-Llama-3-70B-Instruct \\
        --tensor-parallel-size 4 \\
        --out metrics_llama.json
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
    p.add_argument("--model", default="./checkpoints/Meta-Llama-3-70B-Instruct")
    p.add_argument("--task-type", default="report_generation")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--out", required=True)
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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    volume_ids, hyp, ref = load_predictions(args.pred)
    print(f"Loaded {len(volume_ids)} predictions from {args.pred}")

    from shared.metrics.medical.llama_score import compute_llama_score
    print(f"Running Llama-Score with model {args.model} "
          f"(tp={args.tensor_parallel_size}, T={args.temperature}) ...")
    result = compute_llama_score(
        hypotheses=hyp,
        references=ref,
        task_type=args.task_type,
        model_path=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    out = {
        "n_predictions": len(volume_ids),
        "input_file": args.pred,
        "llama_model": args.model,
        "llama_score": result,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
