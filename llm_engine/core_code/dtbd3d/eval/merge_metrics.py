"""
merge_metrics.py — Combine multiple metrics JSON files into one summary.

Each input JSON is one of:
    metrics_fast.json    (text + clinical + CRG, from eval_fast.py)
    metrics_green.json   (GREEN, from eval_slow_green.py)
    metrics_llama.json   (Llama-Score, from eval_slow_llama.py)

Outputs a single JSON keyed by source. Each source's structure is preserved
verbatim under that key plus a top-level `summary` block extracting the
headline numbers.

Usage:
    python merge_metrics.py metrics_fast.json metrics_green.json --out metrics_final.json
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="One or more JSON files")
    p.add_argument("--out", required=True)
    return p.parse_args()


def classify_source(blob: Dict[str, Any]) -> str:
    if "green" in blob: return "green"
    if "llama_score" in blob: return "llama"
    summary = blob.get("summary") if isinstance(blob.get("summary"), dict) else {}
    if (
        "clinical" in blob
        or "text" in blob
        or "clinical_f1" in blob
        or "clinical_crg" in blob
        or "clinical_f1" in summary
        or "clinical_crg" in summary
    ):
        return "fast"
    return "unknown"


def nested(mapping: Dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first(mapping: Dict[str, Any], *paths: tuple[str, ...] | str) -> Any:
    for path in paths:
        keys = (path,) if isinstance(path, str) else path
        value = nested(mapping, *keys)
        if value is not None:
            return value
    return None


def number(mapping: Dict[str, Any], *paths: tuple[str, ...] | str) -> float | None:
    value = first(mapping, *paths)
    if value is None:
        return None
    return float(value)


def make_summary(merged: Dict[str, Any]) -> Dict[str, float]:
    """Pull the headline numbers we care about into a single flat dict."""
    s: Dict[str, float] = {}
    fast = merged.get("fast", {})
    for out_key, paths in {
        "clinical_f1": (("clinical", "f1"), "clinical_f1", ("summary", "clinical_f1")),
        "clinical_precision": (("clinical", "precision"), "clinical_precision", "precision", ("summary", "clinical_precision")),
        "clinical_recall": (("clinical", "recall"), "clinical_recall", "recall", ("summary", "clinical_recall")),
        "crg": (("clinical", "crg"), "clinical_crg", "crg", "crg_score", ("summary", "clinical_crg"), ("summary", "crg")),
    }.items():
        value = number(fast, *paths)
        if value is not None:
            s[out_key] = value
    for k in ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider"):
        value = number(fast, ("text", k), k, ("summary", k))
        if value is not None:
            s[k] = value
    green = merged.get("green", {})
    if "green" in green and isinstance(green["green"], dict):
        # GREEN class outputs vary; surface anything that looks like a score
        for k, v in green["green"].items():
            if isinstance(v, (int, float)) and "score" in k.lower():
                s[f"green_{k}"] = float(v)
    llama = merged.get("llama", {})
    if "llama_score" in llama and isinstance(llama["llama_score"], dict):
        for k, v in llama["llama_score"].items():
            if isinstance(v, (int, float)):
                s[f"llama_{k}"] = float(v)
    return s


def main():
    args = parse_args()
    merged: Dict[str, Any] = {}
    for path in args.inputs:
        blob = json.load(open(path))
        src = classify_source(blob)
        if src in merged:
            print(f"WARN: duplicate source '{src}' (keeping last: {path})")
        merged[src] = blob

    final = {
        "sources": list(merged.keys()),
        "summary": make_summary(merged),
        "details": merged,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(final, f, indent=2, default=float)

    print("Summary:")
    for k, v in final["summary"].items():
        print(f"  {k:24s} = {v:.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
