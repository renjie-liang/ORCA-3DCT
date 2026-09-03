"""Summarize offline report-generation checkpoint eval directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, help="Directory containing <label>_step<step>/summary.json")
    parser.add_argument("--out-csv", help="Summary CSV path. Defaults to <eval-root>/summary.csv")
    parser.add_argument("--plot", help="Optional trend figure path")
    return parser.parse_args()


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first(mapping: dict[str, Any], *paths: tuple[str, ...] | str) -> Any:
    for path in paths:
        keys = (path,) if isinstance(path, str) else path
        value = _nested(mapping, *keys)
        if value is not None:
            return value
    return None


def _number(mapping: dict[str, Any], *paths: tuple[str, ...] | str) -> float | None:
    value = _first(mapping, *paths)
    if value is None:
        return None
    return float(value)


def _integer(mapping: dict[str, Any], *paths: tuple[str, ...] | str) -> int | None:
    value = _first(mapping, *paths)
    if value is None:
        return None
    return int(float(value))


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    """Read nested, flat, and merged ReportGen metric schemas."""

    return {
        "n_predictions": _integer(
            metrics,
            "n_predictions",
            "n",
            "validation_metric_samples",
            ("summary", "n_predictions"),
            ("summary", "validation_metric_samples"),
            ("details", "fast", "n_predictions"),
            ("details", "fast", "n"),
        ),
        "clinical_f1": _number(
            metrics,
            ("clinical", "f1"),
            "clinical_f1",
            ("summary", "clinical_f1"),
            ("details", "fast", "clinical", "f1"),
        ),
        "precision": _number(
            metrics,
            ("clinical", "precision"),
            "clinical_precision",
            "precision",
            ("summary", "clinical_precision"),
            ("details", "fast", "clinical", "precision"),
        ),
        "recall": _number(
            metrics,
            ("clinical", "recall"),
            "clinical_recall",
            "recall",
            ("summary", "clinical_recall"),
            ("details", "fast", "clinical", "recall"),
        ),
        "crg": _number(
            metrics,
            ("clinical", "crg"),
            "clinical_crg",
            "crg",
            "crg_score",
            ("summary", "clinical_crg"),
            ("summary", "crg"),
            ("details", "fast", "clinical", "crg"),
        ),
        "bleu_1": _number(metrics, ("text", "bleu_1"), "bleu_1", ("summary", "bleu_1"), ("details", "fast", "text", "bleu_1")),
        "bleu_4": _number(metrics, ("text", "bleu_4"), "bleu_4", ("summary", "bleu_4"), ("details", "fast", "text", "bleu_4")),
        "rouge_l": _number(metrics, ("text", "rouge_l"), "rouge_l", ("summary", "rouge_l"), ("details", "fast", "text", "rouge_l")),
        "meteor": _number(metrics, ("text", "meteor"), "meteor", ("summary", "meteor"), ("details", "fast", "text", "meteor")),
    }


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    out_csv = Path(args.out_csv) if args.out_csv else eval_root / "summary.csv"
    rows: list[dict[str, object]] = []
    for summary_path in sorted(eval_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        metrics = summary["metrics"]
        normalized = _normalize_metrics(metrics)
        row = {
            "label": summary["label"],
            "step": int(summary["step"]),
            "n_predictions": int(normalized["n_predictions"] or summary["n_valid_loaded"]),
            "elapsed_sec": float(summary["elapsed_sec"]),
            "sec_per_sample": float(summary["sec_per_sample"]),
            "peak_gpu_mem_gb": summary.get("peak_gpu_mem_gb"),
            "clinical_f1": normalized["clinical_f1"],
            "precision": normalized["precision"],
            "recall": normalized["recall"],
            "crg": normalized["crg"],
            "bleu_1": normalized["bleu_1"],
            "bleu_4": normalized["bleu_4"],
            "rouge_l": normalized["rouge_l"],
            "meteor": normalized["meteor"],
            "summary_path": str(summary_path),
        }
        rows.append(row)
    rows.sort(key=lambda r: (str(r["label"]), int(r["step"])))
    if not rows:
        raise SystemExit(f"no summary.json files found under {eval_root}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_csv)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = Path(args.plot)
        metric_specs = [
            ("clinical_f1", "Clinical F1"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("bleu_1", "BLEU-1"),
            ("bleu_4", "BLEU-4"),
            ("crg", "CRG"),
        ]
        labels = ["baseline", "c0", "c025"]
        colors = {"baseline": "#4c78a8", "c0": "#f58518", "c025": "#54a24b"}
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, (key, title) in zip(axes.ravel(), metric_specs):
            for label in labels:
                label_rows = [row for row in rows if row["label"] == label and row[key] is not None]
                if not label_rows:
                    continue
                ax.plot(
                    [int(row["step"]) for row in label_rows],
                    [float(row[key]) for row in label_rows],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=label,
                    color=colors[label],
                )
            ax.set_title(title)
            ax.set_xlabel("checkpoint step")
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="x", rotation=30)
        axes.ravel()[0].legend(loc="best", fontsize=9)
        fig.suptitle("Report Generation Checkpoint Eval Trends")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=200)
        print(plot_path)


if __name__ == "__main__":
    main()
