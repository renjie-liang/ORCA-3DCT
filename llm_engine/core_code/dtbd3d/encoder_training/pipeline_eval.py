"""Evaluation helpers for encoder fine-tune pipeline runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compare_recon_metrics(base_path: Path, candidate_path: Path) -> dict[str, Any]:
    base = load_metrics(base_path)
    candidate = load_metrics(candidate_path)
    base_rows = {row["volume_id"]: row for row in base["rows"]}
    candidate_rows = {row["volume_id"]: row for row in candidate["rows"]}
    volume_ids = sorted(set(base_rows) & set(candidate_rows))
    if not volume_ids:
        raise ValueError("No overlapping volume_id values between metric files")

    rows = []
    for volume_id in volume_ids:
        base_row = base_rows[volume_id]
        candidate_row = candidate_rows[volume_id]
        rows.append(
            {
                "volume_id": volume_id,
                "base_ssim": base_row["ssim"],
                "candidate_ssim": candidate_row["ssim"],
                "delta_ssim": candidate_row["ssim"] - base_row["ssim"],
                "base_psnr": base_row["psnr"],
                "candidate_psnr": candidate_row["psnr"],
                "delta_psnr": candidate_row["psnr"] - base_row["psnr"],
                "base_mse": base_row["mse"],
                "candidate_mse": candidate_row["mse"],
                "delta_mse": candidate_row["mse"] - base_row["mse"],
            }
        )

    summary = {
        "n": len(rows),
        "base_mean_ssim": base["mean_ssim"],
        "candidate_mean_ssim": candidate["mean_ssim"],
        "delta_mean_ssim": candidate["mean_ssim"] - base["mean_ssim"],
        "base_mean_psnr": base["mean_psnr"],
        "candidate_mean_psnr": candidate["mean_psnr"],
        "delta_mean_psnr": candidate["mean_psnr"] - base["mean_psnr"],
        "base_mean_mse": base["mean_mse"],
        "candidate_mean_mse": candidate["mean_mse"],
        "delta_mean_mse": candidate["mean_mse"] - base["mean_mse"],
    }
    if "code_usage" in base and "code_usage" in candidate:
        for field in (
            "unique_codes",
            "active_code_fraction",
            "dead_code_fraction",
            "entropy_bits",
            "perplexity",
            "perplexity_fraction",
            "top1_fraction",
            "top10_fraction",
            "top100_fraction",
            "top1pct_fraction",
        ):
            base_value = base["code_usage"][field]
            candidate_value = candidate["code_usage"][field]
            summary[f"base_code_usage_{field}"] = base_value
            summary[f"candidate_code_usage_{field}"] = candidate_value
            summary[f"delta_code_usage_{field}"] = candidate_value - base_value
    return {"summary": summary, "rows": rows}


def write_comparison(comparison: dict[str, Any], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(comparison, indent=2))
    rows = comparison["rows"]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
