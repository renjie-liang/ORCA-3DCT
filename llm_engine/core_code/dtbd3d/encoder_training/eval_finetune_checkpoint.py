#!/usr/bin/env python3
"""Evaluate a fine-tuned encoder/tokenizer checkpoint on a validation split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dtbd3d.encoder_training.finetune_pipeline import (
    command_text,
    eval_command,
    pipeline_env,
    read_volume_ids,
    run_command,
)
from dtbd3d.encoder_training.pipeline_config import load_pipeline_config, project_root
from dtbd3d.encoder_training.pipeline_eval import compare_recon_metrics, write_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--n-valid", type=int, default=0, help="0 means all validation IDs.")
    parser.add_argument("--skip-base", action="store_true", help="Only evaluate the fine-tuned checkpoint.")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_out_dir(checkpoint: Path, n_valid: int) -> Path:
    run_dir = checkpoint.parent.parent
    tag = "allvalid" if n_valid <= 0 else f"{n_valid}valid"
    return run_dir / "eval" / "reconstruction" / f"full_{tag}_{datetime.now().strftime('%Y-%m-%d_%H%M')}"


def main() -> int:
    args = parse_args()
    root = project_root()
    os.chdir(root)
    config = load_pipeline_config(Path(args.config))
    checkpoint = Path(args.checkpoint).resolve()
    if not args.dry_run and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    n_valid = args.n_valid if args.n_valid > 0 else 0
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir(checkpoint, n_valid)
    logs_dir = out_dir / "logs"
    viz_dir = out_dir / "viz"
    token_ids_dir = out_dir / "token_ids"
    logs_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    if args.save_token_ids:
        token_ids_dir.mkdir(parents=True, exist_ok=True)

    volume_ids = read_volume_ids(config.validation.ids_file, n_valid) if n_valid > 0 else read_volume_ids(config.validation.ids_file, 10**12)
    (out_dir / "volume_ids.txt").write_text("\n".join(volume_ids) + "\n")

    env = pipeline_env(root)
    validation = config.validation
    validation = type(validation)(
        ids_file=validation.ids_file,
        cache_root=validation.cache_root,
        cache_split=validation.cache_split,
        n_valid=len(volume_ids),
        periodic_n_valid=validation.periodic_n_valid,
        periodic_viz_n=validation.periodic_viz_n,
        eval_batch_size=validation.eval_batch_size,
        run_final_eval=True,
        device=validation.device,
        compare_pretrained=not args.skip_base,
        save_viz=not args.no_viz,
        save_tokens=validation.save_tokens,
        viz_window=validation.viz_window,
        viz_level=validation.viz_level,
        viz_diff_window=validation.viz_diff_window,
        viz_dpi=validation.viz_dpi,
    )
    config = type(config)(run=config.run, train=config.train, validation=validation, raw=config.raw)

    base_metrics = out_dir / "base_metrics.json"
    tuned_metrics = out_dir / "tuned_metrics.json"
    if not args.skip_base:
        run_command(
            eval_command(
                config,
                volume_ids,
                base_metrics,
                viz_dir / "base",
                None,
                token_ids_dir / "base" if args.save_token_ids else None,
            ),
            logs_dir / "base_recon.log",
            env,
            args.dry_run,
        )
    run_command(
        eval_command(
            config,
            volume_ids,
            tuned_metrics,
            viz_dir / "tuned",
            checkpoint,
            token_ids_dir / "tuned" if args.save_token_ids else None,
        ),
        logs_dir / "tuned_recon.log",
        env,
        args.dry_run,
    )

    comparison: dict[str, Any] | None = None
    if not args.skip_base and not args.dry_run:
        comparison = compare_recon_metrics(base_metrics, tuned_metrics)
        write_comparison(comparison, out_dir / "comparison.json", out_dir / "comparison.csv")
        print(json.dumps(comparison["summary"], indent=2))

    summary = {
        "status": "dry_run" if args.dry_run else "completed",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint),
        "out_dir": str(out_dir),
        "n_valid": len(volume_ids),
        "save_token_ids": bool(args.save_token_ids),
        "command": command_text(sys.argv),
        "comparison": comparison,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"EVAL_OUT={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
