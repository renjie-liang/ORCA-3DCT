#!/usr/bin/env python3
"""Evaluate one BTB3D tokenizer checkpoint on cached CT-RATE tensors.

This is an eval-only wrapper around the periodic reconstruction evaluator used
by `tiny_recon_train.py`. It keeps base/V2/V3 checkpoint comparisons on the
same cached .npy input path and metric implementation as training-time eval.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dtbd3d.encoder_training.pipeline_config import load_pipeline_config
from dtbd3d.encoder_training.tiny_recon_train import (
    load_model,
    prepare_periodic_eval,
    run_periodic_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Fine-tune YAML config used for data/model defaults.")
    parser.add_argument("--checkpoint", required=True, help="Tokenizer checkpoint to evaluate.")
    parser.add_argument("--out-dir", required=True, help="Output directory for reconstruction eval artifacts.")
    parser.add_argument("--label", default="", help="Human-readable run label written to metadata.")
    parser.add_argument("--n-valid", type=int, default=0, help="Number of validation volumes. 0 uses config validation.n_valid.")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Eval batch size. 0 uses config validation.eval_batch_size.")
    parser.add_argument("--device", default="", help="Override config validation.device, e.g. cuda:0.")
    parser.add_argument("--step", type=int, default=-1, help="Step label for output directory. Default parses checkpoint name.")
    parser.add_argument("--save-token-ids", action="store_true", help="Save flattened quantizer token IDs.")
    parser.add_argument("--save-viz", action="store_true", help="Save reconstruction PNGs.")
    parser.add_argument("--viz-n", type=int, default=0, help="Number of PNGs to save when --save-viz is set.")
    parser.add_argument("--show-model-stdout", action="store_true")
    return parser.parse_args()


def parse_checkpoint_step(path: Path) -> int:
    match = re.search(r"tokenizer_step_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return 0


def write_metadata(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metadata.json").write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    config = load_pipeline_config(config_path)
    n_valid = args.n_valid if args.n_valid > 0 else config.validation.n_valid
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else config.validation.eval_batch_size
    device = args.device or config.validation.device
    step = args.step if args.step >= 0 else parse_checkpoint_step(checkpoint)

    if eval_batch_size < 1:
        raise ValueError(f"eval-batch-size must be >= 1, got {eval_batch_size}")
    if n_valid < 1:
        raise ValueError(f"n-valid must be >= 1, got {n_valid}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    root = Path(__file__).resolve().parents[4]
    eval_args = argparse.Namespace(
        compression=config.train.compression,
        tokenizer_checkpoint=str(checkpoint),
        btb3d_repo=str(root / "Experiment/core_code/btb3d_baseline/encoder-decoder"),
        weights_root=str(root / "./data/btb3d_weights"),
        commitment_cost=config.train.commitment_cost,
        diversity_gamma=config.train.diversity_gamma,
        device=device,
        eval_every=1,
        eval_dir=str(out_dir),
        valid_id_list=str(config.validation.ids_file),
        valid_cache_root=str(config.validation.cache_root),
        valid_cache_split=config.validation.cache_split,
        valid_n=n_valid,
        eval_batch_size=eval_batch_size,
        eval_viz_n=args.viz_n,
        eval_save_viz=args.save_viz,
        eval_save_tokens=args.save_token_ids,
        show_model_stdout=args.show_model_stdout,
    )

    start = time.perf_counter()
    np.random.seed(0)
    torch.manual_seed(0)
    metadata = {
        "label": args.label,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "compression": config.train.compression,
        "n_valid": n_valid,
        "eval_batch_size": eval_batch_size,
        "device": device,
        "save_token_ids": args.save_token_ids,
        "save_viz": args.save_viz,
        "viz_n": args.viz_n,
        "valid_id_list": str(config.validation.ids_file),
        "valid_cache_root": str(config.validation.cache_root),
        "valid_cache_split": config.validation.cache_split,
        "step": step,
    }
    write_metadata(out_dir, metadata)

    periodic_eval_items = prepare_periodic_eval(eval_args)
    model = load_model(eval_args)
    run_periodic_eval(
        model=model,
        args=eval_args,
        eval_items=periodic_eval_items,
        step=step,
        distributed=False,
        rank=0,
        world_size=1,
    )

    metadata["elapsed_sec"] = time.perf_counter() - start
    write_metadata(out_dir, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
