"""Offline evaluation for one saved BTB3D report-generation checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dtbd3d.training.constants import CORE_CODE_ROOT
from dtbd3d.training.data import BTB3DDataCollator
from dtbd3d.training.models import BTB3DReportGenerator
from dtbd3d.training.train.eval_callback import PeriodicEvalRunner
from dtbd3d.training.train.main import _path, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training YAML used for this run")
    parser.add_argument("--checkpoint-dir", required=True, help="Saved checkpoint directory, e.g. checkpoints/step_4786")
    parser.add_argument("--out-dir", required=True, help="Output directory for predictions, metrics, and summary")
    parser.add_argument("--label", required=True, help="Short label written to metadata")
    parser.add_argument("--step", type=int, help="Checkpoint step. Defaults to parsing step_<N> from checkpoint-dir")
    parser.add_argument("--n-valid", type=int, default=500, help="Number of valid samples to evaluate")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Generation batch size")
    parser.add_argument("--device", default="cuda:0", help="Torch/eval_fast device")
    parser.add_argument("--max-new-tokens", type=int, help="Override config evaluation.max_new_tokens")
    parser.add_argument("--repetition-penalty", type=float, help="Override config evaluation.repetition_penalty")
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def _parse_step(checkpoint_dir: Path, explicit_step: int | None) -> int:
    if explicit_step is not None:
        return explicit_step
    name = checkpoint_dir.name
    if name.startswith("step_"):
        return int(name.removeprefix("step_"))
    raise ValueError(f"could not infer step from checkpoint-dir name: {checkpoint_dir}")


def _load_tokenizer(checkpoint_dir: Path, model_name_or_path: str) -> Any:
    """Prefer the saved tokenizer because it preserves added BTB3D token ids."""

    try:
        return AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)


def main() -> None:
    args = parse_args()
    config_path = _path(args.config)
    checkpoint_dir = _path(args.checkpoint_dir)
    out_dir = _path(args.out_dir)
    step = _parse_step(checkpoint_dir, args.step)
    if args.n_valid <= 0:
        raise ValueError(f"--n-valid must be positive, got {args.n_valid}")
    if args.eval_batch_size <= 0:
        raise ValueError(f"--eval-batch-size must be positive, got {args.eval_batch_size}")

    raw = yaml.safe_load(config_path.read_text())
    raw["data"]["valid"]["max_samples"] = args.n_valid
    eval_cfg = raw["evaluation"]
    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else eval_cfg["max_new_tokens"])
    repetition_penalty = float(
        args.repetition_penalty if args.repetition_penalty is not None else eval_cfg["repetition_penalty"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    (out_dir / "evaluations").mkdir(exist_ok=True)

    device = torch.device(args.device)
    tokenizer = _load_tokenizer(checkpoint_dir, raw["model"]["model_name_or_path"])
    model = BTB3DReportGenerator.from_pretrained(
        checkpoint_dir=checkpoint_dir,
        base_model_name_or_path=raw["model"]["model_name_or_path"],
        torch_dtype=_torch_dtype(args.torch_dtype),
        device=device,
    )
    model.eval()

    valid_dataset = build_dataset(raw, tokenizer, "valid", None)
    collator = BTB3DDataCollator(pad_token_id=tokenizer.pad_token_id)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    runner = PeriodicEvalRunner(
        eval_loader=valid_loader,
        eval_fast_path=CORE_CODE_ROOT / "dtbd3d" / "eval" / "eval_fast.py",
        converter_path=CORE_CODE_ROOT / "dtbd3d" / "eval" / "btb3d_to_metrics_jsonl.py",
        reports_csv=_path(raw["data"]["valid"]["reports_csv_path"]),
        labels_csv=_path(eval_cfg["labels_csv"]),
        radbert_checkpoint=_path(eval_cfg["radbert_checkpoint"]),
        device=args.device,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        metrics = runner.run(step, model, tokenizer, out_dir)
    elapsed = time.perf_counter() - start
    peak_mem_gb = None
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    summary = {
        "label": args.label,
        "step": step,
        "checkpoint_dir": str(checkpoint_dir),
        "config": str(config_path),
        "out_dir": str(out_dir),
        "n_valid_requested": args.n_valid,
        "n_valid_loaded": len(valid_dataset),
        "eval_batch_size": args.eval_batch_size,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "elapsed_sec": elapsed,
        "sec_per_sample": elapsed / max(1, len(valid_dataset)),
        "peak_gpu_mem_gb": peak_mem_gb,
        "metrics": metrics,
        "prediction_path": str(out_dir / "predictions" / f"step_{step}.jsonl"),
        "metrics_path": str(out_dir / "evaluations" / f"step_{step}_metrics.json"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
