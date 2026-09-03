#!/usr/bin/env python3
"""Run the encoder/tokenizer fine-tune pipeline from one YAML config."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dtbd3d.encoder_training.pipeline_config import PipelineConfig, load_pipeline_config, project_root
from dtbd3d.encoder_training.pipeline_eval import compare_recon_metrics, write_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def make_run_dir(config: PipelineConfig, out_dir_override: str | None) -> Path:
    if out_dir_override:
        return Path(out_dir_override).resolve()
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    run_id = f"{now_tag()}_{config.run.name}_{job_id}"
    return config.run.out_root / "runs" / run_id


def command_text(command: list[str]) -> str:
    return " ".join(shlex_quote(item) for item in command)


def shlex_quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def run_command(command: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = command_text(command)
    print(command_line)
    with log_path.open("w") as handle:
        handle.write(command_line + "\n")
        handle.flush()
        if dry_run:
            handle.write("DRY_RUN=1\n")
            return
        handle.write("STARTED\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        return_code = process.wait()
        handle.write(f"EXIT_CODE={return_code}\n")
        handle.flush()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def pipeline_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    core_code = root / "Experiment" / "core_code"
    env["PYTHONPATH"] = str(core_code) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", f"/tmp/{env.get('USER', 'user')}_matplotlib_{env.get('SLURM_JOB_ID', 'manual')}")
    env.pop("RANK", None)
    env.pop("WORLD_SIZE", None)
    env.pop("LOCAL_RANK", None)
    env.pop("MASTER_ADDR", None)
    env.pop("MASTER_PORT", None)
    return env


def summarize_train_metrics(train_dir: Path) -> dict[str, Any]:
    summaries = {}
    metric_paths = sorted(train_dir.glob("metrics_rank*.csv"))
    if not metric_paths:
        metric_paths = sorted(train_dir.glob("train_metrics*.csv"))
    for path in metric_paths:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            summaries[path.name] = {"rows": 0}
            continue
        step_times = [float(row["step_time_sec"]) for row in rows]
        load_times = [float(row["load_time_sec"]) for row in rows]
        peak_mem = [float(row["peak_gpu_mem_gb"]) for row in rows]
        summaries[path.name] = {
            "rows": len(rows),
            "first_step": int(rows[0]["step"]),
            "last_step": int(rows[-1]["step"]),
            "first_loss": float(rows[0]["loss"]),
            "last_loss": float(rows[-1]["loss"]),
            "mean_step_time_sec": sum(step_times) / len(step_times),
            "mean_load_time_sec": sum(load_times) / len(load_times),
            "max_peak_gpu_mem_gb": max(peak_mem),
        }
    if not summaries:
        raise FileNotFoundError(f"No metrics CSV under {train_dir}")
    summary_path = train_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    return summaries


def torchrun_command(
    config: PipelineConfig,
    train_dir: Path,
    checkpoint_dir: Path,
    periodic_eval_dir: Path,
) -> list[str]:
    train = config.train
    torchrun = shutil.which("torchrun")
    if torchrun:
        command = [torchrun]
    else:
        command = [sys.executable, "-m", "torch.distributed.run"]
    command.extend(
        [
            "--standalone",
            f"--nproc_per_node={train.nproc_per_node}",
            "-m",
            "dtbd3d.encoder_training.tiny_recon_train",
            "--compression",
            train.compression,
            "--out-dir",
            str(train_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--device",
            "cuda",
            "--steps",
            str(train.steps),
            "--lr",
            str(train.lr),
            "--crop-depth",
            str(train.crop_depth),
            "--batch-size",
            str(train.batch_size),
            "--num-workers",
            str(train.num_workers),
            "--prefetch-factor",
            str(train.prefetch_factor),
            "--recon-loss-weight",
            str(train.recon_loss_weight),
            "--entropy-loss-weight",
            str(train.entropy_loss_weight),
            "--quantizer-aux-loss-weight",
            str(train.quantizer_aux_loss_weight),
            "--commitment-cost",
            str(train.commitment_cost),
            "--diversity-gamma",
            str(train.diversity_gamma),
            "--checkpoint-every",
            str(train.checkpoint_every),
            "--log-every",
            str(train.log_every),
            "--metrics-style",
            "rank",
        ]
    )
    command.extend(
        [
            "--cache-root",
            str(train.cache_root),
            "--cached-id-list",
            str(train.ids_file),
            "--cache-split",
            str(train.cache_split),
        ]
    )
    if train.importance_map_dir is not None:
        command.extend(["--importance-map-dir", str(train.importance_map_dir)])
    if train.importance_lambda_uniform != 0.0:
        command.extend(["--importance-lambda-uniform", str(train.importance_lambda_uniform)])
    if train.tokenizer_checkpoint is not None:
        command.extend(["--tokenizer-checkpoint", str(train.tokenizer_checkpoint)])
    if train.eval_every_steps > 0:
        command.extend(
            [
                "--eval-every",
                str(train.eval_every_steps),
                "--eval-dir",
                str(periodic_eval_dir),
                "--valid-id-list",
                str(config.validation.ids_file),
                "--valid-cache-root",
                str(config.validation.cache_root),
                "--valid-cache-split",
                str(config.validation.cache_split),
                "--valid-n",
                str(config.validation.periodic_n_valid),
                "--eval-batch-size",
                str(config.validation.eval_batch_size),
                "--eval-viz-n",
                str(config.validation.periodic_viz_n),
            ]
        )
        if config.validation.save_viz:
            command.append("--eval-save-viz")
        if config.validation.save_tokens:
            command.append("--eval-save-tokens")
    if train.pin_memory:
        command.append("--pin-memory")
    if train.log_rank0_only:
        command.append("--log-rank0-only")
    if train.use_distributed_batch_entropy:
        command.append("--use-distributed-batch-entropy")
    return command


def read_volume_ids(path: Path, count: int) -> list[str]:
    volume_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return volume_ids[:count]


def eval_command(
    config: PipelineConfig,
    volume_ids: list[str],
    out_json: Path,
    viz_dir: Path,
    checkpoint: Path | None,
    token_ids_dir: Path | None = None,
) -> list[str]:
    validation = config.validation
    command = [
        sys.executable,
        "Experiment/core_code/dtbd3d/eval/verify_btb3d_direct_recon.py",
        "--compression",
        config.train.compression,
        "--device",
        validation.device,
        "--out",
        str(out_json),
        "--viz-window",
        str(validation.viz_window),
        "--viz-level",
        str(validation.viz_level),
        "--viz-diff-window",
        str(validation.viz_diff_window),
        "--viz-dpi",
        str(validation.viz_dpi),
    ]
    for volume_id in volume_ids:
        command.extend(["--volume-id", volume_id])
    if checkpoint:
        command.extend(["--tokenizer-checkpoint", str(checkpoint)])
    if validation.save_viz:
        command.extend(["--viz-dir", str(viz_dir)])
    else:
        command.append("--no-viz")
    if token_ids_dir:
        command.extend(["--token-ids-dir", str(token_ids_dir)])
    return command


def write_hardware_snapshot(out_dir: Path, env: dict[str, str]) -> None:
    lines = [
        datetime.now().isoformat(),
        f"cwd={Path.cwd()}",
        f"python={sys.executable}",
        f"SLURM_JOB_ID={env.get('SLURM_JOB_ID', 'manual')}",
    ]
    for command in (
        ["hostname"],
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
        [
            sys.executable,
            "-c",
            "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())",
        ],
    ):
        try:
            result = subprocess.run(command, check=False, text=True, capture_output=True, env=env)
            lines.append("$ " + command_text(command))
            lines.append(result.stdout.strip())
            if result.stderr.strip():
                lines.append(result.stderr.strip())
        except FileNotFoundError:
            lines.append("$ " + command_text(command))
            lines.append("command not found")
    (out_dir / "hardware_snapshot.txt").write_text("\n".join(lines) + "\n")


def write_manifest(
    out_dir: Path,
    config: PipelineConfig,
    summary: dict[str, Any],
    train_command: list[str],
) -> None:
    train = config.train
    validation = config.validation
    manifest = {
        "schema_version": 1,
        "stage": "encoder_finetune",
        "run_name": config.run.name,
        "status": summary["status"],
        "run_dir": summary["run_dir"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "config_dir": summary["config_dir"],
            "logs_dir": summary["logs_dir"],
            "train_metrics_dir": summary["train_metrics_dir"],
            "checkpoint_dir": summary["checkpoint_dir"],
            "reconstruction_eval_dir": summary["eval_dir"],
            "periodic_reconstruction_eval_dir": summary["periodic_eval_dir"],
        },
        "train": {
            "compression": train.compression,
            "cache_root": str(train.cache_root),
            "ids_file": str(train.ids_file),
            "cache_split": train.cache_split,
            "tokenizer_checkpoint": str(train.tokenizer_checkpoint) if train.tokenizer_checkpoint else None,
            "importance_map_dir": str(train.importance_map_dir) if train.importance_map_dir else None,
            "importance_lambda_uniform": train.importance_lambda_uniform,
            "steps": train.steps,
            "lr": train.lr,
            "batch_size": train.batch_size,
            "nproc_per_node": train.nproc_per_node,
            "checkpoint_every": train.checkpoint_every,
            "eval_every_steps": train.eval_every_steps,
        },
        "validation": {
            "ids_file": str(validation.ids_file),
            "cache_root": str(validation.cache_root),
            "cache_split": validation.cache_split,
            "n_valid": validation.n_valid,
            "periodic_n_valid": validation.periodic_n_valid,
            "periodic_viz_n": validation.periodic_viz_n,
            "eval_batch_size": validation.eval_batch_size,
            "run_final_eval": validation.run_final_eval,
            "compare_pretrained": validation.compare_pretrained,
            "save_viz": validation.save_viz,
            "save_tokens": validation.save_tokens,
        },
        "commands": {
            "train": command_text(train_command),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Encoder Fine-tune Pipeline Run",
        "",
        f"- status: {summary['status']}",
        f"- config_dir: {summary['config_dir']}",
        f"- logs_dir: {summary['logs_dir']}",
        f"- train_metrics_dir: {summary['train_metrics_dir']}",
        f"- checkpoint_dir: {summary['checkpoint_dir']}",
        f"- checkpoint: {summary['checkpoint']}",
        f"- reconstruction_eval_dir: {summary['eval_dir']}",
        f"- periodic_eval_dir: {summary['periodic_eval_dir']}",
    ]
    if summary["comparison"]:
        comp = summary["comparison"]["summary"]
        lines.extend(
            [
                f"- delta_mean_ssim: {comp['delta_mean_ssim']:.6f}",
                f"- delta_mean_psnr: {comp['delta_mean_psnr']:.6f}",
                f"- delta_mean_mse: {comp['delta_mean_mse']:.6f}",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def validate_inputs(config: PipelineConfig) -> None:
    if config.train.cache_split not in {"train", "valid"}:
        raise ValueError(f"cache_split must be train or valid, got {config.train.cache_split}")
    if not config.train.cache_root.exists():
        raise FileNotFoundError(config.train.cache_root)
    if not config.train.ids_file.exists():
        raise FileNotFoundError(config.train.ids_file)
    if config.train.tokenizer_checkpoint is not None and not config.train.tokenizer_checkpoint.exists():
        raise FileNotFoundError(config.train.tokenizer_checkpoint)
    if config.train.importance_map_dir is not None and not config.train.importance_map_dir.exists():
        raise FileNotFoundError(config.train.importance_map_dir)
    if not config.validation.ids_file.exists():
        raise FileNotFoundError(config.validation.ids_file)
    if config.validation.cache_split not in {"train", "valid"}:
        raise ValueError(f"validation cache_split must be train or valid, got {config.validation.cache_split}")
    if not config.validation.cache_root.exists():
        raise FileNotFoundError(config.validation.cache_root)
    if config.train.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")
    if config.train.eval_every_steps < 0:
        raise ValueError("eval_every_steps must be >= 0")


def main() -> int:
    args = parse_args()
    root = project_root()
    os.chdir(root)
    config = load_pipeline_config(Path(args.config))
    validate_inputs(config)
    out_dir = make_run_dir(config, args.out_dir)
    config_dir = out_dir / "config"
    logs_dir = out_dir / "logs"
    train_metrics_dir = out_dir / "metrics" / "train"
    checkpoint_dir = out_dir / "checkpoints"
    eval_dir = out_dir / "eval" / "reconstruction"
    eval_logs_dir = eval_dir / "logs"
    eval_viz_dir = eval_dir / "viz"
    periodic_eval_dir = eval_dir / "periodic"
    run_dirs = [
        out_dir,
        config_dir,
        logs_dir,
        train_metrics_dir,
        checkpoint_dir,
        eval_dir,
        eval_logs_dir,
        eval_viz_dir,
        periodic_eval_dir,
    ]
    for path in run_dirs:
        path.mkdir(parents=True, exist_ok=True)
    env = pipeline_env(root)
    write_hardware_snapshot(logs_dir, env)
    config_path = config_dir / "config_resolved.yaml"
    config_path.write_text(yaml.safe_dump(config.raw, sort_keys=False))

    train_command = torchrun_command(config, train_metrics_dir, checkpoint_dir, periodic_eval_dir)
    run_command(train_command, logs_dir / "train.log", env, args.dry_run)
    train_summary = None if args.dry_run else summarize_train_metrics(train_metrics_dir)
    checkpoint = checkpoint_dir / f"tokenizer_step_{config.train.steps}.safetensors"
    if not args.dry_run and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    base_metrics = eval_dir / "base_metrics.json"
    tuned_metrics = eval_dir / "tuned_metrics.json"
    comparison: dict[str, Any] | None = None
    if config.validation.run_final_eval:
        volume_ids = read_volume_ids(config.validation.ids_file, config.validation.n_valid)
        (eval_dir / "volume_ids.txt").write_text("\n".join(volume_ids) + "\n")
        if config.validation.compare_pretrained:
            run_command(
                eval_command(config, volume_ids, base_metrics, eval_viz_dir / "base", None),
                eval_logs_dir / "base_recon.log",
                env,
                args.dry_run,
            )
        run_command(
            eval_command(config, volume_ids, tuned_metrics, eval_viz_dir / "tuned", checkpoint),
            eval_logs_dir / "tuned_recon.log",
            env,
            args.dry_run,
        )
        if config.validation.compare_pretrained and not args.dry_run:
            comparison = compare_recon_metrics(base_metrics, tuned_metrics)
            write_comparison(comparison, eval_dir / "comparison.json", eval_dir / "comparison.csv")
            print(json.dumps(comparison["summary"], indent=2))

    summary = {
        "status": "dry_run" if args.dry_run else "completed",
        "run_dir": str(out_dir),
        "config_dir": str(config_dir),
        "config_path": str(config_path),
        "logs_dir": str(logs_dir),
        "train_metrics_dir": str(train_metrics_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "eval_dir": str(eval_dir),
        "periodic_eval_dir": str(periodic_eval_dir),
        "checkpoint": str(checkpoint),
        "train_summary": train_summary,
        "comparison": comparison,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_manifest(out_dir, config, summary, train_command)
    write_readme(out_dir, summary)
    print(f"OUT={out_dir}")
    print(f"CONFIG={config_path}")
    print(f"LOGS={logs_dir}")
    print(f"TRAIN_METRICS={train_metrics_dir}")
    print(f"CHECKPOINTS={checkpoint_dir}")
    print(f"RECON_EVAL={eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
