"""Configuration loading for encoder fine-tune pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    name: str
    out_root: Path


@dataclass(frozen=True)
class TrainConfig:
    compression: str
    cache_root: Path
    ids_file: Path
    cache_split: str
    tokenizer_checkpoint: Path | None
    importance_map_dir: Path | None
    importance_lambda_uniform: float
    steps: int
    lr: float
    crop_depth: int
    batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    nproc_per_node: int
    checkpoint_every: int
    eval_every_steps: int
    log_every: int
    log_rank0_only: bool
    recon_loss_weight: float
    entropy_loss_weight: float
    quantizer_aux_loss_weight: float
    commitment_cost: float
    diversity_gamma: float
    use_distributed_batch_entropy: bool


@dataclass(frozen=True)
class ValidationConfig:
    ids_file: Path
    cache_root: Path
    cache_split: str
    n_valid: int
    periodic_n_valid: int
    periodic_viz_n: int
    eval_batch_size: int
    run_final_eval: bool
    device: str
    compare_pretrained: bool
    save_viz: bool
    save_tokens: bool
    viz_window: float
    viz_level: float
    viz_diff_window: float
    viz_dpi: int


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig
    train: TrainConfig
    validation: ValidationConfig
    raw: dict[str, Any]


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def resolve_optional_path(raw: dict[str, Any], key: str, root: Path) -> Path | None:
    value = raw.get(key)
    if value is None or str(value) == "":
        return None
    return resolve_path(str(value), root)


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load a YAML pipeline config and resolve all file paths.

    Relative paths are interpreted from the DTBD3D project root. Training data
    is loaded only from the fixed DTBD3D flat cache layout:
    `{cache_root}/{cache_split}/{volume_id}.npy`.
    """
    root = project_root()
    raw = yaml.safe_load(path.read_text())
    run_raw = raw["run"]
    train_raw = raw["train"]
    validation_raw = raw["validation"]

    run = RunConfig(
        name=str(run_raw["name"]),
        out_root=resolve_path(str(run_raw["out_root"]), root),
    )
    train = TrainConfig(
        compression=str(train_raw["compression"]),
        cache_root=resolve_path(str(train_raw["cache_root"]), root),
        ids_file=resolve_path(str(train_raw["ids_file"]), root),
        cache_split=str(train_raw["cache_split"]),
        tokenizer_checkpoint=resolve_optional_path(train_raw, "tokenizer_checkpoint", root),
        importance_map_dir=resolve_optional_path(train_raw, "importance_map_dir", root),
        importance_lambda_uniform=float(train_raw.get("importance_lambda_uniform", 0.0)),
        steps=int(train_raw["steps"]),
        lr=float(train_raw["lr"]),
        crop_depth=int(train_raw["crop_depth"]),
        batch_size=int(train_raw["batch_size"]),
        num_workers=int(train_raw["num_workers"]),
        prefetch_factor=int(train_raw["prefetch_factor"]),
        pin_memory=bool(train_raw["pin_memory"]),
        nproc_per_node=int(train_raw.get("nproc_per_node", 1)),
        checkpoint_every=int(train_raw["checkpoint_every"]),
        eval_every_steps=int(train_raw.get("eval_every_steps", 0)),
        log_every=int(train_raw["log_every"]),
        log_rank0_only=bool(train_raw["log_rank0_only"]),
        recon_loss_weight=float(train_raw["recon_loss_weight"]),
        entropy_loss_weight=float(train_raw["entropy_loss_weight"]),
        quantizer_aux_loss_weight=float(train_raw["quantizer_aux_loss_weight"]),
        commitment_cost=float(train_raw["commitment_cost"]),
        diversity_gamma=float(train_raw["diversity_gamma"]),
        use_distributed_batch_entropy=bool(train_raw.get("use_distributed_batch_entropy", False)),
    )
    validation = ValidationConfig(
        ids_file=resolve_path(str(validation_raw["ids_file"]), root),
        cache_root=resolve_path(str(validation_raw.get("cache_root", train_raw["cache_root"])), root),
        cache_split=str(validation_raw.get("cache_split", "valid")),
        n_valid=int(validation_raw["n_valid"]),
        periodic_n_valid=int(validation_raw.get("periodic_n_valid", validation_raw["n_valid"])),
        periodic_viz_n=int(validation_raw.get("periodic_viz_n", 0)),
        eval_batch_size=int(validation_raw.get("eval_batch_size", 1)),
        run_final_eval=bool(validation_raw.get("run_final_eval", True)),
        device=str(validation_raw["device"]),
        compare_pretrained=bool(validation_raw["compare_pretrained"]),
        save_viz=bool(validation_raw["save_viz"]),
        save_tokens=bool(validation_raw.get("save_tokens", False)),
        viz_window=float(validation_raw.get("viz_window", 1000.0)),
        viz_level=float(validation_raw.get("viz_level", 0.0)),
        viz_diff_window=float(validation_raw["viz_diff_window"]),
        viz_dpi=int(validation_raw["viz_dpi"]),
    )
    return PipelineConfig(run=run, train=train, validation=validation, raw=raw)
