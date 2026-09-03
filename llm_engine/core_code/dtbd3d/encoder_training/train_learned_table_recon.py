#!/usr/bin/env python3
"""Train Sub-task 6 LFQ-id delta table reconstruction variants.

This runner is intentionally separate from `tiny_recon_train.py`. It reuses
the same cached CT tensors, BTB3D model loader, importance-map weighting, and
token artifact helpers, but keeps the learned-table side work isolated.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from dtbd3d.core.artifact import save_token_row, update_ids
from dtbd3d.core.btb3d_model import expected_token_count
from dtbd3d.diagnostic_token_learning import (
    DiagnosticSupervision,
    SupervisionOutput,
    apply_organ_input_conditioning,
    load_supervision_config,
)
from dtbd3d.encoder_training.debug_utils import (
    assert_finite_tensor,
    export_loaded_batch_debug_json,
    print_hu_stats,
    print_tensor_stats,
    render_input_conditioning_debug_png,
    render_loaded_batch_debug_png,
    render_smoke_debug_png,
)
from dtbd3d.encoder_training.learned_table_checkpoint import (
    load_model,
    project_root,
    save_checkpoints,
)
from dtbd3d.importance_maps.training import crop_depth_tensor, token_weight_maps_to_voxel_batch, weighted_l1_loss
from dtbd3d.learned_vq import (
    LFQDeltaTableAdapter,
    code_usage_metrics,
    delta_table_metrics,
)
from dtbd3d.training.checkpoint import load_full_training_checkpoint
from dtbd3d.training.data.cached_ct import (
    flat_cache_paths,
    make_train_dataloader,
    make_valid_dataloader,
    read_id_list,
)
from dtbd3d.training.distributed import distributed_barrier, init_distributed_runtime
from dtbd3d.training.logging_utils import setup_rank0_logger
from dtbd3d.training.run_io import prepare_run_output, require_existing_paths
from dtbd3d.training.seed import seed_everything



TRAINABLE_MODES = {"table_only", "table_decoder", "encoder_table_decoder"}
LOSS_VARIANTS = {"uniform", "organ_weighted", "organ_sqrt", "organ_clip", "organ_anchor"}
DEFAULT_RECON_VIZ_WINDOW = 1000.0
DEFAULT_RECON_VIZ_LEVEL = 0.0

TRAIN_METRIC_FIELDS = [
    "step",
    "volume_id",
    "loss",
    "recon_loss",
    "aux_loss",
    "disease_loss",
    "disease_pairs",
    "disease_organs",
    "disease_pos",
    "disease_missing_masks",
    "disease_feature_grid",
    "text_loss",
    "text_pairs",
    "text_candidate_pairs",
    "text_supervised_volumes",
    "llama_prefix_loss",
    "llama_prefix_pairs",
    "llama_prefix_target_tokens",
    "quantize_loss",
    "delta_l2_loss",
    "per_sample_entropy_loss",
    "batch_entropy_loss",
    "commitment_loss",
    "importance_weight_mean",
    "importance_weight_min",
    "importance_weight_max",
    "unique_codes",
    "unique_code_ratio",
    "code_usage_entropy",
    "code_usage_perplexity",
    "delta_abs_mean",
    "delta_abs_max",
    "touched_delta_abs_mean",
    "touched_delta_abs_max",
    "step_time_sec",
    "load_time_sec",
    "peak_gpu_mem_gb",
]


@dataclass
class LearnedTableTrainOutput:
    decoded: torch.Tensor
    z_quantized: torch.Tensor
    token_ids: torch.Tensor
    quantized_output: object
    quantize_loss_breakdown: object
    delta_l2_loss: torch.Tensor
    supervision: SupervisionOutput | None


class LearnedTableReconModule(torch.nn.Module):
    """DDP-friendly wrapper around the tokenizer model and delta-table adapter."""

    def __init__(
        self,
        model: torch.nn.Module,
        adapter: LFQDeltaTableAdapter,
        supervision: DiagnosticSupervision | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.supervision = supervision

    def forward(
        self,
        input_data: torch.Tensor,
        entropy_loss_weight: float,
        calculate_quantize_loss: bool,
        use_distributed_batch_entropy: bool | None,
        batch_context: dict[str, Any] | None = None,
    ) -> LearnedTableTrainOutput:
        if batch_context is not None and int(batch_context.get("step", -1)) == 1:
            print("[debug:forward] before LFQDeltaTableAdapter", flush=True)
        out = self.adapter(
            self.model.tokenizer,
            input_data,
            entropy_loss_weight=entropy_loss_weight,
            calculate_quantize_loss=calculate_quantize_loss,
            use_distributed_batch_entropy=use_distributed_batch_entropy,
        )
        if batch_context is not None and int(batch_context.get("step", -1)) == 1:
            print(
                "[debug:forward] after LFQDeltaTableAdapter "
                + json.dumps(
                    {
                        "decoded_shape": list(out.decoded.shape),
                        "z_quantized_shape": list(out.z_quantized.shape),
                        "token_ids_shape": list(out.token_ids.shape),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        supervision_out = None
        if self.supervision is not None and batch_context is not None:
            if int(batch_context.get("step", -1)) == 1:
                print("[debug:forward] before DiagnosticSupervision", flush=True)
            # Expected z_quantized shape:
            #   16x16x8 tokenizer: [B, C, 31, 32, 32]
            #   8x8x8 tokenizer:   [B, C, 31, 64, 64]
            # The disease head pools this grid with organ masks. Text alignment
            # separately merges this grid to [B,31,8,8,C].
            supervision_out = self.supervision(
                z_quantized=out.z_quantized,
                volume_ids=batch_context["volume_ids"],
                importance_weights=batch_context.get("importance_weights"),
                text_batch=batch_context.get("text"),
                disease_batch=batch_context.get("disease"),
                full_shape_dhw=batch_context["full_shape_dhw"],
                crop_depth_value=batch_context["crop_depth_value"],
                step=batch_context["step"],
            )
            if int(batch_context.get("step", -1)) == 1:
                print(
                    "[debug:forward] after DiagnosticSupervision "
                    + json.dumps(
                        {
                            "loss": float(supervision_out.loss.detach().float().cpu()),
                            "components": sorted((supervision_out.components or {}).keys()),
                            "metrics": supervision_out.metrics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        return LearnedTableTrainOutput(
            decoded=out.decoded,
            z_quantized=out.z_quantized,
            token_ids=out.token_ids,
            quantized_output=out.quantized_output,
            quantize_loss_breakdown=out.quantize_loss_breakdown,
            delta_l2_loss=out.delta_l2_loss,
            supervision=supervision_out,
        )


def unwrap_train_module(module: torch.nn.Module) -> LearnedTableReconModule:
    if isinstance(module, DistributedDataParallel):
        return module.module  # type: ignore[return-value]
    return module  # type: ignore[return-value]


def resolve_train_mask_artifact_dir(args: argparse.Namespace) -> Path | None:
    """Resolve the train split mask artifact directory required by organ losses."""
    train_mask_artifact_dir = Path(args.train_mask_artifact_dir) if args.train_mask_artifact_dir else None
    if args.loss_variant != "uniform" and train_mask_artifact_dir is None:
        raise ValueError(f"{args.loss_variant} loss requires reconstruction.mask_artifact_dir")
    if train_mask_artifact_dir is not None and not train_mask_artifact_dir.exists():
        raise FileNotFoundError(train_mask_artifact_dir)
    return train_mask_artifact_dir


def validate_configured_artifacts(args: argparse.Namespace, cached_paths: list[Path]) -> None:
    """Fast-fail on configured checkpoints and artifact roots."""
    required_paths: list[Path] = list(cached_paths)
    if args.resume_checkpoint:
        required_paths.append(Path(args.resume_checkpoint))
    else:
        if args.tokenizer_checkpoint:
            required_paths.append(Path(args.tokenizer_checkpoint))

    supervision = args.supervision_config
    text_config = supervision.text_alignment
    prefix_config = supervision.llama_prefix_prealignment
    if text_config.enabled or prefix_config.enabled:
        for source in text_config.embedding_sources:
            source_root = Path(source.artifact_dir) / source.split
            source_mask_root = Path(source.mask_artifact_dir) / source.split
            required_paths.extend(
                [
                    source_root,
                    source_root / "index.jsonl",
                    source_root / "embeddings.npy",
                    source_root / "encoded_mask.npy",
                    source_mask_root,
                ]
            )
        if not text_config.embedding_sources:
            raise ValueError("text_alignment.embedding_sources is required when text alignment or Llama prefix prealignment is enabled")

    disease_config = supervision.disease_classification
    if disease_config.enabled:
        disease_root = Path(disease_config.artifact_dir) / disease_config.split
        disease_mask_root = Path(disease_config.mask_artifact_dir) / disease_config.split
        required_paths.extend(
            [
                disease_root,
                disease_root / "ids.txt",
                disease_root / "labels.npy",
                disease_root / "valid_mask.npy",
                Path(disease_config.artifact_dir) / "organ_groups.json",
                Path(disease_config.artifact_dir) / "disease_names.json",
                disease_mask_root,
            ]
        )

    if prefix_config.enabled:
        required_paths.append(Path(prefix_config.model_name_or_path))

    require_existing_paths(required_paths, label="configured training artifacts")


def crop_depth(tensor: torch.Tensor, crop_depth_value: int, step: int) -> torch.Tensor:
    return crop_depth_tensor(tensor, crop_depth_value, step)


def maybe_build_importance_weights(
    token_weight_maps: torch.Tensor | None,
    importance_lambda_uniform: float,
    batch_volume_ids: list[str],
    volume: torch.Tensor,
    crop_depth_value: int,
    step: int,
    device: str,
) -> torch.Tensor | None:
    del importance_lambda_uniform, batch_volume_ids
    if token_weight_maps is None:
        return None
    return token_weight_maps_to_voxel_batch(
        token_weight_maps,
        full_shape_dhw=tuple(int(dim) for dim in volume.shape[2:]),
        crop_depth_value=crop_depth_value,
        step=step,
        device=device,
        dtype=torch.float32,
    )


def importance_weight_stats(importance_weights: torch.Tensor | None) -> tuple[float, float, float]:
    if importance_weights is None:
        return 1.0, 1.0, 1.0
    weights = importance_weights.float()
    return as_float(weights.mean()), as_float(weights.min()), as_float(weights.max())


def as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


class TrainMetricsWriter:
    def __init__(self, out_dir: Path, rank: int, *, append: bool) -> None:
        self.path = out_dir / ("train_metrics.csv" if rank == 0 else f"train_metrics_rank{rank}.csv")
        file_exists = self.path.exists()
        if not append or not file_exists:
            with self.path.open("w", newline="") as handle:
                csv.writer(handle).writerow(TRAIN_METRIC_FIELDS)

    def write(self, values: dict[str, Any]) -> None:
        missing = [field for field in TRAIN_METRIC_FIELDS if field not in values]
        if missing:
            raise KeyError(f"missing train metric fields: {missing}")
        with self.path.open("a", newline="") as handle:
            csv.writer(handle).writerow([values[field] for field in TRAIN_METRIC_FIELDS])


@contextlib.contextmanager
def maybe_suppress_model_stdout(show_model_stdout: bool) -> Any:
    if show_model_stdout:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def score_recon(inp_np: np.ndarray, recon_np: np.ndarray, valid_slices) -> dict[str, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    min_d = min(inp_np.shape[0], recon_np.shape[0])
    min_h = min(inp_np.shape[1], recon_np.shape[1])
    min_w = min(inp_np.shape[2], recon_np.shape[2])
    inp_np = inp_np[:min_d, :min_h, :min_w]
    recon_np = np.clip(recon_np[:min_d, :min_h, :min_w], -1.0, 1.0)

    sd, sh, sw = valid_slices
    sd = slice(sd.start, min(sd.stop, min_d))
    sh = slice(sh.start, min(sh.stop, min_h))
    sw = slice(sw.start, min(sw.stop, min_w))
    inp_roi = inp_np[sd, sh, sw]
    recon_roi = recon_np[sd, sh, sw]
    return {
        "ssim": float(structural_similarity(inp_roi, recon_roi, data_range=2.0)),
        "psnr": float(peak_signal_noise_ratio(inp_roi, recon_roi, data_range=2.0)),
        "mse": float(np.mean((inp_roi - recon_roi) ** 2)),
    }


def center_index(slice_obj: slice, upper_bound: int) -> int:
    start = 0 if slice_obj.start is None else max(slice_obj.start, 0)
    stop = upper_bound if slice_obj.stop is None else min(slice_obj.stop, upper_bound)
    return (start + stop) // 2


def render_recon_png(
    volume_id: str,
    inp_np: np.ndarray,
    recon_np: np.ndarray,
    valid_slices,
    metrics: dict[str, float],
    out_png: Path,
    window: float,
    level: float,
    diff_window: float,
    dpi: int,
) -> None:
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    min_d = min(inp_np.shape[0], recon_np.shape[0])
    min_h = min(inp_np.shape[1], recon_np.shape[1])
    min_w = min(inp_np.shape[2], recon_np.shape[2])
    inp_hu = inp_np[:min_d, :min_h, :min_w] * 1000.0
    recon_hu = np.clip(recon_np[:min_d, :min_h, :min_w], -1.0, 1.0) * 1000.0
    diff_hu = recon_hu - inp_hu

    sd, sh, sw = valid_slices
    d_idx = center_index(sd, min_d)
    h_idx = center_index(sh, min_h)
    w_idx = center_index(sw, min_w)
    planes = [
        ("axial", inp_hu[d_idx], recon_hu[d_idx], diff_hu[d_idx]),
        ("coronal", inp_hu[:, h_idx, :], recon_hu[:, h_idx, :], diff_hu[:, h_idx, :]),
        ("sagittal", inp_hu[:, :, w_idx], recon_hu[:, :, w_idx], diff_hu[:, :, w_idx]),
    ]

    vmin = level - window / 2.0
    vmax = level + window / 2.0
    diff_v = diff_window / 2.0
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for row, (plane, inp_slice, recon_slice, diff_slice) in enumerate(planes):
        images = [
            ("input HU", inp_slice, "gray", vmin, vmax),
            ("reconstruction HU", recon_slice, "gray", vmin, vmax),
            ("recon - input HU", diff_slice, "coolwarm", -diff_v, diff_v),
        ]
        for col, (title, image, cmap, lo, hi) in enumerate(images):
            axes[row, col].imshow(image, cmap=cmap, vmin=lo, vmax=hi)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title)
            if col == 0:
                axes[row, col].set_ylabel(plane)
    fig.suptitle(
        f"{volume_id} | SSIM={metrics['ssim']:.4f} "
        f"PSNR={metrics['psnr']:.2f} MSE={metrics['mse']:.5f}"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def prepare_periodic_eval(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.eval_every <= 0:
        return []
    if not args.eval_dir:
        raise ValueError("eval_dir is required when eval_every > 0")
    if not args.valid_id_list:
        raise ValueError("valid_id_list is required when eval_every > 0")
    if not args.valid_cache_root:
        raise ValueError("valid_cache_root is required when eval_every > 0")
    valid_ids = read_id_list(Path(args.valid_id_list))
    if args.valid_n > 0:
        valid_ids = valid_ids[: args.valid_n]
    valid_cache_root = Path(args.valid_cache_root)
    eval_items = [(volume_id, valid_cache_root / args.valid_cache_split / f"{volume_id}.npy") for volume_id in valid_ids]
    for _volume_id, path in eval_items:
        if not path.exists():
            raise FileNotFoundError(path)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.eval_dir) / "volume_ids.txt").write_text("\n".join(valid_ids) + ("\n" if valid_ids else ""))
    return eval_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="", help="Override run output directory.")
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke test using this formal config.")
    parser.add_argument("--steps", type=int, default=0, help="Override train.steps for smoke tests.")
    parser.add_argument("--checkpoint-every", type=int, default=-1, help="Override train.checkpoint_every.")
    parser.add_argument("--eval-every-steps", type=int, default=-1, help="Override train.eval_every_steps.")
    parser.add_argument("--valid-n", type=int, default=-1, help="Override validation.periodic_n_valid for eval smoke tests.")
    parser.add_argument("--resume-checkpoint", default="", help="Resume from a full training checkpoint .safetensors.")
    return parser.parse_args()


def resolve_path(value: str | None, root: Path) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def run_suffix(*, run_name: str, smoke: bool) -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    label = "smoke" if smoke else (job_id or "manual")
    return f"{now_tag()}_{run_name}_{label}"


def load_run_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    root = project_root()
    config_path = Path(cli_args.config)
    raw = yaml.safe_load(config_path.read_text())
    run_raw = raw["run"]
    train_raw = raw["train"]
    validation_raw = raw["validation"]
    learned_raw = raw.get("learned_table", {})
    reconstruction_raw = raw.get("reconstruction", {}) or {}
    default_supervision_split = str(train_raw["cache_split"])
    supervision_config = load_supervision_config(raw, root, default_split=default_supervision_split)
    train_mask_artifact_dir = (
        Path(supervision_config.mask_artifact_dir) / str(train_raw["cache_split"])
        if supervision_config.mask_artifact_dir
        else None
    )

    out_root = resolve_path(str(run_raw["out_root"]), root)
    assert out_root is not None
    if cli_args.out_dir:
        out_dir = Path(cli_args.out_dir).resolve() / run_suffix(run_name=str(run_raw["name"]), smoke=bool(cli_args.smoke))
    else:
        out_dir = out_root / "runs" / run_suffix(run_name=str(run_raw["name"]), smoke=bool(cli_args.smoke))

    trainable_mode = str(learned_raw.get("trainable_mode", train_raw.get("trainable_mode", "table_only")))
    if trainable_mode not in TRAINABLE_MODES:
        raise ValueError(f"unsupported trainable_mode {trainable_mode}; expected one of {sorted(TRAINABLE_MODES)}")

    args = argparse.Namespace(
        config=str(config_path),
        raw_config=raw,
        run_name=str(run_raw["name"]),
        out_dir=str(out_dir),
        smoke=False,
        compression=str(train_raw["compression"]),
        cache_root=str(resolve_path(str(train_raw["cache_root"]), root)),
        cached_id_list=str(resolve_path(str(train_raw["ids_file"]), root)),
        cache_split=str(train_raw["cache_split"]),
        tokenizer_checkpoint=str(resolve_path(str(train_raw.get("tokenizer_checkpoint", "")), root) or ""),
        resume_checkpoint=str(resolve_path(str(cli_args.resume_checkpoint), root) or ""),
        device=str(train_raw.get("device", validation_raw.get("device", "cuda:0"))),
        steps=int(train_raw["steps"]),
        lr=float(train_raw["lr"]),
        seed=int(train_raw.get("seed", 0)),
        crop_depth=int(train_raw["crop_depth"]),
        recon_loss_weight=float(train_raw["recon_loss_weight"]),
        entropy_loss_weight=float(train_raw["entropy_loss_weight"]),
        quantizer_aux_loss_weight=float(train_raw["quantizer_aux_loss_weight"]),
        commitment_cost=float(train_raw["commitment_cost"]),
        diversity_gamma=float(train_raw["diversity_gamma"]),
        use_distributed_batch_entropy=bool(train_raw.get("use_distributed_batch_entropy", False)),
        train_mask_artifact_dir=str(train_mask_artifact_dir or ""),
        importance_lambda_uniform=float(reconstruction_raw.get("importance_lambda_uniform", train_raw.get("importance_lambda_uniform", 0.0))),
        batch_size=int(train_raw["batch_size"]),
        num_workers=int(train_raw["num_workers"]),
        prefetch_factor=int(train_raw["prefetch_factor"]),
        pin_memory=bool(train_raw["pin_memory"]),
        shuffle=bool(train_raw.get("shuffle", True)),
        checkpoint_every=int(train_raw["checkpoint_every"]),
        log_every=int(train_raw["log_every"]),
        log_rank0_only=bool(train_raw.get("log_rank0_only", True)),
        show_model_stdout=bool(train_raw.get("show_model_stdout", False)),
        trainable_mode=trainable_mode,
        loss_variant=str(reconstruction_raw.get("loss_variant", learned_raw.get("loss_variant", "uniform"))),
        importance_power=float(reconstruction_raw.get("importance_power", learned_raw.get("importance_power", 1.0))),
        importance_clip_max=float(reconstruction_raw.get("importance_clip_max", learned_raw.get("importance_clip_max", 0.0))),
        organ_extra_weight=float(reconstruction_raw.get("organ_extra_weight", learned_raw.get("organ_extra_weight", 0.5))),
        delta_scale=float(learned_raw.get("delta_scale", 0.1)),
        delta_l2_weight=float(learned_raw.get("delta_l2_weight", 0.0)),
        delta_lr=float(learned_raw.get("delta_lr", train_raw["lr"])),
        decoder_lr=float(learned_raw.get("decoder_lr", train_raw["lr"])),
        encoder_lr=float(learned_raw.get("encoder_lr", train_raw["lr"])),
        preserve_encoder_ste=bool(learned_raw.get("preserve_encoder_ste", True)),
        supervision_config=supervision_config,
        eval_every=int(train_raw.get("eval_every_steps", 0)),
        eval_dir="",
        valid_id_list=str(resolve_path(str(validation_raw["ids_file"]), root)),
        valid_cache_root=str(resolve_path(str(validation_raw.get("cache_root", train_raw["cache_root"])), root)),
        valid_cache_split=str(validation_raw.get("cache_split", "valid")),
        valid_n=int(validation_raw.get("periodic_n_valid", validation_raw.get("n_valid", 0))),
        eval_batch_size=int(validation_raw.get("eval_batch_size", 1)),
        eval_viz_n=int(validation_raw.get("periodic_viz_n", 0)),
        eval_save_viz=bool(validation_raw.get("save_viz", False)),
        eval_save_tokens=bool(validation_raw.get("save_tokens", False)),
        viz_diff_window=float(validation_raw.get("viz_diff_window", 500.0)),
        viz_dpi=int(validation_raw.get("viz_dpi", 150)),
    )
    args.eval_dir = str(Path(args.out_dir) / "periodic_eval")
    if args.loss_variant not in LOSS_VARIANTS:
        raise ValueError(f"unsupported loss_variant {args.loss_variant}; expected one of {sorted(LOSS_VARIANTS)}")
    if cli_args.smoke:
        args.smoke = True
        args.steps = 2
        args.checkpoint_every = 0
        args.eval_every = 0
        args.valid_n = 0
        args.eval_viz_n = 0
        args.eval_save_viz = False
        args.eval_save_tokens = False
        args.log_every = 1
        args.shuffle = False
        args.num_workers = 0
        args.prefetch_factor = 2
    if cli_args.steps > 0:
        args.steps = cli_args.steps
    if cli_args.checkpoint_every >= 0:
        args.checkpoint_every = cli_args.checkpoint_every
    if cli_args.eval_every_steps >= 0:
        args.eval_every = cli_args.eval_every_steps
    if cli_args.valid_n >= 0:
        args.valid_n = cli_args.valid_n
    return args


def write_config(out_dir: Path, args: argparse.Namespace) -> None:
    payload = args.raw_config.copy()
    payload["_resolved"] = {
        "out_dir": args.out_dir,
        "config": args.config,
        "smoke": bool(getattr(args, "smoke", False)),
        "resume_checkpoint": args.resume_checkpoint,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.json").write_text(json.dumps(payload, indent=2))


def set_trainable_modules(model: torch.nn.Module, mode: str) -> None:
    tokenizer = model.tokenizer
    for param in tokenizer.parameters():
        param.requires_grad = False
    if mode in {"table_decoder", "encoder_table_decoder"}:
        for param in tokenizer.decoder.parameters():
            param.requires_grad = True
    if mode == "encoder_table_decoder":
        for param in tokenizer.encoder.parameters():
            param.requires_grad = True


def optimizer_parameter_groups(
    model: torch.nn.Module,
    adapter: LFQDeltaTableAdapter,
    supervision: DiagnosticSupervision | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Build explicit optimizer groups for delta, decoder, and encoder."""
    groups: list[dict[str, Any]] = []
    delta_params = [param for param in adapter.parameters() if param.requires_grad]
    if delta_params:
        groups.append({"params": delta_params, "lr": args.delta_lr, "name": "delta_table"})

    decoder_params = [param for param in model.tokenizer.decoder.parameters() if param.requires_grad]
    if decoder_params:
        groups.append({"params": decoder_params, "lr": args.decoder_lr, "name": "decoder"})

    encoder_params = [param for param in model.tokenizer.encoder.parameters() if param.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr, "name": "encoder"})

    if supervision is not None:
        supervision_params = [param for param in supervision.parameters() if param.requires_grad]
        if supervision_params:
            supervision_lr = float(supervision.config.supervision_lr or args.lr)
            if supervision_lr <= 0:
                raise ValueError("supervision.supervision_lr must be positive when diagnostic supervision has trainable parameters")
            groups.append({"params": supervision_params, "lr": supervision_lr, "name": "diagnostic_supervision"})

    if not groups:
        raise ValueError("no trainable parameters selected")
    return groups


def normalize_importance_weights(weights: torch.Tensor) -> torch.Tensor:
    """Mean-normalize each sample after any loss-shaping transform."""
    means = weights.float().mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    return (weights.float() / means).to(dtype=weights.dtype)


def shape_importance_weights(importance_weights: torch.Tensor | None, args: argparse.Namespace) -> torch.Tensor | None:
    """Apply lightweight RGenMask shaping for Sub-task 6 screens."""
    if importance_weights is None:
        return None
    weights = importance_weights.float().clamp_min(0.0)
    if args.importance_power != 1.0:
        if args.importance_power <= 0.0:
            raise ValueError(f"importance_power must be > 0, got {args.importance_power}")
        weights = weights.clamp_min(1e-6).pow(args.importance_power)
    if args.importance_clip_max > 0.0:
        weights = weights.clamp_max(args.importance_clip_max)
    return normalize_importance_weights(weights).to(dtype=importance_weights.dtype)


def organ_anchor_l1_loss(input_data: torch.Tensor, recon_output: torch.Tensor, weights: torch.Tensor, organ_weight: float) -> torch.Tensor:
    """Uniform L1 plus a positive organ-weight residual term.

    This keeps the uniform reconstruction objective as the anchor and adds
    only the above-uniform part of the organ map. It is meant to test whether
    the top-5% RGenMask benefit can expand without damaging background MSE.
    """
    if organ_weight < 0.0:
        raise ValueError(f"organ_extra_weight must be >= 0, got {organ_weight}")
    if input_data.shape != recon_output.shape:
        raise ValueError(f"input/recon shape mismatch: {input_data.shape} vs {recon_output.shape}")
    if weights.shape[0] != input_data.shape[0] or weights.shape[2:] != input_data.shape[2:]:
        raise ValueError(f"weight shape {weights.shape} is incompatible with input {input_data.shape}")
    error = torch.abs(input_data.float() - recon_output.float())
    organ_extra = (weights.float() - 1.0).clamp_min(0.0)
    return error.mean() + organ_weight * (error * organ_extra).mean()


def reconstruction_loss(
    input_data: torch.Tensor,
    recon_output: torch.Tensor,
    importance_weights: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Select the reconstruction loss variant for learned-table screens."""
    if args.loss_variant == "uniform":
        return torch.nn.functional.l1_loss(input_data, recon_output)
    if importance_weights is None:
        raise ValueError(f"{args.loss_variant} requires reconstruction.mask_artifact_dir")
    if args.loss_variant in {"organ_weighted", "organ_sqrt", "organ_clip"}:
        return weighted_l1_loss(input_data, recon_output, importance_weights)
    if args.loss_variant == "organ_anchor":
        return organ_anchor_l1_loss(input_data, recon_output, importance_weights, args.organ_extra_weight)
    raise ValueError(f"unsupported loss_variant {args.loss_variant}")


def run_periodic_eval(
    model: torch.nn.Module,
    adapter: LFQDeltaTableAdapter,
    args: argparse.Namespace,
    eval_items: list[tuple[str, Path]],
    step: int,
) -> None:
    was_model_training = model.training
    was_adapter_training = adapter.training
    model.eval()
    adapter.eval()
    eval_dir = Path(args.eval_dir)
    step_dir = eval_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = step_dir / "viz" if args.eval_save_viz else None
    token_dir = step_dir / "token_artifact" if args.eval_save_tokens else None
    expected_tokens = expected_token_count(args.compression)
    viz_volume_ids = {volume_id for volume_id, _path in eval_items[: args.eval_viz_n]} if args.eval_viz_n > 0 else None

    rows: list[dict[str, Any]] = []
    token_ids_written: list[str] = []
    viz_written = 0
    valid_loader = make_valid_dataloader(
        [path for _volume_id, path in eval_items],
        batch_size=args.eval_batch_size,
        num_workers=0 if bool(getattr(args, "smoke", False)) else args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
    )
    with torch.no_grad():
        for batch in valid_loader:
            volume_ids = list(batch["volume_id"])
            inp = batch["ct"].to(args.device, dtype=torch.bfloat16, non_blocking=args.pin_memory)
            with maybe_suppress_model_stdout(args.show_model_stdout):
                out = adapter(
                    model.tokenizer,
                    inp,
                    entropy_loss_weight=0.0,
                    calculate_quantize_loss=False,
                    use_distributed_batch_entropy=False,
                )
            token_rows = None
            if token_dir is not None:
                token_rows = out.token_ids.detach().cpu().numpy().reshape(len(volume_ids), -1).astype(np.uint32)
                if token_rows.shape[1] != expected_tokens:
                    raise ValueError(f"token row width {token_rows.shape[1]}; expected {expected_tokens}")
            inp_batch = inp[:, 0].float().cpu().numpy()
            recon_batch = out.decoded[:, 0].float().cpu().numpy()
            for batch_index, volume_id in enumerate(volume_ids):
                inp_np = inp_batch[batch_index]
                recon_np = recon_batch[batch_index]
                valid_slices = tuple(slice(0, int(dim)) for dim in inp_np.shape)
                metrics = score_recon(inp_np, recon_np, valid_slices)
                rows.append(
                    {
                        "volume_id": volume_id,
                        "token_shape": tuple(int(x) for x in out.z_quantized[batch_index].shape),
                        **metrics,
                    }
                )
                if token_dir is not None and token_rows is not None:
                    save_token_row(token_dir, volume_id, token_rows[batch_index])
                    token_ids_written.append(volume_id)
                if viz_dir is not None and (viz_volume_ids is None or volume_id in viz_volume_ids):
                    render_recon_png(
                        volume_id,
                        inp_np,
                        recon_np,
                        valid_slices,
                        metrics,
                        viz_dir / f"{volume_id}_recon.png",
                        window=DEFAULT_RECON_VIZ_WINDOW,
                        level=DEFAULT_RECON_VIZ_LEVEL,
                        diff_window=args.viz_diff_window,
                        dpi=args.viz_dpi,
                    )
                    viz_written += 1

    if token_dir is not None:
        update_ids(token_dir / "ids.txt", token_ids_written)
        (token_dir / "metadata.json").write_text(
            json.dumps(
                {
                    **checkpoint_metadata(args, step, tokenizer_checkpoint=None),
                    "step": step,
                    "compression": args.compression,
                    "expected_tokens": expected_tokens,
                    "n": len(token_ids_written),
                    "source": "subtask6_delta_table_periodic_eval",
                    "cache_root": args.valid_cache_root,
                    "cache_split": args.valid_cache_split,
                },
                indent=2,
            )
        )

    summary = {
        "step": step,
        "n": len(rows),
        "metric_scope": "full_cached_preprocessed_tensor",
        "eval_batch_size": args.eval_batch_size,
        "viz_written": viz_written,
        "token_artifact_dir": str(token_dir) if token_dir is not None else None,
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])) if rows else 0.0,
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])) if rows else 0.0,
        "mean_mse": float(np.mean([row["mse"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }
    (step_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    summary_csv = eval_dir / "periodic_metrics.csv"
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["step", "n", "mean_ssim", "mean_psnr", "mean_mse"])
        writer.writerow(
            [
                step,
                len(rows),
                f"{summary['mean_ssim']:.8f}",
                f"{summary['mean_psnr']:.8f}",
                f"{summary['mean_mse']:.8f}",
            ]
        )
    print(
        f"periodic_eval step={step} n={len(rows)} SSIM={summary['mean_ssim']:.4f} "
        f"PSNR={summary['mean_psnr']:.2f} MSE={summary['mean_mse']:.5f}",
        flush=True,
    )
    if was_model_training:
        model.train()
    if was_adapter_training:
        adapter.train()


def main() -> int:
    
    cli_args = parse_args()
    args = load_run_args(cli_args)
    dist_state = init_distributed_runtime(args)
    distributed = dist_state.distributed
    rank = dist_state.rank
    local_rank = dist_state.local_rank
    world_size = dist_state.world_size

    seed_everything(args.seed + rank)

    out_dir = prepare_run_output(
        args,
        rank=rank,
        distributed=distributed,
        local_rank=local_rank,
        write_config_fn=write_config,
    )
    cached_paths = flat_cache_paths(Path(args.cache_root), args.cache_split, Path(args.cached_id_list))
    validate_configured_artifacts(args, cached_paths)
    train_mask_artifact_dir = resolve_train_mask_artifact_dir(args)
    logger = setup_rank0_logger("dtbd3d.encoder_training.train_learned_table_recon", out_dir, rank)
    smoke_debug_out_dir = out_dir / "smoke_debug"

    logger.info(f"run={args.run_name}")
    logger.info(f"out_dir={out_dir}")
    logger.info(f"device={args.device} compression={args.compression}")
    logger.info(f"rank={rank} local_rank={local_rank} world_size={world_size} distributed={distributed}")
    logger.info(
        f"smoke={bool(getattr(args, 'smoke', False))} steps={args.steps} checkpoint_every={args.checkpoint_every} "
        f"eval_every={args.eval_every} valid_n={args.valid_n} shuffle={args.shuffle}"
    )
    logger.info(f"trainable_mode={args.trainable_mode} loss_variant={args.loss_variant}")
    logger.info(f"delta_scale={args.delta_scale} delta_l2_weight={args.delta_l2_weight}")
    logger.info(f"lrs: delta={args.delta_lr} decoder={args.decoder_lr} encoder={args.encoder_lr}")
    logger.info(f"resume_checkpoint={args.resume_checkpoint or ''}")
    logger.info(
        "diagnostic_training_config=\n"
        + json.dumps(
            {
                "reconstruction": args.raw_config.get("reconstruction", {}),
                "input_conditioning": args.raw_config.get("input_conditioning", {}),
                "auxiliary_losses": args.raw_config.get("auxiliary_losses", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )

    model = load_model(args, load_fresh_weights=not bool(args.resume_checkpoint))
    set_trainable_modules(model, args.trainable_mode)
    adapter = LFQDeltaTableAdapter(
        model.tokenizer.quantize,
        vocab_size=model.codebook_size,
        embedding_dim=model.config.model.quantize_model.token_size,
        delta_scale=args.delta_scale,
        preserve_encoder_ste=args.preserve_encoder_ste,
    ).to(args.device)
    adapter.train()
    supervision = DiagnosticSupervision(args.supervision_config, visual_dim=model.config.model.quantize_model.token_size).to(args.device)
    supervision.train()
    train_module: torch.nn.Module = LearnedTableReconModule(model, adapter, supervision)
    
    if distributed:
        train_module = DistributedDataParallel(
            train_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.supervision_config.has_trainable_auxiliary,
        )
    base_module = unwrap_train_module(train_module)

    parameter_groups = optimizer_parameter_groups(base_module.model, base_module.adapter, base_module.supervision, args)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=0.0)
    start_step = 0
    if args.resume_checkpoint:
        start_step = load_full_training_checkpoint(base_module, optimizer, Path(args.resume_checkpoint), args.device)
        if start_step >= args.steps:
            raise ValueError(f"resume step {start_step} must be smaller than target steps {args.steps}")
        logger.info(
            f"resumed full training checkpoint={args.resume_checkpoint} start_step={start_step}; "
            "data loader restarts approximately"
        )
    if rank == 0:
        logger.info(json.dumps(
                {
                    "codebook_size": int(model.codebook_size),
                    "token_size": int(model.config.model.quantize_model.token_size),
                    "expected_token_count": int(expected_token_count(args.compression)),
                    "trainable_mode": args.trainable_mode,
                },
                sort_keys=True,
            )
        )
        logger.info(", ".join(f"{group['name']}:lr={group['lr']}:n={sum(p.numel() for p in group['params'])}" for group in parameter_groups)
        )
        supervision_debug: dict[str, Any] = {
            "has_trainable_auxiliary": bool(args.supervision_config.has_trainable_auxiliary),
            "text_enabled": bool(args.supervision_config.text_alignment.enabled),
            "disease_enabled": bool(args.supervision_config.disease_classification.enabled),
            "llama_prefix_enabled": bool(args.supervision_config.llama_prefix_prealignment.enabled),
        }
        text_module = getattr(base_module.supervision, "text", None)
        if text_module is not None:
            supervision_debug["text_sources"] = [
                {
                    "name": source.name,
                    "split": source.split,
                    "artifact_dir": source.artifact_dir,
                    "primary_mask_dir": source.mask_artifact_dir,
                    "max_pairs_per_volume": source.max_pairs_per_volume,
                }
                for source in args.supervision_config.text_alignment.embedding_sources
            ]
            supervision_debug["text_hidden_dim"] = int(text_module.hidden_dim)
            supervision_debug["text_target_dhw"] = list(text_module.target_dhw)
        prefix_module = getattr(base_module.supervision, "llama_prefix", None)
        if prefix_module is not None:
            supervision_debug["llama_prefix"] = {
                "model_name_or_path": args.supervision_config.llama_prefix_prealignment.model_name_or_path,
                "hidden_dim": int(args.supervision_config.llama_prefix_prealignment.hidden_dim),
                "num_prefix_tokens": int(args.supervision_config.llama_prefix_prealignment.num_prefix_tokens),
                "max_pairs_per_batch": int(args.supervision_config.llama_prefix_prealignment.max_pairs_per_batch),
                "target_dhw": list(args.supervision_config.llama_prefix_prealignment.reportgen_grid_dhw),
            }
        disease_module = getattr(base_module.supervision, "disease", None)
        disease_artifact = getattr(disease_module, "artifact", None)
        if disease_artifact is not None:
            supervision_debug["disease"] = {
                "num_ids": len(disease_artifact.ids),
                "organ_groups": disease_artifact.organ_groups,
                "num_diseases": len(disease_artifact.disease_names),
                "mask_root": str(disease_module.mask_root),
                "feature_grid_dhw": list(args.supervision_config.disease_classification.feature_grid_dhw or []),
            }
        print("[debug:supervision_summary] " + json.dumps(supervision_debug, sort_keys=True), flush=True)
    eval_items = prepare_periodic_eval(args)

    train_loader, train_sampler = make_train_dataloader(
        cached_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        shuffle=args.shuffle,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        supervision_config=args.supervision_config,
        train_mask_artifact_dir=str(train_mask_artifact_dir or ""),
        load_reconstruction_weight=args.loss_variant != "uniform",
        load_organ_input_weight=bool(args.supervision_config.organ_input.enabled),
        importance_lambda_uniform=args.importance_lambda_uniform,
    )
    
    metrics_writer = TrainMetricsWriter(out_dir, rank, append=bool(args.resume_checkpoint))
    step = start_step
    epoch = 0
    while step < args.steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            step_start = time.perf_counter()
            load_start = time.perf_counter()
            batch_volume_ids = list(batch["volume_id"])
            volume_cpu = batch["ct"]
            volume_id = "+".join(batch_volume_ids)
            volume = volume_cpu.to(device=args.device, dtype=torch.bfloat16, non_blocking=args.pin_memory)
            load_time = time.perf_counter() - load_start
            if step == 1 and rank == 0:
                print(
                    "[debug:batch] "
                    + json.dumps(
                        {
                            "step": step,
                            "volume_ids": batch_volume_ids,
                            "volume_cpu_shape": list(volume_cpu.shape),
                            "volume_device_shape": list(volume.shape),
                            "volume_dtype": str(volume.dtype),
                            "cache_split": args.cache_split,
                            "crop_depth": args.crop_depth,
                            "train_mask_artifact_dir": str(train_mask_artifact_dir or ""),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                print("[debug:dataloader_timing] " + json.dumps(batch.get("debug_timing", []), sort_keys=True), flush=True)
                loaded_json = export_loaded_batch_debug_json(
                    out_dir=smoke_debug_out_dir,
                    run_name=args.run_name,
                    step=step,
                    batch=batch,
                )
                print(f"[debug:loaded_batch_json] wrote {loaded_json}", flush=True)
                for loaded_png in render_loaded_batch_debug_png(
                    out_dir=smoke_debug_out_dir,
                    run_name=args.run_name,
                    step=step,
                    batch=batch,
                ):
                    print(f"[debug:loaded_batch_viz] wrote {loaded_png}", flush=True)
            breakpoint()
            importance_weights = maybe_build_importance_weights(
                batch.get("reconstruction_weight"),
                args.importance_lambda_uniform,
                batch_volume_ids,
                volume,
                crop_depth_value=args.crop_depth,
                step=step,
                device=args.device,
            )
            shaped_importance_weights = shape_importance_weights(importance_weights, args)
            # volume: raw cached preprocessed CT [B,1,D,H,W].
            # input_data: deterministic depth crop [B,1,crop_depth,H,W].
            # shaped_importance_weights: optional voxel weights matching input_data.
            input_data = crop_depth(volume, args.crop_depth, step)
            model_input = apply_organ_input_conditioning(input_data, shaped_importance_weights, args.supervision_config.organ_input)
            if step == 1 and rank == 0:
                print_tensor_stats(
                    "pre_forward",
                    volume=volume,
                    input_data=input_data,
                    model_input=model_input,
                    organ_input_delta=model_input.float() - input_data.float(),
                    raw_importance_weights=importance_weights,
                    shaped_importance_weights=shaped_importance_weights,
                )
                print_hu_stats(
                    "pre_forward_hu",
                    volume=volume,
                    input_data=input_data,
                    model_input=model_input,
                )
                for input_png in render_input_conditioning_debug_png(
                    out_dir=smoke_debug_out_dir,
                    run_name=args.run_name,
                    step=step,
                    volume_id=volume_id,
                    input_data=input_data,
                    model_input=model_input,
                    shaped_importance_weights=shaped_importance_weights,
                ):
                    print(f"[debug:input_conditioning_viz] wrote {input_png}", flush=True)

            breakpoint()
            assert_finite_tensor("input_data", input_data)
            assert_finite_tensor("model_input", model_input)
            assert_finite_tensor("shaped_importance_weights", shaped_importance_weights)
            if step == 1 and rank == 0 and "PYTEST_CURRENT_TEST" not in os.environ:
                print(
                    "[debug:breakpoint] paused before model forward; inspect volume/input_data/model_input/"
                    "importance_weights/shaped_importance_weights, then type c",
                    flush=True,
                )
                breakpoint()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with maybe_suppress_model_stdout(args.show_model_stdout):
                out = train_module(
                    model_input,
                    entropy_loss_weight=args.entropy_loss_weight,
                    calculate_quantize_loss=args.quantizer_aux_loss_weight > 0,
                    use_distributed_batch_entropy=args.use_distributed_batch_entropy,
                    batch_context={
                    "volume_ids": batch_volume_ids,
                    "importance_weights": shaped_importance_weights,
                    "text": batch.get("text"),
                    "disease": batch.get("disease"),
                    "full_shape_dhw": tuple(int(x) for x in volume.shape[2:]),
                    "crop_depth_value": args.crop_depth,
                    "step": step,
                    },
                )
            breakpoint()
            if step == 1 and rank == 0:
                print(
                    "[debug:forward_return] "
                    + json.dumps(
                        {
                            "decoded_shape": list(out.decoded.shape),
                            "z_quantized_shape": list(out.z_quantized.shape),
                            "token_ids_shape": list(out.token_ids.shape),
                            "token_ids_min": int(out.token_ids.detach().min().cpu()),
                            "token_ids_max": int(out.token_ids.detach().max().cpu()),
                            "has_trainable_auxiliary": bool(args.supervision_config.has_trainable_auxiliary),
                            "supervision_components": sorted((out.supervision.components or {}).keys()) if out.supervision is not None else [],
                            "supervision_metrics": out.supervision.metrics if out.supervision is not None else {},
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            recon_loss = reconstruction_loss(input_data, out.decoded, shaped_importance_weights, args)
            # out.decoded must match input_data: [B,1,crop_depth,H,W].
            # out.z_quantized is the native CT token grid before report-gen merge.
            if out.supervision is None:
                raise RuntimeError("DiagnosticSupervision is expected to run for every training batch")
            supervision_out = out.supervision
            if step == 1 and rank == 0:
                print_tensor_stats(
                    "post_forward",
                    decoded=out.decoded,
                    z_quantized=out.z_quantized,
                    token_ids=out.token_ids,
                    recon_loss=recon_loss,
                    aux_loss=supervision_out.loss,
                )
                png_paths = render_smoke_debug_png(
                    out_dir=smoke_debug_out_dir,
                    run_name=args.run_name,
                    step=step,
                    volume_id=volume_id,
                    input_data=input_data,
                    model_input=model_input,
                    decoded=out.decoded,
                    importance_weights=shaped_importance_weights,
                    diff_window=args.viz_diff_window,
                )
                for png_path in png_paths:
                    print(f"[debug:viz] wrote {png_path}", flush=True)
            breakpoint()
        
            assert_finite_tensor("decoded", out.decoded)
            assert_finite_tensor("z_quantized", out.z_quantized)
            assert_finite_tensor("recon_loss", recon_loss)
            quantize_loss = getattr(out.quantized_output, "entropy_aux_loss", None)
            per_sample_entropy_loss = getattr(out.quantize_loss_breakdown, "per_sample_entropy_loss", None)
            batch_entropy_loss = getattr(out.quantize_loss_breakdown, "batch_entropy_loss", None)
            commitment_loss = getattr(out.quantize_loss_breakdown, "commitment_loss", None)
            loss = args.recon_loss_weight * recon_loss
            aux_loss = supervision_out.loss
            disease_loss_value = float(supervision_out.metrics.get("disease_loss", 0.0))
            disease_pairs_value = int(supervision_out.metrics.get("disease_pairs", 0))
            disease_organs_value = int(supervision_out.metrics.get("disease_organs", 0))
            disease_pos_value = int(supervision_out.metrics.get("disease_pos", 0))
            disease_missing_masks_value = int(supervision_out.metrics.get("disease_missing_masks", 0))
            disease_feature_grid_value = str(supervision_out.metrics.get("disease_feature_grid", ""))
            text_loss_value = float(supervision_out.metrics.get("text_loss", 0.0))
            text_pairs_value = int(supervision_out.metrics.get("text_pairs", 0))
            text_candidate_pairs_value = int(supervision_out.metrics.get("text_candidate_pairs", text_pairs_value))
            text_supervised_volumes_value = int(supervision_out.metrics.get("text_supervised_volumes", 0))
            llama_prefix_loss_value = float(supervision_out.metrics.get("llama_prefix_loss", 0.0))
            llama_prefix_pairs_value = int(supervision_out.metrics.get("llama_prefix_pairs", 0))
            llama_prefix_target_tokens_value = int(supervision_out.metrics.get("llama_prefix_target_tokens", 0))
            if args.supervision_config.disease_classification.enabled and disease_pairs_value <= 0:
                raise RuntimeError(f"disease_classification is enabled but disease_pairs={disease_pairs_value}; fast-fail smoke check")
            loss = loss + aux_loss
            if quantize_loss is not None:
                loss = loss + args.quantizer_aux_loss_weight * quantize_loss
            if args.delta_l2_weight > 0:
                loss = loss + args.delta_l2_weight * out.delta_l2_loss
            assert_finite_tensor("total_loss", loss)
            if step == 1 and rank == 0:
                print(
                    "[debug:loss_assembly] "
                    + json.dumps(
                        {
                            "recon_loss_weight": float(args.recon_loss_weight),
                            "recon_loss": as_float(recon_loss),
                            "aux_loss": as_float(aux_loss),
                            "text_loss": text_loss_value,
                            "text_pairs": text_pairs_value,
                            "text_candidate_pairs": text_candidate_pairs_value,
                            "text_supervised_volumes": text_supervised_volumes_value,
                            "llama_prefix_loss": llama_prefix_loss_value,
                            "llama_prefix_pairs": llama_prefix_pairs_value,
                            "llama_prefix_target_tokens": llama_prefix_target_tokens_value,
                            "disease_loss": disease_loss_value,
                            "disease_pairs": disease_pairs_value,
                            "disease_organs": disease_organs_value,
                            "disease_pos": disease_pos_value,
                            "disease_missing_masks": disease_missing_masks_value,
                            "quantizer_aux_loss_weight": float(args.quantizer_aux_loss_weight),
                            "quantize_loss": as_float(quantize_loss),
                            "delta_l2_weight": float(args.delta_l2_weight),
                            "delta_l2_loss": as_float(out.delta_l2_loss),
                            "total_loss": as_float(loss),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if step == 1 and rank == 0 and "PYTEST_CURRENT_TEST" not in os.environ:
                print("[debug:breakpoint] paused after loss assembly; inspect tensors/metrics, then type c", flush=True)
                breakpoint()
            loss.backward()
            breakpoint()
        
            if step == 1 and rank == 0:
                grad_debug = {
                    "loss": as_float(loss),
                    "adapter_has_grad": any(param.grad is not None for param in base_module.adapter.parameters()),
                    "supervision_has_grad": any(param.grad is not None for param in base_module.supervision.parameters()),
                    "model_has_grad": any(param.grad is not None for param in base_module.model.parameters()),
                }
                print("[debug:backward] " + json.dumps(grad_debug, sort_keys=True), flush=True)
            if step == 1 and rank == 0 and "PYTEST_CURRENT_TEST" not in os.environ:
                print("[debug:breakpoint] paused after backward before optimizer.step(); inspect gradients, then type c", flush=True)
                breakpoint()
            optimizer.step()
            breakpoint()

            peak_mem_gb = torch.cuda.max_memory_allocated(device=args.device) / (1024**3) if torch.cuda.is_available() else 0.0
            weight_mean, weight_min, weight_max = importance_weight_stats(shaped_importance_weights)
            base_module = unwrap_train_module(train_module)
            usage = code_usage_metrics(out.token_ids, vocab_size=base_module.model.codebook_size)
            table_stats = delta_table_metrics(base_module.adapter, out.token_ids)
            step_time = time.perf_counter() - step_start
            metrics_writer.write(
                {
                    "step": step,
                    "volume_id": volume_id,
                    "loss": f"{as_float(loss):.8f}",
                    "recon_loss": f"{as_float(recon_loss):.8f}",
                    "aux_loss": f"{as_float(aux_loss):.8f}",
                    "disease_loss": f"{disease_loss_value:.8f}",
                    "disease_pairs": disease_pairs_value,
                    "disease_organs": disease_organs_value,
                    "disease_pos": disease_pos_value,
                    "disease_missing_masks": disease_missing_masks_value,
                    "disease_feature_grid": disease_feature_grid_value,
                    "text_loss": f"{text_loss_value:.8f}",
                    "text_pairs": text_pairs_value,
                    "text_candidate_pairs": text_candidate_pairs_value,
                    "text_supervised_volumes": text_supervised_volumes_value,
                    "llama_prefix_loss": f"{llama_prefix_loss_value:.8f}",
                    "llama_prefix_pairs": llama_prefix_pairs_value,
                    "llama_prefix_target_tokens": llama_prefix_target_tokens_value,
                    "quantize_loss": f"{as_float(quantize_loss):.8f}",
                    "delta_l2_loss": f"{as_float(out.delta_l2_loss):.8f}",
                    "per_sample_entropy_loss": f"{as_float(per_sample_entropy_loss):.8f}",
                    "batch_entropy_loss": f"{as_float(batch_entropy_loss):.8f}",
                    "commitment_loss": f"{as_float(commitment_loss):.8f}",
                    "importance_weight_mean": f"{weight_mean:.8f}",
                    "importance_weight_min": f"{weight_min:.8f}",
                    "importance_weight_max": f"{weight_max:.8f}",
                    "unique_codes": usage["unique_codes"],
                    "unique_code_ratio": f"{usage['unique_code_ratio']:.8f}",
                    "code_usage_entropy": f"{usage['code_usage_entropy']:.8f}",
                    "code_usage_perplexity": f"{usage['code_usage_perplexity']:.8f}",
                    "delta_abs_mean": f"{table_stats['delta_abs_mean']:.8f}",
                    "delta_abs_max": f"{table_stats['delta_abs_max']:.8f}",
                    "touched_delta_abs_mean": f"{table_stats.get('touched_delta_abs_mean', 0.0):.8f}",
                    "touched_delta_abs_max": f"{table_stats.get('touched_delta_abs_max', 0.0):.8f}",
                    "step_time_sec": f"{step_time:.4f}",
                    "load_time_sec": f"{load_time:.4f}",
                    "peak_gpu_mem_gb": f"{peak_mem_gb:.4f}",
                }
            )
            should_log_rank = not args.log_rank0_only or rank == 0
            if should_log_rank and (step % args.log_every == 0 or step == 1 or step == args.steps):
                print(
                    f"rank={rank} step={step} volume={volume_id} loss={as_float(loss):.6f} "
                    f"recon={as_float(recon_loss):.6f} aux={as_float(aux_loss):.6f} "
                    f"disease={disease_loss_value:.6f} disease_pairs={disease_pairs_value} "
                    f"disease_pos={disease_pos_value} disease_grid={disease_feature_grid_value} "
                    f"text={text_loss_value:.6f} text_pairs={text_pairs_value}/{text_candidate_pairs_value} "
                    f"text_vols={text_supervised_volumes_value} "
                    f"quant={as_float(quantize_loss):.6f} "
                    f"unique={usage['unique_codes']} delta_abs={table_stats['delta_abs_mean']:.6f} "
                    f"peak_mem={peak_mem_gb:.2f}GB",
                    flush=True,
                )
            if rank == 0 and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                tokenizer_path, delta_path, full_state_path = save_checkpoints(train_module, optimizer, args, step)
                print(f"saved tokenizer checkpoint: {tokenizer_path}", flush=True)
                print(f"saved delta checkpoint: {delta_path}", flush=True)
                print(f"saved full training checkpoint: {full_state_path}", flush=True)
            if distributed and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                distributed_barrier(local_rank)
            if args.eval_every > 0 and step % args.eval_every == 0:
                if distributed:
                    distributed_barrier(local_rank)
                if rank == 0:
                    base_module = unwrap_train_module(train_module)
                    run_periodic_eval(base_module.model, base_module.adapter, args, eval_items, step)
                if distributed:
                    distributed_barrier(local_rank)
            if step >= args.steps:
                break
        epoch += 1
    breakpoint()

    if distributed:
        distributed_barrier(local_rank)
    if rank == 0 and (args.checkpoint_every == 0 or args.steps % args.checkpoint_every != 0):
        tokenizer_path, delta_path, full_state_path = save_checkpoints(train_module, optimizer, args, args.steps)
        print(f"saved tokenizer checkpoint: {tokenizer_path}", flush=True)
        print(f"saved delta checkpoint: {delta_path}", flush=True)
        print(f"saved full training checkpoint: {full_state_path}", flush=True)
    if distributed:
        distributed_barrier(local_rank)
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
