#!/usr/bin/env python
"""Tiny reconstruction-only training smoke for the BTB3D encoder/tokenizer.

This is intentionally smaller and cleaner than the frozen author training
script. It is for resource profiling and smoke validation only:

    preprocessed CT cache -> BTB3D tokenizer/decoder -> L1 reconstruction loss

It does not run GAN, perceptual loss, Accelerate, TensorBoard, or report-gen.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from dtbd3d.core.artifact import save_token_row, update_ids
from dtbd3d.core.btb3d_model import CHECKPOINT_DIRS, CONFIGS, expected_token_count
from dtbd3d.core.ct_preprocess import preprocess_volume
from dtbd3d.importance_maps.training import build_importance_weight_batch, crop_depth_tensor, weighted_l1_loss


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_links = root / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compression", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--cache-root", required=True, help="Flat preprocessed tensor cache root containing train/ and valid/.")
    parser.add_argument("--cached-id-list", required=True, help="Text file with one volume ID per line for flat cache loading.")
    parser.add_argument("--cache-split", choices=["train", "valid"], required=True, help="Flat cache split directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="", help="Optional directory for tokenizer checkpoints.")
    parser.add_argument("--tokenizer-checkpoint", default="", help="Optional tokenizer checkpoint to resume from.")
    parser.add_argument("--btb3d-repo", default=str(root / "Experiment/core_code/btb3d_baseline/encoder-decoder"))
    parser.add_argument("--weights-root", default=str(data_links / "btb3d_weights"))
    parser.add_argument("--metadata-csv", default=str(data_links / "ct_rate/dataset/metadata/validation_metadata.csv"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop-depth", type=int, default=105)
    parser.add_argument("--recon-loss-weight", type=float, default=5.0)
    parser.add_argument("--importance-map-dir", default="", help="Directory containing per-volume token weight .npy files.")
    parser.add_argument(
        "--importance-lambda-uniform",
        type=float,
        default=0.0,
        help="Mix loaded base importance maps with the uniform baseline. Keep 0.0 for already-mixed maps.",
    )
    parser.add_argument("--entropy-loss-weight", type=float, default=0.1)
    parser.add_argument("--quantizer-aux-loss-weight", type=float, default=1.0)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--diversity-gamma", type=float, default=1.0)
    parser.add_argument("--use-distributed-batch-entropy", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--no-save", action="store_true", help="Do not save tokenizer checkpoints.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--log-rank0-only", action="store_true")
    parser.add_argument("--show-model-stdout", action="store_true")
    parser.add_argument("--metrics-style", choices=["legacy", "rank"], default="legacy")
    parser.add_argument("--eval-every", type=int, default=0, help="Run periodic validation every N steps on rank 0.")
    parser.add_argument("--eval-dir", default="", help="Directory for periodic validation metrics.")
    parser.add_argument("--valid-id-list", default="", help="Validation volume ID text file for periodic validation.")
    parser.add_argument("--valid-cache-root", default="", help="Flat validation cache root containing valid/{volume_id}.npy.")
    parser.add_argument("--valid-cache-split", choices=["train", "valid"], default="valid")
    parser.add_argument("--valid-n", type=int, default=0, help="Number of validation IDs to use for periodic validation.")
    parser.add_argument("--valid-data-root", default=str(data_links / "ct_rate"))
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Periodic eval decode batch size.")
    parser.add_argument("--eval-viz-n", type=int, default=0, help="Max periodic eval PNGs per step; 0 means all evaluated volumes.")
    parser.add_argument("--eval-save-viz", action="store_true", help="Save periodic validation PNGs.")
    parser.add_argument("--eval-save-tokens", action="store_true", help="Save periodic eval token indices as token artifacts.")
    return parser.parse_args()


def distributed_state() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def distributed_barrier(local_rank: int) -> None:
    dist.barrier(device_ids=[local_rank])


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    btb3d_repo = Path(args.btb3d_repo)
    sys.path.insert(0, str(btb3d_repo))
    from modeling.magvit_model import VisionTokenizer
    from src.utils import get_config

    config_file = btb3d_repo / "configs" / CONFIGS[args.compression]
    ckpt_file = Path(args.weights_root) / "encoder-decoder" / CHECKPOINT_DIRS[args.compression] / "3rd_stage.ckpt"
    if not config_file.exists():
        raise FileNotFoundError(config_file)
    if not ckpt_file.exists():
        raise FileNotFoundError(ckpt_file)

    model_config = get_config(str(config_file))
    model = VisionTokenizer(
        config=model_config,
        commitment_cost=args.commitment_cost,
        diversity_gamma=args.diversity_gamma,
        use_gan=False,
        use_lecam_ema=False,
        use_perceptual=False,
    )
    states = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    model.tokenizer.load_state_dict(states, strict=True)
    if args.tokenizer_checkpoint:
        tokenizer_state = load_file(args.tokenizer_checkpoint, device="cpu")
        model.tokenizer.load_state_dict(tokenizer_state, strict=True)
    model.train()
    model.to(args.device).to(torch.bfloat16)
    return model


def load_cached_volume(path: Path, device: str) -> torch.Tensor:
    array = np.load(path)
    if array.ndim != 5:
        raise ValueError(f"cached tensor must be 5D (B,C,D,H,W), got {array.shape}: {path}")
    tensor = torch.from_numpy(array)
    return tensor.to(device=device, dtype=torch.bfloat16)


def load_cached_eval_volume(path: Path) -> torch.Tensor:
    array = np.load(path)
    if array.ndim != 5:
        raise ValueError(f"cached tensor must be 5D (B,C,D,H,W), got {array.shape}: {path}")
    if array.shape[0] != 1:
        raise ValueError(f"cached tensor leading batch dim must be 1, got {array.shape}: {path}")
    return torch.from_numpy(array[0])


def read_id_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def flat_cache_paths(cache_root: Path, split: str, ids_file: Path) -> list[Path]:
    return [cache_root / split / f"{volume_id}.npy" for volume_id in read_id_list(ids_file)]


def find_nii(volume_dir: Path, volume_id: str) -> Path:
    target = f"{volume_id}.nii.gz"
    for dirpath, _dirnames, filenames in os.walk(volume_dir):
        if target in filenames:
            return Path(dirpath) / target
    raise FileNotFoundError(f"Could not find {target} under {volume_dir}")


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


class CachedVolumeDataset(Dataset[tuple[str, torch.Tensor]]):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        path = self.paths[index]
        array = np.load(path)
        if array.ndim != 5:
            raise ValueError(f"cached tensor must be 5D (B,C,D,H,W), got {array.shape}: {path}")
        if array.shape[0] != 1:
            raise ValueError(f"cached tensor leading batch dim must be 1, got {array.shape}: {path}")
        return path.stem, torch.from_numpy(array[0])


def infinite_cached_batches(
    paths: list[Path],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> Any:
    dataset = CachedVolumeDataset(paths)
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    while True:
        for volume_ids, volumes in loader:
            yield list(volume_ids), volumes


def crop_depth(tensor: torch.Tensor, crop_depth_value: int, step: int) -> torch.Tensor:
    return crop_depth_tensor(tensor, crop_depth_value, step)


def maybe_build_importance_weights(
    importance_map_dir: Path | None,
    importance_lambda_uniform: float,
    batch_volume_ids: list[str],
    volume: torch.Tensor,
    crop_depth_value: int,
    step: int,
    device: str,
) -> torch.Tensor | None:
    """Load V3 weights for a batch, or return `None` for the uniform baseline."""
    if importance_map_dir is None:
        return None
    return build_importance_weight_batch(
        batch_volume_ids,
        importance_map_dir,
        full_shape_dhw=tuple(int(dim) for dim in volume.shape[2:]),
        crop_depth_value=crop_depth_value,
        step=step,
        device=device,
        dtype=torch.float32,
        lambda_uniform=importance_lambda_uniform,
    )


def importance_weight_stats(importance_weights: torch.Tensor | None) -> tuple[float, float, float]:
    """Return CSV-ready weight stats; uniform baseline is represented as 1/1/1."""
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


@contextlib.contextmanager
def maybe_suppress_model_stdout(show_model_stdout: bool) -> Any:
    if show_model_stdout:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def save_tokenizer_checkpoint(
    model: torch.nn.Module,
    out_dir: Path,
    step: int,
    checkpoint_dir: Path | None = None,
) -> Path:
    if isinstance(model, DistributedDataParallel):
        model = model.module
    ckpt_dir = checkpoint_dir if checkpoint_dir is not None else out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.tokenizer.state_dict().items()
    }
    path = ckpt_dir / f"tokenizer_step_{step}.safetensors"
    save_file(state, path)
    return path


def write_config(out_dir: Path, args: argparse.Namespace) -> None:
    serializable = vars(args).copy()
    (out_dir / "config.json").write_text(json.dumps(serializable, indent=2))


def read_valid_ids(path: Path, count: int) -> list[str]:
    volume_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if count > 0:
        volume_ids = volume_ids[:count]
    return volume_ids


def prepare_periodic_eval(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.eval_every <= 0:
        return []
    if not args.eval_dir:
        raise ValueError("--eval-dir is required when --eval-every > 0")
    if not args.valid_id_list:
        raise ValueError("--valid-id-list is required when --eval-every > 0")
    if not args.valid_cache_root:
        raise ValueError("--valid-cache-root is required when --eval-every > 0")
    valid_ids = read_valid_ids(Path(args.valid_id_list), args.valid_n)
    valid_cache_root = Path(args.valid_cache_root)
    eval_items = [(volume_id, valid_cache_root / args.valid_cache_split / f"{volume_id}.npy") for volume_id in valid_ids]
    for _volume_id, path in eval_items:
        if not path.exists():
            raise FileNotFoundError(path)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.eval_dir) / "volume_ids.txt").write_text("\n".join(valid_ids) + "\n")
    return eval_items


def iter_chunks(items: list[tuple[str, Path]], chunk_size: int) -> Any:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def run_periodic_eval(
    model: torch.nn.Module,
    args: argparse.Namespace,
    eval_items: list[tuple[str, Path]],
    step: int,
    distributed: bool,
    rank: int,
    world_size: int,
) -> None:
    eval_model = model.module if isinstance(model, DistributedDataParallel) else model
    was_training = eval_model.training
    eval_model.eval()
    eval_dir = Path(args.eval_dir)
    step_dir = eval_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = step_dir / "viz" if args.eval_save_viz else None
    token_dir = step_dir / "token_artifact" if args.eval_save_tokens else None
    token_ids: list[str] = []
    expected_tokens = expected_token_count(args.compression)
    viz_volume_ids = {volume_id for volume_id, _path in eval_items[: args.eval_viz_n]} if args.eval_viz_n > 0 else None
    rank_eval_items = eval_items[rank::world_size] if distributed else eval_items

    rows = []
    viz_written = 0
    with torch.no_grad():
        progress = tqdm(
            iter_chunks(rank_eval_items, args.eval_batch_size),
            total=(len(rank_eval_items) + args.eval_batch_size - 1) // args.eval_batch_size,
            desc=f"periodic_eval step={step} rank={rank}",
            unit="batch",
            disable=rank != 0,
        )
        for batch_items in progress:
            volume_ids = [volume_id for volume_id, _path in batch_items]
            volume_tensors = [load_cached_eval_volume(path) for _volume_id, path in batch_items]
            inp = torch.stack(volume_tensors, dim=0).to(args.device, dtype=torch.bfloat16)
            with maybe_suppress_model_stdout(args.show_model_stdout):
                decoded, z_quantized, quantized_output, _ = eval_model.tokenizer(
                    inp,
                    entropy_loss_weight=0.0,
                    calculate_quantize_loss=False,
                )
            token_rows = None
            if token_dir is not None:
                token_rows = quantized_output.indices.detach().cpu().numpy().reshape(len(volume_ids), -1).astype(np.uint32)
                if token_rows.shape[1] != expected_tokens:
                    raise ValueError(f"token row width {token_rows.shape[1]}; expected {expected_tokens}")
            inp_batch = inp[:, 0].float().cpu().numpy()
            recon_batch = decoded[:, 0].float().cpu().numpy()
            for batch_index, volume_id in enumerate(volume_ids):
                inp_np = inp_batch[batch_index]
                recon_np = recon_batch[batch_index]
                valid_slices = tuple(slice(0, int(dim)) for dim in inp_np.shape)
                metrics = score_recon(inp_np, recon_np, valid_slices)
                token_shape = tuple(int(x) for x in z_quantized[batch_index].shape)
                rows.append({"volume_id": volume_id, "token_shape": token_shape, **metrics})
                if token_dir is not None and token_rows is not None:
                    save_token_row(token_dir, volume_id, token_rows[batch_index])
                    token_ids.append(volume_id)
                should_write_viz = viz_dir is not None and (viz_volume_ids is None or volume_id in viz_volume_ids)
                if should_write_viz:
                    render_recon_png(
                        volume_id,
                        inp_np,
                        recon_np,
                        valid_slices,
                        metrics,
                        viz_dir / f"{volume_id}_recon.png",
                        window=1200.0,
                        level=-700.0,
                        diff_window=500.0,
                        dpi=150,
                    )
                    viz_written += 1

    gathered: list[dict[str, Any] | None]
    local_payload = {"rows": rows, "token_ids": token_ids, "viz_written": viz_written}
    if distributed:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]
    if rank != 0:
        if was_training:
            eval_model.train()
        return

    all_rows = [row for payload in gathered if payload is not None for row in payload["rows"]]
    all_token_ids = [volume_id for payload in gathered if payload is not None for volume_id in payload["token_ids"]]
    total_viz_written = sum(int(payload["viz_written"]) for payload in gathered if payload is not None)
    row_order = {volume_id: index for index, (volume_id, _path) in enumerate(eval_items)}
    all_rows.sort(key=lambda row: row_order.get(row["volume_id"], len(row_order)))
    token_id_set = set(all_token_ids)
    all_token_ids = [volume_id for volume_id, _path in eval_items if volume_id in token_id_set]

    if token_dir is not None:
        update_ids(token_dir / "ids.txt", all_token_ids)
        (token_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "compression": args.compression,
                    "expected_tokens": expected_tokens,
                    "n": len(all_token_ids),
                    "source": "periodic_reconstruction_eval",
                    "cache_root": args.valid_cache_root,
                    "cache_split": args.valid_cache_split,
                    "distributed_eval": distributed,
                    "world_size": world_size,
                },
                indent=2,
            )
        )

    summary = {
        "step": step,
        "n": len(all_rows),
        "metric_scope": "full_cached_preprocessed_tensor",
        "eval_batch_size": args.eval_batch_size,
        "distributed_eval": distributed,
        "world_size": world_size,
        "viz_written": total_viz_written,
        "token_artifact_dir": str(token_dir) if token_dir is not None else None,
        "mean_ssim": float(np.mean([row["ssim"] for row in all_rows])),
        "mean_psnr": float(np.mean([row["psnr"] for row in all_rows])),
        "mean_mse": float(np.mean([row["mse"] for row in all_rows])),
        "rows": all_rows,
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
                len(all_rows),
                f"{summary['mean_ssim']:.8f}",
                f"{summary['mean_psnr']:.8f}",
                f"{summary['mean_mse']:.8f}",
            ]
        )
    print(
        f"periodic_eval step={step} n={len(all_rows)} "
        f"batch_size={args.eval_batch_size} distributed={distributed} world_size={world_size} "
        f"viz={total_viz_written} "
        f"SSIM={summary['mean_ssim']:.4f} PSNR={summary['mean_psnr']:.2f} "
        f"MSE={summary['mean_mse']:.5f}",
        flush=True,
    )
    if was_training:
        eval_model.train()


def main() -> int:
    args = parse_args()
    distributed, rank, local_rank, world_size = distributed_state()
    if distributed:
        if args.device.startswith("cuda"):
            args.device = f"cuda:{local_rank}"
        torch.cuda.set_device(args.device)
        dist.init_process_group(backend="nccl", device_id=torch.device(args.device))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_config(out_dir, args)
    if distributed:
        distributed_barrier(local_rank)

    cached_paths = flat_cache_paths(Path(args.cache_root), args.cache_split, Path(args.cached_id_list))
    if args.batch_size < 1:
        raise ValueError(f"batch-size must be >=1, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"num-workers must be >=0, got {args.num_workers}")
    if args.prefetch_factor < 1:
        raise ValueError(f"prefetch-factor must be >=1, got {args.prefetch_factor}")
    if args.log_every < 1:
        raise ValueError(f"log-every must be >=1, got {args.log_every}")
    if args.eval_every < 0:
        raise ValueError(f"eval-every must be >=0, got {args.eval_every}")
    if args.eval_batch_size < 1:
        raise ValueError(f"eval-batch-size must be >=1, got {args.eval_batch_size}")
    if args.eval_viz_n < 0:
        raise ValueError(f"eval-viz-n must be >=0, got {args.eval_viz_n}")
    active_paths = cached_paths
    for path in active_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"rank={rank} local_rank={local_rank} world_size={world_size} distributed={distributed}", flush=True)
    print(f"device={args.device}", flush=True)
    print(f"compression={args.compression}", flush=True)
    print(f"steps={args.steps} lr={args.lr} crop_depth={args.crop_depth} batch_size={args.batch_size}", flush=True)
    print(
        f"volumes={len(active_paths)} cache_root={args.cache_root} cache_split={args.cache_split} "
        f"num_workers={args.num_workers} pin_memory={args.pin_memory}",
        flush=True,
    )
    importance_map_dir = Path(args.importance_map_dir) if args.importance_map_dir else None
    if importance_map_dir is not None and not importance_map_dir.exists():
        raise FileNotFoundError(importance_map_dir)
    print(f"importance_map_dir={importance_map_dir}", flush=True)

    periodic_eval_items = prepare_periodic_eval(args)

    model = load_model(args)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    tokenizer_params = model.module.tokenizer.parameters() if isinstance(model, DistributedDataParallel) else model.tokenizer.parameters()
    optimizer = torch.optim.AdamW(tokenizer_params, lr=args.lr, weight_decay=0.0)
    loss_weights = {
        "recon_loss_weight": args.recon_loss_weight,
        "quantizer_entropy_loss_weight": args.entropy_loss_weight,
        "quantizer_aux_loss_weight": args.quantizer_aux_loss_weight,
    }

    cached_batch_iter = infinite_cached_batches(
        cached_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    if args.metrics_style == "rank":
        metrics_name = f"metrics_rank{rank}.csv"
    else:
        metrics_name = "train_metrics.csv" if rank == 0 else f"train_metrics_rank{rank}.csv"
    metrics_path = out_dir / metrics_name
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "volume_id",
                "loss",
                "recon_loss",
                "quantize_loss",
                "per_sample_entropy_loss",
                "batch_entropy_loss",
                "commitment_loss",
                "importance_weight_mean",
                "importance_weight_min",
                "importance_weight_max",
                "step_time_sec",
                "load_time_sec",
                "peak_gpu_mem_gb",
            ]
        )

        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            load_start = time.perf_counter()
            volume_ids, volume_cpu = next(cached_batch_iter)
            batch_volume_ids = list(volume_ids)
            volume_id = "+".join(batch_volume_ids)
            volume = volume_cpu.to(device=args.device, dtype=torch.bfloat16, non_blocking=args.pin_memory)
            load_time = time.perf_counter() - load_start

            importance_weights = maybe_build_importance_weights(
                importance_map_dir,
                args.importance_lambda_uniform,
                batch_volume_ids,
                volume,
                crop_depth_value=args.crop_depth,
                step=step,
                device=args.device,
            )
            input_data = crop_depth(volume, args.crop_depth, step)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with maybe_suppress_model_stdout(args.show_model_stdout):
                recon_output, _indices, recon_loss, quantize_loss, _breakdown = model(
                    forward_mode="reconstruction",
                    data_mode="3d",
                    input_data=input_data,
                    calculate_loss=True,
                    loss_weights=loss_weights,
                    use_distributed_batch_entropy=args.use_distributed_batch_entropy,
                )
            if importance_weights is not None:
                recon_loss = weighted_l1_loss(input_data, recon_output, importance_weights)
            loss = args.recon_loss_weight * recon_loss
            if quantize_loss is not None:
                loss = loss + args.quantizer_aux_loss_weight * quantize_loss
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated(device=args.device) / (1024**3)
            else:
                peak_mem_gb = 0.0
            importance_weight_mean, importance_weight_min, importance_weight_max = importance_weight_stats(importance_weights)
            step_time = time.perf_counter() - step_start
            writer.writerow(
                [
                    step,
                    volume_id,
                    f"{as_float(loss):.8f}",
                    f"{as_float(recon_loss):.8f}",
                    f"{as_float(quantize_loss):.8f}",
                    f"{as_float(_breakdown.per_sample_entropy_loss):.8f}",
                    f"{as_float(_breakdown.batch_entropy_loss):.8f}",
                    f"{as_float(_breakdown.commitment_loss):.8f}",
                    f"{importance_weight_mean:.8f}",
                    f"{importance_weight_min:.8f}",
                    f"{importance_weight_max:.8f}",
                    f"{step_time:.4f}",
                    f"{load_time:.4f}",
                    f"{peak_mem_gb:.4f}",
                ]
            )
            handle.flush()
            should_log_step = step % args.log_every == 0 or step == 1 or step == args.steps
            should_log_rank = not args.log_rank0_only or rank == 0
            if should_log_step and should_log_rank:
                print(
                    f"rank={rank} step={step} volume={volume_id} loss={as_float(loss):.6f} "
                    f"recon={as_float(recon_loss):.6f} quant={as_float(quantize_loss):.6f} "
                    f"step_time={step_time:.2f}s load={load_time:.2f}s peak_mem={peak_mem_gb:.2f}GB",
                    flush=True,
                )

            if not args.no_save and rank == 0 and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                ckpt_path = save_tokenizer_checkpoint(model, out_dir, step, checkpoint_dir)
                print(f"saved checkpoint: {ckpt_path}", flush=True)

            if args.eval_every > 0 and step % args.eval_every == 0:
                if distributed:
                    distributed_barrier(local_rank)
                run_periodic_eval(model, args, periodic_eval_items, step, distributed, rank, world_size)
                if distributed:
                    distributed_barrier(local_rank)

    if distributed:
        distributed_barrier(local_rank)
    if not args.no_save and rank == 0 and (args.checkpoint_every == 0 or args.steps % args.checkpoint_every != 0):
        ckpt_path = save_tokenizer_checkpoint(model, out_dir, args.steps, checkpoint_dir)
        print(f"saved checkpoint: {ckpt_path}", flush=True)
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
