"""Training helpers for weighted CT reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def depth_crop_bounds(depth: int, crop_depth_value: int, step: int) -> tuple[int, int]:
    """Return the same deterministic depth crop used by the CT train loop."""
    if crop_depth_value <= 0 or crop_depth_value >= depth:
        return 0, depth
    if crop_depth_value < 5:
        raise ValueError(f"crop-depth must be >=5 or 0, got {crop_depth_value}")
    max_start = depth - crop_depth_value
    start = (step * 17) % (max_start + 1)
    return start, start + crop_depth_value


def crop_depth_tensor(tensor: torch.Tensor, crop_depth_value: int, step: int) -> torch.Tensor:
    start, end = depth_crop_bounds(int(tensor.shape[2]), crop_depth_value, step)
    return tensor[:, :, start:end, :, :]


def apply_uniform_mix(weight: np.ndarray, lambda_uniform: float) -> np.ndarray:
    """Mix a mean-normalized anatomical map with the uniform baseline."""
    if not 0.0 <= lambda_uniform <= 1.0:
        raise ValueError(f"lambda_uniform must be in [0,1], got {lambda_uniform}")
    weight = np.asarray(weight, dtype=np.float32)
    if lambda_uniform == 0.0:
        return weight
    if lambda_uniform == 1.0:
        return np.ones_like(weight, dtype=np.float32)
    return ((1.0 - lambda_uniform) * weight + lambda_uniform).astype(np.float32, copy=False)


def load_token_weight_map(map_dir: str | Path, volume_id: str, lambda_uniform: float = 0.0) -> np.ndarray:
    """Synthesize one token-grid reconstruction weight map from a TS mask npz."""
    map_dir = Path(map_dir)
    npz_path = map_dir / f"{volume_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    npz = np.load(npz_path, allow_pickle=False)
    masks = np.asarray(npz["mask_token"], dtype=np.float32)
    if masks.ndim != 4:
        raise ValueError(f"mask_token must be [M,D,H,W], got {masks.shape}: {npz_path}")
    if masks.shape[0] == 0:
        raise ValueError(f"empty mask_token cannot build reconstruction weights: {npz_path}")
    union = np.clip(masks.max(axis=0), 0.0, 1.0)
    weight = 1.0 + union
    weight = weight / max(float(weight.mean()), 1e-6)
    if weight.ndim != 3:
        raise ValueError(f"importance map must be 3D, got {weight.shape}: {map_dir}/{volume_id}")
    return apply_uniform_mix(weight, lambda_uniform=lambda_uniform)


def build_importance_weight_batch(
    volume_ids: list[str],
    map_dir: str | Path,
    full_shape_dhw: tuple[int, int, int],
    crop_depth_value: int,
    step: int,
    device: str | torch.device,
    dtype: torch.dtype,
    lambda_uniform: float = 0.0,
) -> torch.Tensor:
    """Convert saved token-grid maps to voxel weights for reconstruction loss.

    Saved maps are token-grid arrays with mean close to 1. The reconstruction
    loss is voxel-level after the CT crop, so this function upsamples maps to
    the full preprocessed CT shape, applies the same depth crop, and renormalizes
    each sample after cropping. The post-crop renormalization keeps the weighted
    L1 scale comparable to the uniform baseline.
    """
    maps = [load_token_weight_map(map_dir, volume_id, lambda_uniform=lambda_uniform) for volume_id in volume_ids]
    token_weights = torch.from_numpy(np.stack(maps, axis=0))[:, None].to(device=device, dtype=torch.float32)
    voxel_weights = F.interpolate(
        token_weights,
        size=full_shape_dhw,
        mode="trilinear",
        align_corners=False,
    )
    voxel_weights = crop_depth_tensor(voxel_weights, crop_depth_value, step)
    means = voxel_weights.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    voxel_weights = voxel_weights / means
    return voxel_weights.to(dtype=dtype)


def token_weight_maps_to_voxel_batch(
    token_maps: torch.Tensor,
    full_shape_dhw: tuple[int, int, int],
    crop_depth_value: int,
    step: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a preloaded [B,D,H,W] token-weight batch to cropped voxel weights."""
    if token_maps.ndim != 4:
        raise ValueError(f"token_maps must be [B,D,H,W], got {tuple(token_maps.shape)}")
    token_weights = token_maps[:, None].to(device=device, dtype=torch.float32)
    voxel_weights = F.interpolate(
        token_weights,
        size=full_shape_dhw,
        mode="trilinear",
        align_corners=False,
    )
    voxel_weights = crop_depth_tensor(voxel_weights, crop_depth_value, step)
    means = voxel_weights.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    voxel_weights = voxel_weights / means
    return voxel_weights.to(dtype=dtype)


def weighted_l1_loss(input_data: torch.Tensor, recon_output: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted replacement for the tokenizer's internal uniform L1 loss."""
    if input_data.shape != recon_output.shape:
        raise ValueError(f"input/recon shape mismatch: {input_data.shape} vs {recon_output.shape}")
    if weights.shape[0] != input_data.shape[0] or weights.shape[2:] != input_data.shape[2:]:
        raise ValueError(f"weight shape {weights.shape} is incompatible with input {input_data.shape}")
    error = torch.abs(input_data.float() - recon_output.float())
    return (error * weights.float()).mean()
