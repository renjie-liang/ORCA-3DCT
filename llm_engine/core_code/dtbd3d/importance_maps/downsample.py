"""Downsample voxel-resolution fields to BTB3D token grids."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def downsample_to_token_grid(
    voxel_field: np.ndarray,
    target_shape: tuple[int, int, int],
    method: str,
) -> np.ndarray:
    if voxel_field.ndim != 3:
        raise ValueError(f"voxel_field must be 3D (D,H,W), got {voxel_field.shape}")
    field = np.asarray(voxel_field, dtype=np.float32)
    tensor = torch.from_numpy(field)[None, None]
    if method == "mean":
        pooled = F.adaptive_avg_pool3d(tensor, target_shape)
    elif method == "max":
        pooled = F.adaptive_max_pool3d(tensor, target_shape)
    elif method == "any":
        pooled = (F.adaptive_max_pool3d(tensor, target_shape) > 0).to(torch.float32)
    else:
        raise ValueError(f"Unsupported downsample method: {method}")
    return pooled[0, 0].numpy().astype(np.float32, copy=False)

