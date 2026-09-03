"""Fast token-grid approximations for anatomical importance maps."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from dtbd3d.importance_maps.downsample import downsample_to_token_grid
from dtbd3d.importance_maps.ontology import ClinicalOntology
from dtbd3d.importance_maps.ts_mask_loader import (
    TARGET_SHAPE_HWD,
    TARGET_SPACING_DHW,
    _center_crop_pad_hwd,
    _metadata_spacing,
    ts_mask_path,
)


def organ_count_lookup(ontology: ClinicalOntology) -> np.ndarray:
    """Return TS class -> number of CT-RATE findings attached to that class."""
    lookup = np.zeros(118, dtype=np.float32)
    for finding in ontology.findings:
        for class_id in finding.ts_class_ids:
            lookup[int(class_id)] += 1.0
    return lookup


def _token_shape_hwd(target_shape_dhw: tuple[int, int, int]) -> tuple[int, int, int]:
    d, h, w = target_shape_dhw
    return h, w, d


def _resampled_token_shape_hwd(
    source_shape_hwd: tuple[int, int, int],
    xy_spacing: float,
    z_spacing: float,
    target_shape_dhw: tuple[int, int, int],
    scale: int,
) -> tuple[int, int, int]:
    source_h, source_w, source_d = source_shape_hwd
    full_h = source_h * xy_spacing / TARGET_SPACING_DHW[1]
    full_w = source_w * xy_spacing / TARGET_SPACING_DHW[2]
    full_d = source_d * z_spacing / TARGET_SPACING_DHW[0]
    token_h, token_w, token_d = (int(x) * scale for x in _token_shape_hwd(target_shape_dhw))
    target_h, target_w, target_d = TARGET_SHAPE_HWD
    return (
        max(1, int(round(full_h * token_h / target_h))),
        max(1, int(round(full_w * token_w / target_w))),
        max(1, int(round(full_d * token_d / target_d))),
    )


def _center_crop_pad_to_shape_hwd(array: np.ndarray, target_shape_hwd: tuple[int, int, int]) -> np.ndarray:
    if target_shape_hwd == TARGET_SHAPE_HWD:
        return _center_crop_pad_hwd(array)
    target_h, target_w, target_d = target_shape_hwd
    h, w, d = array.shape
    h_start = max((h - target_h) // 2, 0)
    h_end = min(h_start + target_h, h)
    w_start = max((w - target_w) // 2, 0)
    w_end = min(w_start + target_w, w)
    d_start = max((d - target_d) // 2, 0)
    d_end = min(d_start + target_d, d)
    cropped = array[h_start:h_end, w_start:w_end, d_start:d_end]

    out = np.zeros(target_shape_hwd, dtype=array.dtype)
    crop_h, crop_w, crop_d = cropped.shape
    pad_h0 = (target_h - crop_h) // 2
    pad_w0 = (target_w - crop_w) // 2
    pad_d0 = (target_d - crop_d) // 2
    out[
        pad_h0 : pad_h0 + crop_h,
        pad_w0 : pad_w0 + crop_w,
        pad_d0 : pad_d0 + crop_d,
    ] = cropped
    return out


def build_organ_soft_fast_token(
    volume_id: str,
    ts_total_root: str | Path,
    metadata_df: pd.DataFrame,
    ontology: ClinicalOntology,
    target_shape_dhw: tuple[int, int, int],
    scale: int = 1,
) -> np.ndarray:
    """Approximate organ_soft directly at BTB3D token resolution.

    The reference builder resamples the TS label mask into full CT space and
    then average-pools a finding-count field. This fast path first converts TS
    labels to finding counts and applies the same spacing/crop/pad geometry at
    token resolution. It is intended for validation against the reference path
    before full-data use.
    """
    path = ts_mask_path(ts_total_root, volume_id)
    if not path.exists():
        raise FileNotFoundError(path)
    xy_spacing, z_spacing = _metadata_spacing(metadata_df, volume_id)
    mask_hwd = np.asanyarray(nib.load(str(path)).dataobj).astype(np.int16, copy=False)
    if mask_hwd.min() < 0 or mask_hwd.max() > 117:
        raise ValueError(f"TS mask values out of range for {volume_id}: {mask_hwd.min()}..{mask_hwd.max()}")

    lookup = organ_count_lookup(ontology)
    native_weight_hwd = lookup[mask_hwd]
    if scale < 1:
        raise ValueError(f"scale must be >=1, got {scale}")
    token_shape_hwd = tuple(int(x) * scale for x in _token_shape_hwd(target_shape_dhw))
    resampled_token_shape = _resampled_token_shape_hwd(
        source_shape_hwd=tuple(int(x) for x in native_weight_hwd.shape),
        xy_spacing=xy_spacing,
        z_spacing=z_spacing,
        target_shape_dhw=target_shape_dhw,
        scale=scale,
    )
    zoom_factors = tuple(
        resampled_token_shape[index] / native_weight_hwd.shape[index]
        for index in range(3)
    )
    resized = zoom(native_weight_hwd, zoom_factors, order=1).astype(np.float32, copy=False)
    token_hwd = _center_crop_pad_to_shape_hwd(resized, token_shape_hwd)
    token_dhw = token_hwd.transpose(2, 0, 1).astype(np.float32, copy=False)
    if scale > 1:
        token_dhw = downsample_to_token_grid(token_dhw, target_shape=target_shape_dhw, method="mean")
    if float(token_dhw.sum()) <= 0.0:
        raise ValueError(f"organ_soft fast-token: no relevant TS organs in volume {volume_id}")
    return token_dhw


def build_ts_class_basis_fast_token(
    volume_id: str,
    ts_total_root: str | Path,
    metadata_df: pd.DataFrame,
    class_ids: list[int],
    target_shape_dhw: tuple[int, int, int],
    scale: int = 4,
) -> np.ndarray:
    """Build token-level TS class fractions without full-resolution preprocessing.

    This is the basis-map analogue of :func:`build_organ_soft_fast_token`.
    The native integer TS mask is resampled with nearest-neighbor interpolation
    directly to ``scale * token_grid`` resolution, cropped/padded with the same
    geometry, then pooled to the requested token grid. The output axis order is
    ``class,z,y,x`` and values are approximate per-token class fractions.
    """
    if scale < 1:
        raise ValueError(f"scale must be >=1, got {scale}")
    if not class_ids:
        raise ValueError("class_ids must be non-empty")
    path = ts_mask_path(ts_total_root, volume_id)
    if not path.exists():
        raise FileNotFoundError(path)
    xy_spacing, z_spacing = _metadata_spacing(metadata_df, volume_id)
    mask_hwd = np.asanyarray(nib.load(str(path)).dataobj).astype(np.int16, copy=False)
    if mask_hwd.min() < 0 or mask_hwd.max() > 117:
        raise ValueError(f"TS mask values out of range for {volume_id}: {mask_hwd.min()}..{mask_hwd.max()}")

    token_shape_hwd = tuple(int(x) * scale for x in _token_shape_hwd(target_shape_dhw))
    resampled_token_shape = _resampled_token_shape_hwd(
        source_shape_hwd=tuple(int(x) for x in mask_hwd.shape),
        xy_spacing=xy_spacing,
        z_spacing=z_spacing,
        target_shape_dhw=target_shape_dhw,
        scale=scale,
    )
    zoom_factors = tuple(
        resampled_token_shape[index] / mask_hwd.shape[index]
        for index in range(3)
    )
    resized = zoom(mask_hwd, zoom_factors, order=0).astype(np.int16, copy=False)
    token_hwd = _center_crop_pad_to_shape_hwd(resized, token_shape_hwd)
    token_dhw = token_hwd.transpose(2, 0, 1)

    basis = np.empty((len(class_ids),) + target_shape_dhw, dtype=np.float32)
    for index, class_id in enumerate(class_ids):
        class_mask = (token_dhw == int(class_id)).astype(np.float32, copy=False)
        if scale > 1:
            basis[index] = downsample_to_token_grid(
                class_mask,
                target_shape=target_shape_dhw,
                method="mean",
            )
        else:
            basis[index] = class_mask
    return basis
