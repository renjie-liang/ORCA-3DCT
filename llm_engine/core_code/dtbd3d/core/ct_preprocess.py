"""CT preprocessing utilities matching the BTB3D encoder-decoder pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def resize_array(array: torch.Tensor, current_spacing, target_spacing) -> np.ndarray:
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    return F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False).cpu().numpy()


def metadata_row_for_volume(path: str | Path, metadata_df: pd.DataFrame) -> pd.Series:
    path = Path(path)
    file_name = os.path.basename(path)
    row = metadata_df[metadata_df["VolumeName"] == file_name]
    if row.empty:
        raise KeyError(f"Metadata row not found for {file_name}")
    return row.iloc[0]


def nifti_to_hu_array(nii_img: nib.Nifti1Image) -> np.ndarray:
    """Return HU values from CT-RATE fixed NIfTI files.

    These NIfTI files already store HU-like values. Do not reapply metadata
    `RescaleSlope`/`RescaleIntercept`; doing so double-shifts HU and can turn
    valid CT volumes into all-air tensors after clipping.
    """

    return np.asarray(nii_img.get_fdata(dtype=np.float32), dtype=np.float32)


def preprocess_volume(
    path: str | Path,
    metadata_df: pd.DataFrame,
) -> tuple[torch.Tensor, tuple[slice, slice, slice]]:
    """Mirror author preprocessing: HU rescale, resample, clip, normalize, crop/pad.

    Returns:
        tensor: `(1, 1, D, H, W)` normalized to `[-1, 1]`
        valid_slices: ROI slices in `(D,H,W)` order excluding crop/pad padding
    """

    path = Path(path)
    nii_img = nib.load(str(path))
    row = metadata_row_for_volume(path, metadata_df)

    xy_spacing = float(row["XYSpacing"][1:][:-2].split(",")[0])
    z_spacing = float(row["ZSpacing"])

    img_data = nifti_to_hu_array(nii_img)
    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    img_data = resize_array(tensor, (z_spacing, xy_spacing, xy_spacing), (1.5, 0.75, 0.75))
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = np.clip(img_data, -1000, 1000)
    img_data = (img_data / 1000.0).astype(np.float32)

    tensor = torch.tensor(img_data)
    h, w, d = tensor.shape
    dh, dw, dd = 512, 512, 241
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    crop_h, crop_w, crop_d = tensor.size(0), tensor.size(1), tensor.size(2)
    ph0 = (dh - crop_h) // 2
    ph1 = dh - crop_h - ph0
    pw0 = (dw - crop_w) // 2
    pw1 = dw - crop_w - pw0
    pd0 = (dd - crop_d) // 2
    pd1 = dd - crop_d - pd0
    tensor = F.pad(tensor, (pd0, pd1, pw0, pw1, ph0, ph1), value=-1)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    valid_slices = (slice(pd0, pd0 + crop_d), slice(ph0, ph0 + crop_h), slice(pw0, pw0 + crop_w))
    return tensor, valid_slices


def center_crop_axis2(tensor: torch.Tensor) -> torch.Tensor:
    """Crop depth axis to `1 + 4*n`, matching the tokenizer requirement."""

    depth = tensor.shape[2]
    n = (depth - 1) // 4
    new_size = 1 + 4 * n
    start = (depth - new_size) // 2
    slices = [slice(None)] * tensor.ndim
    slices[2] = slice(start, start + new_size)
    return tensor[tuple(slices)]
