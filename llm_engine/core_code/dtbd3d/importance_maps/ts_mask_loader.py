"""Load TotalSegmentator masks and align them to BTB3D CT preprocessing.

This is the only place where TS masks are moved from native CT-RATE geometry
into the BTB3D preprocessing space used by cached training tensors:

native HWD mask -> DHW spacing resize -> HWD center crop/pad -> DHW mask

The output shape matches the full preprocessed CT volume before random depth
cropping. Token-grid downsampling happens later in the importance-map variant.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


TARGET_SPACING_DHW = (1.5, 0.75, 0.75)
TARGET_SHAPE_HWD = (512, 512, 241)


def volume_parts(volume_id: str) -> tuple[str, str, str, str]:
    parts = volume_id.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected CT-RATE volume id: {volume_id}")
    return parts[0], parts[1], parts[2], parts[3]


def fixed_volume_path(fixed_root: str | Path, volume_id: str) -> Path:
    split, number, series, _index = volume_parts(volume_id)
    root = Path(fixed_root)
    return root / f"{split}_{number}" / f"{split}_{number}_{series}" / f"{volume_id}.nii.gz"


def ts_mask_path(ts_total_root: str | Path, volume_id: str) -> Path:
    split, number, series, _index = volume_parts(volume_id)
    root = Path(ts_total_root)
    return root / f"{split}_fixed" / f"{split}_{number}" / f"{split}_{number}_{series}" / f"{volume_id}.nii.gz"


def _metadata_spacing(metadata_df: pd.DataFrame, volume_id: str) -> tuple[float, float]:
    volume_name = f"{volume_id}.nii.gz"
    row = metadata_df[metadata_df["VolumeName"] == volume_name]
    if row.empty:
        raise KeyError(f"Metadata row not found for {volume_name}")
    xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
    z_spacing = float(row["ZSpacing"].iloc[0])
    return xy_spacing, z_spacing


def _center_crop_pad_hwd(array: np.ndarray) -> np.ndarray:
    target_h, target_w, target_d = TARGET_SHAPE_HWD
    h, w, d = array.shape
    h_start = max((h - target_h) // 2, 0)
    h_end = min(h_start + target_h, h)
    w_start = max((w - target_w) // 2, 0)
    w_end = min(w_start + target_w, w)
    d_start = max((d - target_d) // 2, 0)
    d_end = min(d_start + target_d, d)
    cropped = array[h_start:h_end, w_start:w_end, d_start:d_end]

    out = np.zeros(TARGET_SHAPE_HWD, dtype=array.dtype)
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


def preprocess_ts_mask(mask_hwd: np.ndarray, xy_spacing: float, z_spacing: float) -> np.ndarray:
    mask_dhw = mask_hwd.transpose(2, 0, 1)
    source_shape = mask_dhw.shape
    target_shape = (
        int(source_shape[0] * z_spacing / TARGET_SPACING_DHW[0]),
        int(source_shape[1] * xy_spacing / TARGET_SPACING_DHW[1]),
        int(source_shape[2] * xy_spacing / TARGET_SPACING_DHW[2]),
    )
    zoom_factors = tuple(target_shape[index] / source_shape[index] for index in range(3))
    resized_dhw = zoom(mask_dhw, zoom_factors, order=0)
    resized_hwd = resized_dhw.transpose(1, 2, 0).astype(np.int16, copy=False)
    padded_hwd = _center_crop_pad_hwd(resized_hwd)
    return padded_hwd.transpose(2, 0, 1).astype(np.int16, copy=False)


def read_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(metadata_csv)


def metadata_frame(metadata_csv_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(metadata_csv_or_df, pd.DataFrame):
        return metadata_csv_or_df
    return read_metadata(metadata_csv_or_df)


def load_ts_mask(volume_id: str, ts_total_root: str | Path, metadata_csv: str | Path | pd.DataFrame) -> np.ndarray:
    path = ts_mask_path(ts_total_root, volume_id)
    if not path.exists():
        raise FileNotFoundError(path)
    metadata_df = metadata_frame(metadata_csv)
    xy_spacing, z_spacing = _metadata_spacing(metadata_df, volume_id)
    image = nib.load(str(path))
    mask_hwd = np.asanyarray(image.dataobj).astype(np.int16, copy=False)
    mask = preprocess_ts_mask(mask_hwd, xy_spacing=xy_spacing, z_spacing=z_spacing)
    if mask.min() < 0 or mask.max() > 117:
        raise ValueError(f"TS mask values out of range for {volume_id}: {mask.min()}..{mask.max()}")
    return mask
