"""Volume id normalization helpers."""

from __future__ import annotations

from pathlib import Path


def normalize_volume_id(value: str) -> str:
    name = Path(str(value)).name
    for suffix in (".nii.gz", ".nii", ".npy", ".npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for prefix in ("img_", "seg_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name

