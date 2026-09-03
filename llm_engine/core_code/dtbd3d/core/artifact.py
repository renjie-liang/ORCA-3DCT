"""BTB3D token artifact read/write helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def read_ids(ids_path: str | Path) -> list[str]:
    ids_path = Path(ids_path)
    if not ids_path.exists():
        return []
    return [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]


def update_ids(ids_path: str | Path, new_ids: list[str]) -> list[str]:
    """Append ids while preserving first-seen order and write atomically."""

    ids_path = Path(ids_path)
    existing = read_ids(ids_path)
    merged = list(dict.fromkeys(existing + new_ids))
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ids_path.with_suffix(ids_path.suffix + ".tmp")
    tmp.write_text("\n".join(merged) + ("\n" if merged else ""))
    os.replace(tmp, ids_path)
    return merged


def token_path(token_dir: str | Path, volume_id: str) -> Path:
    return Path(token_dir) / "tokens" / f"{volume_id}.npy"


def save_token_row(token_dir: str | Path, volume_id: str, tokens: np.ndarray) -> Path:
    out_path = token_path(token_dir, volume_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, tokens.astype(np.uint32, copy=False))
    return out_path


def load_token_row(
    token_dir: str | Path,
    volume_id: str,
    id_to_idx: dict[str, int] | None = None,
    matrix=None,
) -> np.ndarray:
    token_dir = Path(token_dir)
    per_volume_path = token_path(token_dir, volume_id)
    if per_volume_path.exists():
        return np.load(per_volume_path).astype(np.uint32, copy=False)
    if id_to_idx is None or matrix is None:
        raise FileNotFoundError(f"Missing {per_volume_path} and no tokens_int.npy fallback is available")
    if volume_id not in id_to_idx:
        raise KeyError(f"{volume_id} not found in ids.txt")
    return np.asarray(matrix[id_to_idx[volume_id]], dtype=np.uint32)


def open_matrix(token_dir: str | Path):
    token_dir = Path(token_dir)
    ids = read_ids(token_dir / "ids.txt")
    matrix_path = token_dir / "tokens_int.npy"
    if not ids or not matrix_path.exists():
        return None, None
    return {vol_id: i for i, vol_id in enumerate(ids)}, np.load(matrix_path, mmap_mode="r")


def write_tokens_matrix(token_dir: str | Path, expected_tokens: int) -> Path:
    token_dir = Path(token_dir)
    ids = read_ids(token_dir / "ids.txt")
    tokens_dir = token_dir / "tokens"
    matrix_path = token_dir / "tokens_int.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.uint32,
        shape=(len(ids), expected_tokens),
    )
    for i, volume_id in enumerate(ids):
        row_path = tokens_dir / f"{volume_id}.npy"
        if not row_path.exists():
            raise FileNotFoundError(f"Missing per-volume token file: {row_path}")
        row = np.load(row_path)
        if row.shape != (expected_tokens,):
            raise ValueError(f"{row_path} shape {row.shape}; expected {(expected_tokens,)}")
        matrix[i] = row.astype(np.uint32, copy=False)
    matrix.flush()
    return matrix_path
