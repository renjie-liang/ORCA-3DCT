#!/usr/bin/env python
"""Audit cached CT tensors and render HU-windowed central slices."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(value: str | None, root: Path) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def read_ids(path: Path, limit: int) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return ids[:limit] if limit > 0 else ids


def cache_stats(volume_id: str, path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 5:
        raise ValueError(f"expected cached tensor [B,C,D,H,W], got {array.shape}: {path}")
    data = np.asarray(array, dtype=np.float32)
    hu = data * 1000.0
    return {
        "volume_id": volume_id,
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "normalized_min": float(data.min()),
        "normalized_max": float(data.max()),
        "hu_min": float(hu.min()),
        "hu_max": float(hu.max()),
        "hu_mean": float(hu.mean()),
        "non_air_frac": float(np.count_nonzero(hu > -999.0) / hu.size),
        "hu_percentiles": {
            str(q): float(v) for q, v in zip([0, 1, 5, 50, 95, 99, 100], np.percentile(hu, [0, 1, 5, 50, 95, 99, 100]))
        },
    }


def center_slice_hu(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    data = np.asarray(array[0, 0], dtype=np.float32)
    return data[data.shape[0] // 2] * 1000.0


DEFAULT_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("w1000_l0", 1000.0, 0.0),
    ("w400_l40", 400.0, 40.0),
)


def render_slices(rows: list[dict[str, Any]], *, out_dir: Path, tag: str, window: float, level: float) -> Path:
    import os

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    vmin = level - window / 2.0
    vmax = level + window / 2.0
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(5, 5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, rows):
        image = center_slice_hu(Path(row["path"]))
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{row['volume_id']} | non_air={row['non_air_frac']:.4f} "
            f"HU=[{row['hu_min']:.0f},{row['hu_max']:.0f}]"
        )
        ax.axis("off")
    path = out_dir / f"cached_ct_window_{tag}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--first-n", type=int, default=16)
    parser.add_argument("--render-n", type=int, default=4)
    parser.add_argument("--out-dir", default="tmp/ct_window_audit")
    parser.add_argument(
        "--single-window",
        nargs=2,
        type=float,
        metavar=("WINDOW", "LEVEL"),
        help="Render only one HU window instead of the smoke/debug defaults.",
    )
    args = parser.parse_args()

    root = project_root()
    raw = yaml.safe_load(Path(args.config).read_text())
    section = raw["train"] if args.split == "train" else raw["validation"]
    cache_root = resolve_path(str(section["cache_root"]), root)
    ids_file = resolve_path(str(section["ids_file"]), root)
    cache_split = str(section.get("cache_split", args.split))
    if cache_root is None or ids_file is None:
        raise ValueError("cache_root and ids_file are required")

    ids = read_ids(ids_file, args.first_n)
    rows = []
    for volume_id in ids:
        path = cache_root / cache_split / f"{volume_id}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        row = cache_stats(volume_id, path)
        rows.append(row)
        print("[ct_window_audit] " + json.dumps(row, sort_keys=True), flush=True)

    nonempty_rows = [row for row in rows if row["non_air_frac"] > 0.0]
    render_rows = (nonempty_rows or rows)[: args.render_n]
    windows = (
        ((f"w{args.single_window[0]:g}_l{args.single_window[1]:g}", args.single_window[0], args.single_window[1]),)
        if args.single_window is not None
        else DEFAULT_WINDOWS
    )
    if render_rows:
        for tag, window, level in windows:
            png = render_slices(render_rows, out_dir=Path(args.out_dir), tag=tag, window=window, level=level)
            print(f"[ct_window_audit] wrote {png}", flush=True)
    print(
        "[ct_window_audit] summary "
        + json.dumps(
            {
                "checked": len(rows),
                "empty": sum(row["non_air_frac"] == 0.0 for row in rows),
                "nonempty": len(nonempty_rows),
                "windows": [{"tag": tag, "window": window, "level": level} for tag, window, level in windows],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
