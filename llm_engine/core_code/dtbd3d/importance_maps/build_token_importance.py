"""Precompute per-volume token-importance maps for adaptive token selection.

Reads per-volume anatomy/lesion masks ([M,31,64,64] float) and produces a
compact per-volume importance grid resized to the ReportGen base token grid
(default 16x16x8, D,H,W). Importance = max over selected mask groups of the
organ/region presence. Background (air/padding) -> ~0, anatomy/lesion -> high.

Consumed at selection time by `_select_visual_tokens(mode="organ_mask"/...)`,
which resizes this base grid to whatever packed grid the features are in.

Usage (one shard):
  python -m dtbd3d.importance_maps.build_token_importance \
    --source organ --split valid --num-shards 16 --shard-id 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ORGAN_ROOT = Path("./data/organ_masks/btb3d")
LESION_ROOT = Path("./data/organ_masks/radgenome_8x8x8")
DEFAULT_OUT = Path("./data/token_importance")
SPLIT_ID_FILE = {
    "train": "./data/ids/train_ids.txt",
    "valid": "./data/ids/valid_ids.txt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["organ", "lesion"], required=True)
    p.add_argument("--split", choices=["train", "valid"], required=True)
    # base ReportGen token grid is the stored LFQ grid 31x32x32 (D,H,W); selection
    # resizes this down to whatever packed grid the features are in (e.g. 15x16x16
    # for pack2x2x2). Precomputing at 16x16x8 mis-aligned the W axis -> use 31x32x32.
    p.add_argument("--grid-dhw", type=int, nargs=3, default=[31, 32, 32])
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    # lesion: only keep masks whose source is "region" (report-grounded finding regions)
    p.add_argument("--lesion-source", choices=["region", "anatomy", "all"], default="region")
    return p.parse_args()


def mask_root(source: str) -> Path:
    return ORGAN_ROOT if source == "organ" else LESION_ROOT


def list_ids(split: str) -> list[str]:
    # Enumerate from the on-disk masks (authoritative; avoids split-file drift).
    return sorted(p.stem for p in (ORGAN_ROOT / split).glob("*.npz"))


def presence_from_npz(path: Path, source: str, lesion_source: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        mask = np.asarray(npz["mask_token"], dtype=np.float32)  # [M,31,64,64]
        if mask.shape[0] == 0:
            return np.zeros(mask.shape[1:], dtype=np.float32)
        if source == "lesion" and lesion_source != "all" and "mask_sources" in npz.files:
            srcs = np.asarray([str(s) for s in npz["mask_sources"]])
            keep = srcs == lesion_source
            if keep.any():
                mask = mask[keep]
    return mask.max(axis=0)  # [31,64,64] any-group presence


def resize_to_grid(presence_dhw: np.ndarray, grid_dhw: tuple[int, int, int]) -> np.ndarray:
    t = torch.from_numpy(presence_dhw)[None, None]  # [1,1,D,H,W]
    out = F.adaptive_avg_pool3d(t, output_size=grid_dhw)[0, 0]
    return out.numpy().astype(np.float16, copy=False)


def main() -> int:
    args = parse_args()
    grid = tuple(int(x) for x in args.grid_dhw)
    root = mask_root(args.source)
    out_dir = Path(args.out_dir) / args.source / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list_ids(args.split)
    if args.limit:
        ids = ids[: args.limit]
    ids = [vid for i, vid in enumerate(ids) if i % args.num_shards == args.shard_id]

    n_ok = n_skip = 0
    for i, vid in enumerate(ids):
        out_path = out_dir / f"{vid}.npy"
        if out_path.exists():
            n_skip += 1
            continue
        src_path = root / args.split / f"{vid}.npz"
        if not src_path.exists():  # boundary: missing mask -> uniform (zeros) importance
            np.save(out_path, np.zeros(grid, dtype=np.float16))
            n_ok += 1
            continue
        presence = presence_from_npz(src_path, args.source, args.lesion_source)
        imp = resize_to_grid(presence, grid)
        tmp = out_dir / f"{vid}.tmp.npy"  # ends in .npy so np.save won't append
        np.save(tmp, imp)
        tmp.rename(out_path)
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"[{args.source}/{args.split} shard {args.shard_id}] {i+1}/{len(ids)} ok={n_ok}", flush=True)

    print(f"DONE {args.source}/{args.split} shard {args.shard_id}: ok={n_ok} skip={n_skip} total={len(ids)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
