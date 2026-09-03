#!/usr/bin/env python
"""Build RadGenome token-grid mask artifacts for diagnostic token learning.

Each output file stores only RadGenome masks projected to the BTB3D base token
grid. The artifact is intentionally small and task-agnostic; training code can
later choose which masks to load and how to turn them into supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import zoom

from dtbd3d.importance_maps.downsample import downsample_to_token_grid
from dtbd3d.importance_maps.fast_token import _resampled_token_shape_hwd
from dtbd3d.importance_maps.ts_mask_loader import _center_crop_pad_hwd


TARGET_SHAPE_DHW = (31, 64, 64)


@dataclass(frozen=True)
class CatalogEntry:
    mask_id: int
    mask_name: str
    mask_name_zh: str
    group: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_links = root / "data"
    cache_index = Path("./data/volumes/prep_npy_index")
    default_out = Path("./data/organ_masks/radgenome_8x8x8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "valid", "both"], required=True)
    parser.add_argument("--out-dir", default=str(default_out))
    parser.add_argument("--radgenome-root", default=str(data_links / "radgenome" / "dataset"))
    parser.add_argument("--catalog-yaml", default=str(root / "Experiment" / "configs" / "radgenome_mask_catalog_curated.yaml"))
    parser.add_argument("--train-metadata-csv", default=str(data_links / "ct_rate" / "dataset" / "metadata" / "train_metadata.csv"))
    parser.add_argument("--valid-metadata-csv", default=str(data_links / "ct_rate" / "dataset" / "metadata" / "validation_metadata.csv"))
    parser.add_argument("--train-ids-file", default=str(cache_index / "train_ids.txt"))
    parser.add_argument("--valid-ids-file", default=str(cache_index / "valid_ids.txt"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--projection-scale", type=int, default=1)
    parser.add_argument("--resize-order", type=int, choices=[0, 1], default=1)
    parser.add_argument("--source-priority", default="region,anatomy")
    parser.add_argument("--mask-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_names(split: str) -> list[str]:
    return ["train", "valid"] if split == "both" else [split]


def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def shard_ids(ids: list[str], num_shards: int, shard_id: int, limit: int) -> list[str]:
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >=1, got {num_shards}")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"--shard-id must be in [0,{num_shards - 1}], got {shard_id}")
    selected = ids[shard_id::num_shards]
    if limit > 0:
        selected = selected[:limit]
    return selected


def iter_progress(items: list[str], description: str) -> Iterable[str]:
    try:
        from tqdm import tqdm

        yield from tqdm(items, desc=description, unit="vol")
    except ImportError:
        total = len(items)
        for index, item in enumerate(items, start=1):
            if index == 1 or index == total or index % 25 == 0:
                print(f"{description}: {index}/{total}", flush=True)
            yield item


def load_catalog(path: Path) -> tuple[list[CatalogEntry], dict[str, object]]:
    doc = yaml.safe_load(path.read_text())
    entries: list[CatalogEntry] = []
    mask_id = 0
    for group in doc.get("groups", []):
        group_name = str(group["name"])
        zh_lookup = group.get("mask_labels_zh", {}) or {}
        for label in group.get("mask_labels", []):
            label = str(label)
            entries.append(
                CatalogEntry(
                    mask_id=mask_id,
                    mask_name=label,
                    mask_name_zh=str(zh_lookup.get(label, "")),
                    group=group_name,
                )
            )
            mask_id += 1
    if not entries:
        raise ValueError(f"No keep labels found in catalog: {path}")
    return entries, doc


def metadata_lookup(path: Path) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(path)
    out: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        volume_name = str(row["VolumeName"])
        volume_id = volume_name[:-7] if volume_name.endswith(".nii.gz") else Path(volume_name).stem
        out[volume_id] = row
    return out


def row_shape_spacing(row: dict[str, object], volume_id: str) -> tuple[tuple[int, int, int], float, float]:
    try:
        shape_hwd = (int(row["Rows"]), int(row["Columns"]), int(row["NumberofSlices"]))
        xy_spacing_text = str(row["XYSpacing"]).strip()
        xy_spacing = float(xy_spacing_text.strip("[]").split(",")[0])
        z_spacing = float(row["ZSpacing"])
    except Exception as error:  # pragma: no cover - defensive error context
        raise ValueError(f"Invalid metadata row for {volume_id}: {row}") from error
    return shape_hwd, xy_spacing, z_spacing


def mask_root(radgenome_root: Path, split: str, source: str) -> Path:
    return radgenome_root / f"{split}_{source}_mask"


def volume_dir(radgenome_root: Path, split: str, source: str, volume_id: str) -> Path:
    return mask_root(radgenome_root, split, source) / f"seg_{volume_id}"


def available_volume_ids(radgenome_root: Path, split: str) -> list[str]:
    ids: set[str] = set()
    for source in ("region", "anatomy"):
        root = mask_root(radgenome_root, split, source)
        if not root.exists():
            continue
        for entry in os.scandir(root):
            if entry.is_dir() and entry.name.startswith("seg_"):
                ids.add(entry.name[4:])
    return sorted(ids)


def choose_mask_path(
    radgenome_root: Path,
    split: str,
    volume_id: str,
    mask_name: str,
    source_priority: tuple[str, ...],
) -> tuple[Path | None, str | None]:
    filename = f"{mask_name}.nii.gz"
    for source in source_priority:
        path = volume_dir(radgenome_root, split, source, volume_id) / filename
        if path.exists():
            return path, source
    return None, None


def center_crop_pad_to_shape_hwd(array: np.ndarray, target_shape_hwd: tuple[int, int, int]) -> np.ndarray:
    if target_shape_hwd == (512, 512, 241):
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
    out[pad_h0 : pad_h0 + crop_h, pad_w0 : pad_w0 + crop_w, pad_d0 : pad_d0 + crop_d] = cropped
    return out


def project_mask_to_token_grid(
    mask_hwd: np.ndarray,
    ct_shape_hwd: tuple[int, int, int],
    xy_spacing: float,
    z_spacing: float,
    target_shape_dhw: tuple[int, int, int],
    scale: int,
    resize_order: int,
) -> np.ndarray:
    if scale < 1:
        raise ValueError(f"scale must be >=1, got {scale}")
    mask_hwd = (mask_hwd > 0).astype(np.float32, copy=False)
    mask_hwd = mask_hwd[::-1, ::-1, :]
    token_shape_hwd = (target_shape_dhw[1] * scale, target_shape_dhw[2] * scale, target_shape_dhw[0] * scale)
    resampled_token_shape = _resampled_token_shape_hwd(
        source_shape_hwd=ct_shape_hwd,
        xy_spacing=xy_spacing,
        z_spacing=z_spacing,
        target_shape_dhw=target_shape_dhw,
        scale=scale,
    )
    zoom_factors = tuple(resampled_token_shape[index] / mask_hwd.shape[index] for index in range(3))
    resized = zoom(mask_hwd, zoom_factors, order=resize_order).astype(np.float32, copy=False)
    token_hwd = center_crop_pad_to_shape_hwd(resized, token_shape_hwd)
    token_dhw = np.clip(token_hwd.transpose(2, 0, 1), 0.0, 1.0).astype(np.float32, copy=False)
    if scale > 1:
        token_dhw = downsample_to_token_grid(token_dhw, target_shape=target_shape_dhw, method="mean")
    if token_dhw.shape != target_shape_dhw:
        raise ValueError(f"Projected mask shape {token_dhw.shape} != expected {target_shape_dhw}")
    return token_dhw


def save_npz_atomic(path: Path, **arrays: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    tmp_npz = tmp.with_suffix(tmp.suffix + ".npz")
    if tmp_npz.exists():
        tmp_npz.replace(path)
    else:
        tmp.replace(path)


def process_volume(
    volume_id: str,
    split: str,
    out_path: Path,
    radgenome_root: Path,
    catalog: list[CatalogEntry],
    metadata: dict[str, dict[str, object]],
    source_priority: tuple[str, ...],
    mask_dtype: np.dtype,
    projection_scale: int,
    resize_order: int,
) -> dict[str, object]:
    if volume_id not in metadata:
        raise KeyError(f"Metadata missing for {volume_id}")
    ct_shape_hwd, xy_spacing, z_spacing = row_shape_spacing(metadata[volume_id], volume_id)

    masks: list[np.ndarray] = []
    mask_ids: list[int] = []
    mask_names: list[str] = []
    mask_groups: list[str] = []
    mask_sources: list[str] = []
    missing = 0
    empty_after_projection = 0
    corrupt_source_masks: list[str] = []

    for entry in catalog:
        path, source = choose_mask_path(radgenome_root, split, volume_id, entry.mask_name, source_priority)
        if path is None or source is None:
            missing += 1
            continue
        try:
            mask_hwd = np.asanyarray(nib.load(str(path)).dataobj)
        except Exception as error:
            corrupt_source_masks.append(f"{source}:{entry.mask_name}:{type(error).__name__}")
            continue
        token = project_mask_to_token_grid(
            mask_hwd=mask_hwd,
            ct_shape_hwd=ct_shape_hwd,
            xy_spacing=xy_spacing,
            z_spacing=z_spacing,
            target_shape_dhw=TARGET_SHAPE_DHW,
            scale=projection_scale,
            resize_order=resize_order,
        )
        if float(token.sum()) <= 0.0:
            empty_after_projection += 1
            continue
        masks.append(token.astype(mask_dtype, copy=False))
        mask_ids.append(entry.mask_id)
        mask_names.append(entry.mask_name)
        mask_groups.append(entry.group)
        mask_sources.append(source)

    if masks:
        mask_token = np.stack(masks, axis=0).astype(mask_dtype, copy=False)
    else:
        mask_token = np.zeros((0,) + TARGET_SHAPE_DHW, dtype=mask_dtype)
    save_npz_atomic(
        out_path,
        mask_token=mask_token,
        mask_ids=np.asarray(mask_ids, dtype=np.int32),
        mask_names=np.asarray(mask_names, dtype=str),
        mask_groups=np.asarray(mask_groups, dtype=str),
        mask_sources=np.asarray(mask_sources, dtype=str),
    )
    return {
        "volume_id": volume_id,
        "split": split,
        "status": "ok",
        "n_masks": len(mask_ids),
        "n_missing_catalog_masks": missing,
        "n_empty_after_projection": empty_after_projection,
        "n_corrupt_source_masks": len(corrupt_source_masks),
        "corrupt_source_masks": ";".join(corrupt_source_masks),
        "output": str(out_path),
    }


def write_metadata(out_dir: Path, args: argparse.Namespace, catalog: list[CatalogEntry], catalog_doc: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact": "radgenome_mask_8x8x8",
        "created_at": utc_now(),
        "mask_shape_dhw": list(TARGET_SHAPE_DHW),
        "mask_dtype": args.mask_dtype,
        "projection_scale": args.projection_scale,
        "resize_order": args.resize_order,
        "source_priority": [x.strip() for x in args.source_priority.split(",") if x.strip()],
        "radgenome_root": str(Path(args.radgenome_root).resolve()),
        "catalog_yaml": str(Path(args.catalog_yaml).resolve()),
        "catalog_version": catalog_doc.get("version"),
        "storage_contract": {
            "per_volume_npz": "<radgenome_mask_8x8x8>/<split>/<volume_id>.npz",
            "mask_token": "[M_present,31,64,64] float16|float32",
            "mask_ids": "[M_present] int32",
            "mask_names": "[M_present] string",
            "mask_groups": "[M_present] string",
            "mask_sources": "[M_present] string, region|anatomy",
        },
        "catalog": [
            {
                "mask_id": entry.mask_id,
                "mask_name": entry.mask_name,
                "mask_name_zh": entry.mask_name_zh,
                "group": entry.group,
            }
            for entry in catalog
        ],
    }
    tmp = out_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out_dir / "metadata.json")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    radgenome_root = Path(args.radgenome_root)
    catalog, catalog_doc = load_catalog(Path(args.catalog_yaml))
    source_priority = tuple(x.strip() for x in args.source_priority.split(",") if x.strip())
    if not source_priority or any(x not in {"region", "anatomy"} for x in source_priority):
        raise ValueError("--source-priority must contain region and/or anatomy")
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite cannot both be set")
    if args.shard_id == 0:
        write_metadata(out_dir, args, catalog, catalog_doc)

    metadata_by_split = {
        "train": metadata_lookup(Path(args.train_metadata_csv)),
        "valid": metadata_lookup(Path(args.valid_metadata_csv)),
    }
    cache_ids_by_split = {
        "train": read_ids(Path(args.train_ids_file)),
        "valid": read_ids(Path(args.valid_ids_file)),
    }
    mask_dtype = np.dtype(args.mask_dtype)
    stats_dir = out_dir / "index" / "shard_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "created_at": utc_now(),
        "split": args.split,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "limit": args.limit,
        "stats": {},
    }

    for split in split_names(args.split):
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        ids = available_volume_ids(radgenome_root, split)
        cache_ids = cache_ids_by_split[split]
        if cache_ids:
            ids = [volume_id for volume_id in ids if volume_id in cache_ids]
        selected = shard_ids(ids, args.num_shards, args.shard_id, args.limit)
        ok = skipped = failed = 0
        for volume_id in iter_progress(selected, f"{split} shard {args.shard_id}/{args.num_shards}"):
            out_path = split_out / f"{volume_id}.npz"
            if out_path.exists() and args.skip_existing and not args.overwrite:
                skipped += 1
                stats_rows.append({"volume_id": volume_id, "split": split, "status": "skipped_existing", "output": str(out_path)})
                continue
            try:
                row = process_volume(
                    volume_id=volume_id,
                    split=split,
                    out_path=out_path,
                    radgenome_root=radgenome_root,
                    catalog=catalog,
                    metadata=metadata_by_split[split],
                    source_priority=source_priority,
                    mask_dtype=mask_dtype,
                    projection_scale=args.projection_scale,
                    resize_order=args.resize_order,
                )
                ok += 1
                stats_rows.append(row)
            except Exception as error:  # pragma: no cover - keeps long array jobs debuggable
                failed += 1
                stats_rows.append(
                    {
                        "volume_id": volume_id,
                        "split": split,
                        "status": "failed",
                        "error": repr(error),
                        "output": str(out_path),
                    }
                )
                print(f"failed {split} {volume_id}: {error!r}", flush=True)
        manifest["stats"][split] = {
            "available_ids": len(ids),
            "selected_ids": len(selected),
            "ok": ok,
            "skipped": skipped,
            "failed": failed,
        }

    stats_path = stats_dir / f"shard_{args.shard_id:04d}_of_{args.num_shards:04d}.csv"
    fieldnames = sorted({key for row in stats_rows for key in row})
    with stats_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
    manifest["stats_csv"] = str(stats_path)
    manifest_path = stats_dir / f"shard_{args.shard_id:04d}_of_{args.num_shards:04d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
