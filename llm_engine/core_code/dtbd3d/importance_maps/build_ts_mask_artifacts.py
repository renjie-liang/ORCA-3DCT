#!/usr/bin/env python
"""Build TS token-grid mask artifacts for diagnostic token learning."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from dtbd3d.importance_maps.fast_token import build_ts_class_basis_fast_token


TARGET_SHAPE_DHW = (31, 64, 64)


@dataclass(frozen=True)
class TSMaskGroup:
    mask_id: int
    name: str
    name_zh: str
    ts_labels: tuple[str, ...]
    ts_class_ids: tuple[int, ...]


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_links = root / "data"
    cache_index = Path("./data/volumes/prep_npy_index")
    default_out = Path("./data/organ_masks/btb3d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "valid", "both"], required=True)
    parser.add_argument("--out-dir", default=str(default_out))
    parser.add_argument("--catalog-yaml", default=str(root / "Experiment" / "configs" / "ts_mask_catalog_8x8x8.yaml"))
    parser.add_argument("--ts-total-root", default=str(data_links / "ct_rate" / "dataset" / "ts_seg" / "ts_total"))
    parser.add_argument("--train-metadata-csv", default=str(data_links / "ct_rate" / "dataset" / "metadata" / "train_metadata.csv"))
    parser.add_argument("--valid-metadata-csv", default=str(data_links / "ct_rate" / "dataset" / "metadata" / "validation_metadata.csv"))
    parser.add_argument("--train-ids-file", default=str(cache_index / "train_ids.txt"))
    parser.add_argument("--valid-ids-file", default=str(cache_index / "valid_ids.txt"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fast-token-scale", type=int, default=4)
    parser.add_argument("--group-min-fraction", type=float, default=0.0)
    parser.add_argument("--mask-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_names(split: str) -> list[str]:
    return ["train", "valid"] if split == "both" else [split]


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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


def total_class_name_to_id() -> dict[str, int]:
    try:
        from totalsegmentator.map_to_binary import class_map
    except Exception as error:  # pragma: no cover - dependency/environment guard
        raise RuntimeError("totalsegmentator is required to resolve TS total class ids") from error
    total_map = class_map["total"]
    return {str(name): int(class_id) for class_id, name in total_map.items()}


def load_catalog(path: Path) -> tuple[list[TSMaskGroup], dict[str, object]]:
    doc = yaml.safe_load(path.read_text())
    label_to_id = total_class_name_to_id()
    groups: list[TSMaskGroup] = []
    for mask_id, group in enumerate(doc.get("groups", [])):
        labels = tuple(str(label) for label in group.get("ts_labels", []))
        missing = [label for label in labels if label not in label_to_id]
        if missing:
            raise KeyError(f"Unknown TotalSegmentator total labels in {path}: {missing}")
        class_ids = tuple(label_to_id[label] for label in labels)
        groups.append(
            TSMaskGroup(
                mask_id=mask_id,
                name=str(group["name"]),
                name_zh=str(group.get("name_zh", "")),
                ts_labels=labels,
                ts_class_ids=class_ids,
            )
        )
    if not groups:
        raise ValueError(f"No TS groups found in catalog: {path}")
    return groups, doc


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
    out_path: Path,
    args: argparse.Namespace,
    metadata_df: pd.DataFrame,
    groups: list[TSMaskGroup],
    all_class_ids: list[int],
    mask_dtype: np.dtype,
) -> dict[str, object]:
    basis = build_ts_class_basis_fast_token(
        volume_id=volume_id,
        ts_total_root=args.ts_total_root,
        metadata_df=metadata_df,
        class_ids=all_class_ids,
        target_shape_dhw=TARGET_SHAPE_DHW,
        scale=args.fast_token_scale,
    ).astype(np.float32, copy=False)
    class_index = {class_id: index for index, class_id in enumerate(all_class_ids)}

    masks: list[np.ndarray] = []
    mask_ids: list[int] = []
    mask_names: list[str] = []
    mask_names_zh: list[str] = []
    ts_class_ids_present: list[str] = []
    empty_after_projection = 0
    threshold = float(args.group_min_fraction)
    for group in groups:
        indices = [class_index[class_id] for class_id in group.ts_class_ids if class_id in class_index]
        token = np.clip(basis[indices].sum(axis=0), 0.0, 1.0)
        if threshold > 0:
            token = (token > threshold).astype(np.float32, copy=False)
        if float(token.sum()) <= 0.0:
            empty_after_projection += 1
            continue
        masks.append(token.astype(mask_dtype, copy=False))
        mask_ids.append(group.mask_id)
        mask_names.append(group.name)
        mask_names_zh.append(group.name_zh)
        ts_class_ids_present.append(",".join(str(class_id) for class_id in group.ts_class_ids))

    if masks:
        mask_token = np.stack(masks, axis=0).astype(mask_dtype, copy=False)
    else:
        mask_token = np.zeros((0,) + TARGET_SHAPE_DHW, dtype=mask_dtype)
    save_npz_atomic(
        out_path,
        mask_token=mask_token,
        mask_ids=np.asarray(mask_ids, dtype=np.int32),
        mask_names=np.asarray(mask_names, dtype=str),
        mask_names_zh=np.asarray(mask_names_zh, dtype=str),
        ts_class_ids=np.asarray(ts_class_ids_present, dtype=str),
    )
    return {
        "volume_id": volume_id,
        "status": "ok",
        "n_masks": len(mask_ids),
        "n_empty_after_projection": empty_after_projection,
        "output": str(out_path),
    }


def write_metadata(out_dir: Path, args: argparse.Namespace, groups: list[TSMaskGroup], catalog_doc: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact": "ts_mask_8x8x8",
        "created_at": utc_now(),
        "mask_shape_dhw": list(TARGET_SHAPE_DHW),
        "mask_dtype": args.mask_dtype,
        "fast_token_scale": args.fast_token_scale,
        "group_min_fraction": args.group_min_fraction,
        "ts_total_root": str(Path(args.ts_total_root).resolve()),
        "catalog_yaml": str(Path(args.catalog_yaml).resolve()),
        "catalog_version": catalog_doc.get("version"),
        "storage_contract": {
            "per_volume_npz": "<ts_mask_8x8x8>/<split>/<volume_id>.npz",
            "mask_token": "[M_present,31,64,64] float16|float32",
            "mask_ids": "[M_present] int32",
            "mask_names": "[M_present] string",
            "mask_names_zh": "[M_present] string",
            "ts_class_ids": "[M_present] comma-separated TS total class ids",
        },
        "catalog": [
            {
                "mask_id": group.mask_id,
                "mask_name": group.name,
                "mask_name_zh": group.name_zh,
                "ts_labels": list(group.ts_labels),
                "ts_class_ids": list(group.ts_class_ids),
            }
            for group in groups
        ],
    }
    tmp = out_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out_dir / "metadata.json")


def main() -> int:
    args = parse_args()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite cannot both be set")
    groups, catalog_doc = load_catalog(Path(args.catalog_yaml))
    all_class_ids = sorted({class_id for group in groups for class_id in group.ts_class_ids})
    out_dir = Path(args.out_dir)
    if args.shard_id == 0:
        write_metadata(out_dir, args, groups, catalog_doc)

    metadata_by_split = {
        "train": pd.read_csv(args.train_metadata_csv),
        "valid": pd.read_csv(args.valid_metadata_csv),
    }
    ids_file_by_split = {
        "train": Path(args.train_ids_file),
        "valid": Path(args.valid_ids_file),
    }
    mask_dtype = np.dtype(args.mask_dtype)
    stats_dir = out_dir / "index" / "shard_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "created_at": utc_now(),
        "artifact": "ts_mask_8x8x8",
        "split": args.split,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "limit": args.limit,
        "stats": {},
    }

    for split in split_names(args.split):
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        ids = read_ids(ids_file_by_split[split])
        selected = shard_ids(ids, args.num_shards, args.shard_id, args.limit)
        ok = skipped = failed = 0
        for volume_id in iter_progress(selected, f"TS {split} shard {args.shard_id}/{args.num_shards}"):
            out_path = split_out / f"{volume_id}.npz"
            if out_path.exists() and args.skip_existing and not args.overwrite:
                skipped += 1
                stats_rows.append({"volume_id": volume_id, "split": split, "status": "skipped_existing", "output": str(out_path)})
                continue
            try:
                row = process_volume(
                    volume_id=volume_id,
                    out_path=out_path,
                    args=args,
                    metadata_df=metadata_by_split[split],
                    groups=groups,
                    all_class_ids=all_class_ids,
                    mask_dtype=mask_dtype,
                )
                row["split"] = split
                ok += 1
                stats_rows.append(row)
            except Exception as error:  # pragma: no cover - long array job diagnostics
                failed += 1
                stats_rows.append({"volume_id": volume_id, "split": split, "status": "failed", "error": repr(error), "output": str(out_path)})
                print(f"failed {split} {volume_id}: {error!r}", flush=True)
        manifest["stats"][split] = {
            "ids_total": len(ids),
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
