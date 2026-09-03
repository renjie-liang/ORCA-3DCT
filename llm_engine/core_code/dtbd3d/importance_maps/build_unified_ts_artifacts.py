#!/usr/bin/env python
"""Build unified token-grid TS artifacts for diagnostic token learning.

This augments the existing organ-soft maps in
``maps_organ_soft_base_16x16x8/{train,valid}`` with compact ``.npz`` files.
The old ``.npy`` files are intentionally left in place during migration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dtbd3d.importance_maps import AnatomicalImportanceMapBuilder, ClinicalOntology, load_ontology
from dtbd3d.importance_maps.builder import TOKEN_SHAPES
from dtbd3d.importance_maps.fast_token import build_organ_soft_fast_token, build_ts_class_basis_fast_token


@dataclass(frozen=True)
class OrganGroup:
    name: str
    aliases: tuple[str, ...]
    is_proxy: bool
    note: str


DEFAULT_ORGAN_GROUPS = (
    OrganGroup("lung", ("lung_lobes",), False, "TS lung lobe classes."),
    OrganGroup(
        "cardiomediastinal",
        ("mediastinum_proxy",),
        True,
        "Coarse TS proxy for mediastinal/cardiovascular findings.",
    ),
    OrganGroup("hiatal", ("hiatal_region_proxy",), True, "Esophagus plus stomach proxy."),
    OrganGroup("airway", ("central_airway_proxy",), True, "Trachea-only TS airway proxy."),
    OrganGroup("pleura_proxy", ("pleura_proxy",), True, "Pleura is not a TS class; lung lobes are used as proxy."),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_links = root / "data"
    cache_index = Path("./data/volumes/prep_npy_index")
    default_out = Path("./data/token_importance/maps_organ_soft_base_16x16x8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "valid", "both"], required=True)
    parser.add_argument("--compression", choices=sorted(TOKEN_SHAPES), default="16x16x8")
    parser.add_argument("--out-dir", default=str(default_out))
    parser.add_argument("--train-ids-file", default=str(cache_index / "train_ids.txt"))
    parser.add_argument("--valid-ids-file", default=str(cache_index / "valid_ids.txt"))
    parser.add_argument("--ontology-yaml", default=str(root / "Experiment" / "configs" / "clinical_ontology_ts_v2.yaml"))
    parser.add_argument("--ts-total-root", default=str(data_links / "ct_rate" / "dataset" / "ts_seg" / "ts_total"))
    parser.add_argument("--train-metadata-csv", default=str(data_links / "ct_rate" / "dataset" / "metadata" / "train_metadata.csv"))
    parser.add_argument("--valid-metadata-csv", default=str(data_links / "ct_rate" / "dataset" / "metadata" / "validation_metadata.csv"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fast-token-scale", type=int, default=4)
    parser.add_argument("--group-min-fraction", type=float, default=0.0)
    parser.add_argument("--importance-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--recompute-importance",
        action="store_true",
        help="Recompute organ-soft importance instead of reusing an existing <volume_id>.npy when present.",
    )
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_names(split: str) -> list[str]:
    if split == "both":
        return ["train", "valid"]
    return [split]


def shard_ids(ids: list[str], num_shards: int, shard_id: int, limit: int) -> list[str]:
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
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


def resolve_organ_ids(ontology: ClinicalOntology, names: Iterable[str]) -> frozenset[int]:
    resolved: set[int] = set()

    def resolve_one(name: str, stack: tuple[str, ...]) -> None:
        if name in ontology.ts_class_id:
            resolved.add(int(ontology.ts_class_id[name]))
            return
        if name not in ontology.organ_alias:
            raise KeyError(f"Unknown TS organ alias/class: {name}")
        if name in stack:
            raise ValueError(f"Cyclic TS organ alias: {' -> '.join(stack + (name,))}")
        for child in ontology.organ_alias[name]:
            resolve_one(child, stack + (name,))

    for name in names:
        resolve_one(str(name), ())
    return frozenset(resolved)


def default_group_specs(ontology: ClinicalOntology) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for bit, group in enumerate(DEFAULT_ORGAN_GROUPS):
        class_ids = sorted(resolve_organ_ids(ontology, group.aliases))
        specs.append(
            {
                "bit": bit,
                "name": group.name,
                "aliases": list(group.aliases),
                "ts_class_ids": class_ids,
                "is_proxy": group.is_proxy,
                "note": group.note,
            }
        )
    return specs


def group_bitmask_from_basis(
    basis: np.ndarray,
    class_ids: list[int],
    group_specs: list[dict[str, object]],
    min_fraction: float,
) -> np.ndarray:
    if basis.ndim != 4:
        raise ValueError(f"basis must have shape [class,z,y,x], got {basis.shape}")
    if basis.shape[0] != len(class_ids):
        raise ValueError(f"basis class axis {basis.shape[0]} != len(class_ids) {len(class_ids)}")
    class_index = {int(class_id): index for index, class_id in enumerate(class_ids)}
    bitmask = np.zeros(basis.shape[1:], dtype=np.uint8)
    threshold = float(min_fraction)
    for spec in group_specs:
        bit = int(spec["bit"])
        spec_class_ids = [int(class_id) for class_id in spec["ts_class_ids"]]
        indices = [class_index[class_id] for class_id in spec_class_ids if class_id in class_index]
        if not indices:
            continue
        group_fraction = basis[indices].sum(axis=0)
        active = group_fraction > threshold
        bitmask[active] |= np.uint8(1 << bit)
    return bitmask


def load_or_build_importance(
    volume_id: str,
    split_dir: Path,
    args: argparse.Namespace,
    metadata_df: pd.DataFrame,
    ontology: ClinicalOntology,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, str]:
    old_path = split_dir / f"{volume_id}.npy"
    if old_path.exists() and not args.recompute_importance:
        importance = np.load(old_path).astype(np.float32, copy=False)
        source = "existing_npy"
    else:
        builder = AnatomicalImportanceMapBuilder(
            ontology=ontology,
            variant="organ_soft",
            compression=args.compression,
            lambda_uniform=0.0,
        )
        try:
            raw = build_organ_soft_fast_token(
                volume_id=volume_id,
                ts_total_root=args.ts_total_root,
                metadata_df=metadata_df,
                ontology=ontology,
                target_shape_dhw=target_shape,
                scale=args.fast_token_scale,
            )
            importance = builder.finalize_raw(raw)
            source = "computed_fast_token"
        except ValueError as error:
            if "no relevant TS organs" not in str(error):
                raise
            importance = np.ones(target_shape, dtype=np.float32)
            source = "computed_uniform_empty_ts"
    if importance.shape != target_shape:
        raise ValueError(f"{old_path} shape {importance.shape} != expected {target_shape}")
    return importance, source


def save_npz_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp.npz")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, path)


def build_one_split(
    split: str,
    ids_file: Path,
    metadata_csv: Path,
    args: argparse.Namespace,
    ontology: ClinicalOntology,
    group_specs: list[dict[str, object]],
) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    split_dir = out_dir / split
    index_dir = out_dir / "index"
    target_shape = TOKEN_SHAPES[args.compression]
    ids_all = read_ids(ids_file)
    ids = shard_ids(ids_all, args.num_shards, args.shard_id, args.limit)
    metadata_df = pd.read_csv(metadata_csv)
    class_ids = sorted({class_id for spec in group_specs for class_id in spec["ts_class_ids"]})
    artifact_dtype = np.float16 if args.importance_dtype == "float16" else np.float32

    shard_tag = f"{split}_shard_{args.shard_id:02d}_of_{args.num_shards:02d}"
    stats_path = index_dir / f"{shard_tag}_unified_stats.csv"
    rows: list[dict[str, object]] = []
    n_written = 0
    n_skipped = 0
    n_reused_importance = 0
    n_computed_importance = 0
    n_empty_uniform = 0

    fieldnames = [
        "volume_id",
        "status",
        "importance_source",
        "importance_mean",
        "importance_max",
        "bitmask_any_fraction",
    ] + [f"group_{spec['name']}_fraction" for spec in group_specs]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for volume_id in iter_progress(ids, f"build unified TS {split} shard {args.shard_id}/{args.num_shards}"):
            artifact_path = split_dir / f"{volume_id}.npz"
            if artifact_path.exists() and not args.skip_existing and not args.overwrite:
                raise FileExistsError(f"{artifact_path} exists; use --skip-existing or --overwrite")
            if artifact_path.exists() and args.skip_existing and not args.overwrite:
                row = {
                    "volume_id": volume_id,
                    "status": "skipped_existing",
                    "importance_source": "unknown_existing_npz",
                    "importance_mean": "",
                    "importance_max": "",
                    "bitmask_any_fraction": "",
                }
                for spec in group_specs:
                    row[f"group_{spec['name']}_fraction"] = ""
                writer.writerow(row)
                rows.append(row)
                n_skipped += 1
                continue

            importance, importance_source = load_or_build_importance(
                volume_id=volume_id,
                split_dir=split_dir,
                args=args,
                metadata_df=metadata_df,
                ontology=ontology,
                target_shape=target_shape,
            )
            if importance_source == "existing_npy":
                n_reused_importance += 1
            elif importance_source == "computed_uniform_empty_ts":
                n_computed_importance += 1
                n_empty_uniform += 1
            else:
                n_computed_importance += 1

            basis = build_ts_class_basis_fast_token(
                volume_id=volume_id,
                ts_total_root=args.ts_total_root,
                metadata_df=metadata_df,
                class_ids=class_ids,
                target_shape_dhw=target_shape,
                scale=args.fast_token_scale,
            )
            bitmask = group_bitmask_from_basis(
                basis=basis,
                class_ids=class_ids,
                group_specs=group_specs,
                min_fraction=args.group_min_fraction,
            )
            save_npz_atomic(
                artifact_path,
                importance_token=importance.astype(artifact_dtype, copy=False),
                organ_group_bitmask_token=bitmask,
                organ_group_names=np.array([str(spec["name"]) for spec in group_specs]),
                organ_group_bits=np.array([int(spec["bit"]) for spec in group_specs], dtype=np.uint8),
            )
            n_written += 1

            row = {
                "volume_id": volume_id,
                "status": "written",
                "importance_source": importance_source,
                "importance_mean": f"{float(importance.mean()):.8f}",
                "importance_max": f"{float(importance.max()):.8f}",
                "bitmask_any_fraction": f"{float((bitmask > 0).mean()):.8f}",
            }
            for spec in group_specs:
                bit = int(spec["bit"])
                row[f"group_{spec['name']}_fraction"] = f"{float(((bitmask & (1 << bit)) > 0).mean()):.8f}"
            writer.writerow(row)
            handle.flush()
            rows.append(row)

    manifest = {
        "split": split,
        "compression": args.compression,
        "target_shape_dhw": list(target_shape),
        "ids_file": str(ids_file),
        "ids_total": len(ids_all),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "shard_policy": "round_robin_index_mod",
        "limit": args.limit,
        "n_selected": len(ids),
        "n_written": n_written,
        "n_skipped_existing": n_skipped,
        "n_reused_importance_npy": n_reused_importance,
        "n_computed_importance": n_computed_importance,
        "n_empty_uniform_importance": n_empty_uniform,
        "out_dir": str(out_dir),
        "split_dir": str(split_dir),
        "stats_csv": str(stats_path),
        "created_at": datetime.now().isoformat(),
    }
    write_json_atomic(index_dir / f"{shard_tag}_unified_manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")
    if args.fast_token_scale < 1:
        raise ValueError("--fast-token-scale must be >= 1")
    if args.group_min_fraction < 0.0:
        raise ValueError("--group-min-fraction must be >= 0")

    ontology = load_ontology(args.ontology_yaml)
    group_specs = default_group_specs(ontology)
    out_dir = Path(args.out_dir)
    index_dir = out_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    root_metadata = {
        "artifact_version": "ts_unified_token_grid_v1",
        "artifact_layout": {
            "per_volume_npz": "<out_dir>/<split>/<volume_id>.npz",
            "legacy_importance_npy": "<out_dir>/<split>/<volume_id>.npy",
        },
        "arrays": {
            "importance_token": {
                "shape": list(TOKEN_SHAPES[args.compression]),
                "dtype": args.importance_dtype,
                "meaning": "Organ-soft reconstruction/loss weight, normalized to mean 1.",
            },
            "organ_group_bitmask_token": {
                "shape": list(TOKEN_SHAPES[args.compression]),
                "dtype": "uint8",
                "meaning": "Bitmask over token-grid organ groups; token can belong to multiple groups.",
            },
        },
        "organ_groups": group_specs,
        "ontology_yaml": str(args.ontology_yaml),
        "ts_total_root_contract": str(args.ts_total_root),
        "group_min_fraction": args.group_min_fraction,
        "fast_token_scale": args.fast_token_scale,
        "created_at": datetime.now().isoformat(),
    }
    if args.shard_id == 0:
        write_json_atomic(index_dir / "unified_ts_artifact_metadata.json", root_metadata)

    manifests: list[dict[str, object]] = []
    for split in split_names(args.split):
        ids_file = Path(args.train_ids_file if split == "train" else args.valid_ids_file)
        metadata_csv = Path(args.train_metadata_csv if split == "train" else args.valid_metadata_csv)
        manifests.append(
            build_one_split(
                split=split,
                ids_file=ids_file,
                metadata_csv=metadata_csv,
                args=args,
                ontology=ontology,
                group_specs=group_specs,
            )
        )
    print(json.dumps({"status": "ok", "manifests": manifests}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
