#!/usr/bin/env python
"""Build a small or full split of DTBD3D anatomical importance maps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from dtbd3d.importance_maps import AnatomicalImportanceMapBuilder, load_ontology
from dtbd3d.importance_maps.fast_token import build_organ_soft_fast_token
from dtbd3d.importance_maps.ts_mask_loader import load_ts_mask


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_links = root / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["uniform", "organ_soft"], required=True)
    parser.add_argument("--compression", choices=["16x16x8", "8x8x8"], required=True)
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ontology-yaml", default=str(root / "Experiment" / "configs" / "clinical_ontology.yaml"))
    parser.add_argument("--ts-total-root", default=str(data_links / "ct_rate" / "dataset" / "ts_seg" / "ts_total"))
    parser.add_argument("--train-metadata-csv", default=str(data_links / "ct_rate" / "dataset" / "metadata" / "train_metadata.csv"))
    parser.add_argument("--valid-metadata-csv", default=str(data_links / "ct_rate" / "dataset" / "metadata" / "validation_metadata.csv"))
    parser.add_argument("--lambda-uniform", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save-maps", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing maps and include them in stats.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing maps.")
    parser.add_argument("--legacy-layout", action="store_true", help="Write maps under out-dir/maps/{split} for old smoke runs.")
    parser.add_argument(
        "--builder-mode",
        choices=["reference", "fast-token"],
        default="reference",
        help="reference matches the full-resolution TS preprocessing; fast-token approximates directly at token resolution.",
    )
    parser.add_argument(
        "--fast-token-scale",
        type=int,
        default=1,
        help="For --builder-mode fast-token, build at scale x token resolution before pooling back.",
    )
    parser.add_argument(
        "--empty-policy",
        choices=["error", "uniform"],
        default="error",
        help="How to handle volumes with no ontology-relevant TS organs.",
    )
    return parser.parse_args()


def read_ids(path: Path, limit: int) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit > 0:
        return ids[:limit]
    return ids


def needs_ts_mask(variant: str, lambda_uniform: float) -> bool:
    """Return whether this build needs to read TotalSegmentator masks."""
    return variant != "uniform" and lambda_uniform != 1.0


def should_save_map(variant: str, no_save_maps: bool) -> bool:
    """Uniform maps are implicit during training, so only non-uniform maps are saved."""
    return not no_save_maps and variant != "uniform"


def iter_progress(items: list[str], description: str) -> Iterable[str]:
    try:
        from tqdm import tqdm

        yield from tqdm(items, desc=description, unit="vol")
    except ImportError:
        total = len(items)
        for index, item in enumerate(items, start=1):
            if index == 1 or index == total or index % 100 == 0:
                print(f"{description}: {index}/{total}", flush=True)
            yield item


def map_dir_for(out_dir: Path, split: str, legacy_layout: bool) -> Path:
    if legacy_layout:
        return out_dir / "maps" / split
    return out_dir / split


def index_dir_for(out_dir: Path, legacy_layout: bool) -> Path:
    if legacy_layout:
        return out_dir
    return out_dir / "index"


def save_map_atomic(path: Path, weight: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp.npy")
    np.save(tmp_path, weight.astype(np.float32, copy=False))
    os.replace(tmp_path, path)


def is_empty_organ_error(error: Exception) -> bool:
    return "no relevant TS organs" in str(error)


def map_stats(volume_id: str, weight: np.ndarray) -> dict[str, object]:
    return {
        "volume_id": volume_id,
        "shape": "x".join(str(int(x)) for x in weight.shape),
        "sum": float(weight.sum()),
        "mean": float(weight.mean()),
        "min": float(weight.min()),
        "max": float(weight.max()),
        "sparsity_ge_0_5": float((weight >= 0.5).mean()),
        "n_active_ge_0_5": int((weight >= 0.5).sum()),
    }


def main() -> int:
    args = parse_args()
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = map_dir_for(out_dir, args.split, args.legacy_layout)
    index_dir = index_dir_for(out_dir, args.legacy_layout)
    index_dir.mkdir(parents=True, exist_ok=True)
    save_maps = should_save_map(args.variant, args.no_save_maps)
    load_masks = needs_ts_mask(args.variant, args.lambda_uniform)
    if save_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = args.train_metadata_csv if args.split == "train" else args.valid_metadata_csv
    metadata_df = pd.read_csv(metadata_csv) if load_masks else None
    ontology = load_ontology(args.ontology_yaml)
    if args.builder_mode == "fast-token" and args.variant != "organ_soft":
        raise ValueError("--builder-mode fast-token currently supports --variant organ_soft only")
    builder = AnatomicalImportanceMapBuilder(
        ontology=ontology,
        variant=args.variant,
        compression=args.compression,
        lambda_uniform=args.lambda_uniform,
    )
    ids = read_ids(Path(args.ids_file), args.limit)
    config = {
        "variant": args.variant,
        "compression": args.compression,
        "split": args.split,
        "ids_file": args.ids_file,
        "n_ids": len(ids),
        "ontology_yaml": args.ontology_yaml,
        "ts_total_root": args.ts_total_root,
        "metadata_csv": metadata_csv,
        "lambda_uniform": args.lambda_uniform,
        "save_maps": save_maps,
        "load_ts_masks": load_masks,
        "builder_mode": args.builder_mode,
        "fast_token_scale": args.fast_token_scale,
        "empty_policy": args.empty_policy,
        "layout": "legacy" if args.legacy_layout else "flat",
        "maps_dir": str(maps_dir),
        "skip_existing": args.skip_existing,
        "overwrite": args.overwrite,
    }
    (index_dir / f"{args.split}_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    rows: list[dict[str, object]] = []
    n_built = 0
    n_skipped_existing = 0
    n_empty_uniform = 0
    stats_path = index_dir / f"{args.split}_stats.csv"
    with stats_path.open("w", newline="") as handle:
        fieldnames = ["volume_id", "shape", "sum", "mean", "min", "max", "sparsity_ge_0_5", "n_active_ge_0_5"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for volume_id in iter_progress(ids, f"build {args.split} {args.compression} {args.variant}"):
            map_path = maps_dir / f"{volume_id}.npy"
            if save_maps and map_path.exists() and not args.skip_existing and not args.overwrite:
                raise FileExistsError(f"{map_path} exists; use --skip-existing or --overwrite")
            if save_maps and map_path.exists() and args.skip_existing and not args.overwrite:
                weight = np.load(map_path).astype(np.float32, copy=False)
                n_skipped_existing += 1
            else:
                try:
                    if load_masks and args.builder_mode == "fast-token":
                        assert metadata_df is not None
                        raw = build_organ_soft_fast_token(
                            volume_id=volume_id,
                            ts_total_root=args.ts_total_root,
                            metadata_df=metadata_df,
                            ontology=ontology,
                            target_shape_dhw=builder.target_shape,
                            scale=args.fast_token_scale,
                        )
                        weight = builder.finalize_raw(raw)
                    else:
                        ts_mask = None
                        if load_masks:
                            assert metadata_df is not None
                            ts_mask = load_ts_mask(volume_id, args.ts_total_root, metadata_df)
                        weight = builder.build(volume_id=volume_id, ts_mask=ts_mask)
                except ValueError as error:
                    if args.empty_policy != "uniform" or not is_empty_organ_error(error):
                        raise
                    weight = np.ones(builder.target_shape, dtype=np.float32)
                    n_empty_uniform += 1
                n_built += 1
            row = map_stats(volume_id, weight)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if save_maps and (args.overwrite or not map_path.exists()):
                save_map_atomic(map_path, weight)

    metrics = {
        "n_requested": len(ids),
        "n_volumes": len(rows),
        "n_built": n_built,
        "n_skipped_existing": n_skipped_existing,
        "n_empty_uniform": n_empty_uniform,
        "mean_map_mean": float(np.mean([float(row["mean"]) for row in rows])) if rows else 0.0,
        "mean_map_max": float(np.mean([float(row["max"]) for row in rows])) if rows else 0.0,
        "mean_sparsity_ge_0_5": float(np.mean([float(row["sparsity_ge_0_5"]) for row in rows])) if rows else 0.0,
    }
    (index_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest = {
        **config,
        **metrics,
        "created_at": datetime.now().isoformat(),
        "stats_csv": str(stats_path),
        "metrics_json": str(index_dir / f"{args.split}_metrics.json"),
    }
    (index_dir / f"{args.split}_manifest.json").write_text(json.dumps(manifest, indent=2))
    readme = [
        "# V3 Importance Map Build",
        "",
        f"- variant: {args.variant}",
        f"- compression: {args.compression}",
        f"- split: {args.split}",
        f"- maps_dir: {maps_dir}",
        f"- n_volumes: {len(rows)}",
        f"- n_built: {n_built}",
        f"- n_skipped_existing: {n_skipped_existing}",
        f"- n_empty_uniform: {n_empty_uniform}",
        f"- mean_map_mean: {metrics['mean_map_mean']:.6f}",
        f"- mean_map_max: {metrics['mean_map_max']:.6f}",
        f"- mean_sparsity_ge_0_5: {metrics['mean_sparsity_ge_0_5']:.6f}",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(f"out_dir={out_dir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
