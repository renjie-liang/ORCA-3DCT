#!/usr/bin/env python
"""Build a project-local cache of preprocessed CT tensors for encoder training.

The cache stores deterministic preprocessing output only:

    NIfTI HU + metadata geometry -> preprocess_volume() -> center_crop_axis2() -> .npy

Training still performs the random depth crop at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dtbd3d.core.ct_preprocess import center_crop_axis2, preprocess_volume

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def volume_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "valid"])
    parser.add_argument("--input", nargs="+", default=[], help="NIfTI volume paths.")
    parser.add_argument("--input-list", default=None, help="Text file with one NIfTI path per line.")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--out-dir", required=True, help="Output split directory, e.g. ct_rate_cache/train.")
    parser.add_argument("--shard-metadata-dir", required=True, help="Directory for per-shard ids/config/metrics.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if bool(args.input) == bool(args.input_list):
        raise ValueError("Provide exactly one of --input or --input-list")
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >=1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")

    if args.input_list:
        lines = Path(args.input_list).read_text().splitlines()
        paths = [Path(line.strip()) for line in lines if line.strip()]
    else:
        paths = [Path(item) for item in args.input]

    if args.limit is not None:
        paths = paths[: args.limit]
    return [path for idx, path in enumerate(paths) if idx % args.num_shards == args.shard_index]


def iter_progress(items: list[Path]):
    if tqdm is not None:
        return tqdm(items, desc="preprocess cache", unit="vol")
    return items


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_metadata_dir = Path(args.shard_metadata_dir)
    shard_metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_df = pd.read_csv(args.metadata_csv)
    dtype = np.float16 if args.dtype == "float16" else np.float32
    shard_tag = f"{args.split}_shard_{args.shard_index:03d}"

    input_paths = resolve_inputs(args)
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    (shard_metadata_dir / f"{shard_tag}_config.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "input_list": args.input_list,
                "metadata_csv": args.metadata_csv,
                "out_dir": str(out_dir),
                "dtype": args.dtype,
                "limit": args.limit,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            },
            indent=2,
        )
    )

    ids: list[str] = []
    metrics_path = shard_metadata_dir / f"{shard_tag}_cache_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "shard_index", "volume_id", "source_path", "shape", "dtype", "seconds", "output_bytes"])
        progress = iter_progress(input_paths)
        for path in progress:
            volume_id = volume_id_from_path(path)
            if tqdm is not None:
                progress.set_postfix_str(volume_id)
            out_path = out_dir / f"{volume_id}.npy"
            if out_path.exists() and not args.overwrite:
                ids.append(volume_id)
                arr = np.load(out_path, mmap_mode="r")
                writer.writerow(
                    [args.split, args.shard_index, volume_id, str(path), tuple(arr.shape), str(arr.dtype), "0.0000", out_path.stat().st_size]
                )
                continue

            start = time.perf_counter()
            tensor, _ = preprocess_volume(path, metadata_df)
            tensor = center_crop_axis2(tensor)
            array = tensor.numpy().astype(dtype, copy=False)
            tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}.npy")
            np.save(tmp_path, array)
            tmp_path.replace(out_path)
            seconds = time.perf_counter() - start
            ids.append(volume_id)
            writer.writerow(
                [args.split, args.shard_index, volume_id, str(path), tuple(array.shape), str(array.dtype), f"{seconds:.4f}", out_path.stat().st_size]
            )
            handle.flush()
            message = f"cached {volume_id} shape={tuple(array.shape)} dtype={array.dtype} seconds={seconds:.2f} path={out_path}"
            if tqdm is not None:
                progress.set_postfix_str(f"{volume_id} {seconds:.2f}s")
                tqdm.write(message)
            else:
                print(message)

    (shard_metadata_dir / f"{shard_tag}_ids.txt").write_text("\n".join(ids) + "\n")
    print(f"split={args.split}")
    print(f"tensor_dir={out_dir}")
    print(f"shard_metadata_dir={shard_metadata_dir}")
    print(f"metrics={metrics_path}")
    print(f"n_volumes={len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
