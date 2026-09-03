#!/usr/bin/env python3
"""Inspect a BTB3D token artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from dtbd3d.core.artifact import read_ids, token_path
    from dtbd3d.core.btb3d_model import TOKEN_LAYOUTS, expected_token_count
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dtbd3d.core.artifact import read_ids, token_path
    from dtbd3d.core.btb3d_model import TOKEN_LAYOUTS, expected_token_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--token-dir", dest="token_dir", default=None, help="Token artifact directory.")
    p.add_argument("--canonical-dir", dest="token_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--compression", choices=sorted(TOKEN_LAYOUTS), required=True)
    p.add_argument("--check-per-volume", action="store_true")
    p.add_argument("--out", default=None, help="Optional JSON summary path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token_dir:
        raise SystemExit("Missing required argument: --token-dir")
    token_dir = Path(args.token_dir)
    expected_tokens = expected_token_count(args.compression)
    ids = read_ids(token_dir / "ids.txt")
    matrix_path = token_dir / "tokens_int.npy"
    tokens_dir = token_dir / "tokens"

    summary = {
        "token_dir": str(token_dir),
        "compression": args.compression,
        "expected_tokens_per_volume": expected_tokens,
        "num_ids": len(ids),
        "tokens_dir_exists": tokens_dir.exists(),
        "tokens_int_exists": matrix_path.exists(),
        "tokens_int_shape": None,
        "tokens_int_dtype": None,
        "num_token_files": None,
        "missing_token_files": [],
        "bad_token_shapes": [],
    }

    if matrix_path.exists():
        matrix = np.load(matrix_path, mmap_mode="r")
        summary["tokens_int_shape"] = tuple(int(x) for x in matrix.shape)
        summary["tokens_int_dtype"] = str(matrix.dtype)
        if matrix.shape != (len(ids), expected_tokens) or matrix.dtype != np.uint32:
            summary["bad_token_shapes"].append(
                {
                    "path": str(matrix_path),
                    "shape": tuple(int(x) for x in matrix.shape),
                    "dtype": str(matrix.dtype),
                    "expected_shape": (len(ids), expected_tokens),
                    "expected_dtype": "uint32",
                }
            )

    if tokens_dir.exists():
        summary["num_token_files"] = len(list(tokens_dir.glob("*.npy")))

    if args.check_per_volume:
        for volume_id in ids:
            row_path = token_path(token_dir, volume_id)
            if not row_path.exists():
                summary["missing_token_files"].append(str(row_path))
                continue
            arr = np.load(row_path, mmap_mode="r")
            if arr.shape != (expected_tokens,) or arr.dtype != np.uint32:
                summary["bad_token_shapes"].append(
                    {
                        "path": str(row_path),
                        "shape": tuple(int(x) for x in arr.shape),
                        "dtype": str(arr.dtype),
                        "expected_shape": (expected_tokens,),
                        "expected_dtype": "uint32",
                    }
                )

    ok = (
        bool(ids)
        and summary["tokens_dir_exists"]
        and summary["tokens_int_exists"]
        and summary["tokens_int_shape"] == (len(ids), expected_tokens)
        and summary["tokens_int_dtype"] == "uint32"
        and not summary["missing_token_files"]
        and not summary["bad_token_shapes"]
    )
    summary["ok"] = ok

    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
