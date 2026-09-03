#!/usr/bin/env python
"""Summarize completed TS mask artifact shards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="./data/organ_masks/btb3d")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--expected-num-shards", type=int, default=16)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_stats_rows(stats_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(stats_dir.glob("shard_*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row, stats_csv=str(path)) for row in reader)
    return rows


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    index_dir = artifact_dir / "index"
    stats_dir = index_dir / "shard_stats"
    rows = read_stats_rows(stats_dir)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    summary: dict[str, object] = {
        "artifact": "ts_mask_8x8x8",
        "created_at": utc_now(),
        "artifact_dir": str(artifact_dir),
        "expected_num_shards": args.expected_num_shards,
        "stats_csv_count": len(list(stats_dir.glob("shard_*.csv"))),
        "status_counts": dict(sorted(Counter(row.get("status", "") for row in rows).items())),
        "splits": {},
    }
    for split in splits:
        split_dir = artifact_dir / split
        files = sorted(split_dir.glob("*.npz"))
        ids = [path.stem for path in files]
        rows_for_split = [row for row in rows if row.get("split") == split]
        n_masks = [
            int(row["n_masks"])
            for row in rows_for_split
            if row.get("status") == "ok" and str(row.get("n_masks", "")).strip()
        ]
        summary["splits"][split] = {
            "npz_count": len(files),
            "stats_rows": len(rows_for_split),
            "status_counts": dict(sorted(Counter(row.get("status", "") for row in rows_for_split).items())),
            "ids_file": str(index_dir / f"{split}_ids.txt"),
            "n_masks_min": min(n_masks) if n_masks else None,
            "n_masks_median": sorted(n_masks)[len(n_masks) // 2] if n_masks else None,
            "n_masks_max": max(n_masks) if n_masks else None,
        }
        write_text_atomic(index_dir / f"{split}_ids.txt", "\n".join(ids) + ("\n" if ids else ""))
    write_text_atomic(index_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
