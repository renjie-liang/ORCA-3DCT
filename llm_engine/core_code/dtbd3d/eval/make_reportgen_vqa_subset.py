#!/usr/bin/env python3
"""Build a small report-generation VQA JSON subset from a token artifact ids.txt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def volume_id_from_image(image: str) -> str:
    name = Path(image).name
    for suffix in [".nii.gz", ".nii", ".nii_embedded.npz", ".npz"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-json", required=True)
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--type-filter", default="report_generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vqa_json = Path(args.vqa_json)
    ids_file = Path(args.ids_file)
    out_path = Path(args.out)
    if args.n < 1:
        raise ValueError(f"--n must be >= 1, got {args.n}")
    if not vqa_json.exists():
        raise FileNotFoundError(vqa_json)
    if not ids_file.exists():
        raise FileNotFoundError(ids_file)

    wanted = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    wanted = wanted[: args.n]
    wanted_set = set(wanted)
    records = json.loads(vqa_json.read_text())
    subset = [
        record
        for record in records
        if record.get("id", "").startswith(args.type_filter)
        and volume_id_from_image(record["image"]) in wanted_set
    ]
    order = {volume_id: index for index, volume_id in enumerate(wanted)}
    subset.sort(key=lambda record: order[volume_id_from_image(record["image"])])
    if len(subset) != len(wanted):
        found = {volume_id_from_image(record["image"]) for record in subset}
        missing = [volume_id for volume_id in wanted if volume_id not in found]
        raise ValueError(
            f"requested {len(wanted)} IDs but found {len(subset)} matching "
            f"{args.type_filter!r} VQA records; first missing={missing[:10]}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, indent=2) + "\n")
    (out_path.with_suffix(".ids.txt")).write_text("\n".join(wanted) + "\n")
    print(f"wrote {len(subset)} report-generation records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
