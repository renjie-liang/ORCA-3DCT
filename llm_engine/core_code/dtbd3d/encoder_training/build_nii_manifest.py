#!/usr/bin/env python
"""Build CT-RATE NIfTI path manifests from BTB3D ids.txt.

This avoids a slow recursive `find` over CT-RATE directories. By default it
does not call `Path.exists()` for every volume, because that metadata walk is
slow on shared filesystems. Use `--check-exists` when you explicitly want a
validated manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True, help="BTB3D ids.txt path.")
    parser.add_argument("--fixed-root", required=True, help="CT-RATE {train,valid}_fixed root.")
    parser.add_argument("--out", required=True, help="Output manifest path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-exists", action="store_true")
    return parser.parse_args()


def path_from_volume_id(fixed_root: Path, volume_id: str) -> Path:
    parts = volume_id.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected CT-RATE volume id: {volume_id}")
    patient = "_".join(parts[:2])
    accession = "_".join(parts[:3])
    return fixed_root / patient / accession / f"{volume_id}.nii.gz"


def iter_progress(items: list[str], enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(items, desc="build manifest", unit="vol")
    return items


def main() -> int:
    args = parse_args()
    ids_path = Path(args.ids)
    fixed_root = Path(args.fixed_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        ids = ids[: args.limit]

    paths: list[str] = []
    missing: list[str] = []
    for volume_id in iter_progress(ids, True):
        path = path_from_volume_id(fixed_root, volume_id)
        if args.check_exists and not path.exists():
            missing.append(str(path))
        paths.append(str(path))

    out_path.write_text("\n".join(paths) + "\n")
    print(f"wrote={out_path}")
    print(f"n_paths={len(paths)}")
    print(f"check_exists={args.check_exists}")
    print(f"missing={len(missing)}")
    if missing:
        missing_path = out_path.with_suffix(out_path.suffix + ".missing")
        missing_path.write_text("\n".join(missing) + "\n")
        print(f"missing_file={missing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
