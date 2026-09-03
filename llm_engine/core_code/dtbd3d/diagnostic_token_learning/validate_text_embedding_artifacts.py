#!/usr/bin/env python
"""Validate diagnostic text embedding artifacts before CT-token training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dtbd3d.diagnostic_token_learning.volume_id import normalize_volume_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--sources", default="radgenome_region_abnormality,ctrate_organ_disease_text")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--mask-artifact-dirs", default="", help="Optional source=dir pairs separated by comma.")
    parser.add_argument("--sample-missing", type=int, default=10)
    parser.add_argument("--skip-mask-check", action="store_true", help="Only validate text embedding files.")
    parser.add_argument("--max-mask-volumes-per-split", type=int, default=0, help="Limit mask checks per source/split; 0 means all.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print mask-check progress every N volumes; 0 disables.")
    return parser.parse_args()


def parse_mask_dirs(raw: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"mask artifact dir entry must be source=path, got {item!r}")
        source, path = item.split("=", 1)
        out[source.strip()] = Path(path.strip())
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def validate_split(
    source: str,
    split: str,
    artifact_root: Path,
    mask_dirs: dict[str, Path],
    sample_missing: int,
    skip_mask_check: bool,
    max_mask_volumes: int,
    progress_every: int,
) -> dict[str, Any]:
    split_dir = artifact_root / source / split
    index_path = split_dir / "index.jsonl"
    embeddings_path = split_dir / "embeddings.npy"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    _log(f"[validate] {source}/{split}: loading index and embeddings")
    rows = read_jsonl(index_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"{embeddings_path} must be 2D, got shape={embeddings.shape}")
    if embeddings.shape[0] != len(rows):
        raise ValueError(f"{source}/{split}: index rows={len(rows)} but embeddings rows={embeddings.shape[0]}")
    encoded_mask_path = split_dir / "encoded_mask.npy"
    if encoded_mask_path.exists():
        encoded_mask = np.asarray(np.load(encoded_mask_path, mmap_mode="r"), dtype=bool)
        if encoded_mask.shape != (len(rows),):
            raise ValueError(f"{encoded_mask_path} shape={encoded_mask.shape}, expected {(len(rows),)}")
    else:
        encoded_mask = np.ones((len(rows),), dtype=bool)
    encoded_rows = [row for row, is_encoded in zip(rows, encoded_mask, strict=True) if bool(is_encoded)]
    _log(
        f"[validate] {source}/{split}: rows={len(rows)} encoded={len(encoded_rows)} "
        f"embedding_shape={tuple(embeddings.shape)}"
    )

    missing_masks: list[dict[str, Any]] = []
    mask_dir = None if skip_mask_check else mask_dirs.get(source)
    mask_check_total_volumes = 0
    mask_check_checked_volumes = 0
    requested_unique_pairs = 0
    matched_unique_pairs = 0
    missing_unique_pairs = 0
    requested_text_rows = 0
    matched_text_rows = 0
    missing_text_rows = 0
    requested_unique_by_group: Counter[str] = Counter()
    matched_unique_by_group: Counter[str] = Counter()
    missing_unique_by_group: Counter[str] = Counter()
    requested_rows_by_group: Counter[str] = Counter()
    matched_rows_by_group: Counter[str] = Counter()
    missing_rows_by_group: Counter[str] = Counter()
    if mask_dir is not None:
        by_volume: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        for row in encoded_rows:
            by_volume[normalize_volume_id(str(row["volume_id"]))][
                (str(row["mask_name"]), str(row.get("organ_group", "")))
            ] += 1
        items = list(by_volume.items())
        mask_check_total_volumes = len(items)
        if max_mask_volumes > 0:
            items = items[:max_mask_volumes]
        _log(
            f"[validate] {source}/{split}: checking masks for {len(items)}/{mask_check_total_volumes} volumes "
            f"in {mask_dir / split}"
        )
        start_time = time.time()
        def log_progress(idx: int) -> None:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            unique_coverage = matched_unique_pairs / max(requested_unique_pairs, 1)
            row_coverage = matched_text_rows / max(requested_text_rows, 1)
            _log(
                f"[validate] {source}/{split}: volume {idx}/{len(items)} rate={rate:.1f} vol/s "
                f"unique_pairs={matched_unique_pairs}/{requested_unique_pairs} "
                f"missing_unique={missing_unique_pairs} cov={unique_coverage:.2%} "
                f"text_rows={matched_text_rows}/{requested_text_rows} "
                f"missing_rows={missing_text_rows} cov={row_coverage:.2%}"
            )

        for idx, (volume_id, requested_masks) in enumerate(items, start=1):
            mask_check_checked_volumes = idx
            requested_unique_pairs += len(requested_masks)
            requested_text_rows += sum(requested_masks.values())
            for (_, organ_group), count in requested_masks.items():
                requested_unique_by_group[organ_group] += 1
                requested_rows_by_group[organ_group] += count
            mask_path = mask_dir / split / f"{volume_id}.npz"
            if not mask_path.exists():
                for (mask_name, organ_group), count in sorted(requested_masks.items()):
                    missing_unique_pairs += 1
                    missing_text_rows += count
                    missing_unique_by_group[organ_group] += 1
                    missing_rows_by_group[organ_group] += count
                    missing_masks.append(
                        {
                            "volume_id": volume_id,
                            "mask_name": mask_name,
                            "organ_group": organ_group,
                            "text_row_count": count,
                            "reason": "missing_npz",
                        }
                    )
                if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(items)):
                    log_progress(idx)
                continue
            npz = np.load(mask_path, allow_pickle=False)
            names = {str(name) for name in npz["mask_names"]}
            groups = {str(group) for group in npz["mask_groups"]} if "mask_groups" in npz.files else set(names)
            for (mask_name, organ_group), count in sorted(requested_masks.items()):
                if mask_name in names or organ_group in groups:
                    matched_unique_pairs += 1
                    matched_text_rows += count
                    matched_unique_by_group[organ_group] += 1
                    matched_rows_by_group[organ_group] += count
                    continue
                missing_unique_pairs += 1
                missing_text_rows += count
                missing_unique_by_group[organ_group] += 1
                missing_rows_by_group[organ_group] += count
                missing_masks.append(
                    {
                        "volume_id": volume_id,
                        "mask_name": mask_name,
                        "organ_group": organ_group,
                        "text_row_count": count,
                        "reason": "missing_mask_name_or_group",
                    }
                )
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(items)):
                log_progress(idx)

    encoded_indices = np.flatnonzero(encoded_mask)
    norm_indices = encoded_indices[: min(len(encoded_indices), 2048)]
    norms = np.linalg.norm(np.asarray(embeddings[norm_indices], dtype=np.float32), axis=1) if len(norm_indices) else np.asarray([], dtype=np.float32)
    return {
        "source": source,
        "split": split,
        "rows": len(rows),
        "encoded_rows": len(encoded_rows),
        "complete": bool(encoded_mask.all()) if len(encoded_mask) else True,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "unique_volumes": len({normalize_volume_id(str(row["volume_id"])) for row in rows}),
        "organ_group_counts": dict(sorted(Counter(str(row.get("organ_group", "")) for row in rows).items())),
        "mask_check": None
        if mask_dir is None
        else {
            "mask_artifact_dir": str(mask_dir),
            "checked_volumes": mask_check_checked_volumes,
            "candidate_volumes": mask_check_total_volumes,
            "is_full_check": max_mask_volumes <= 0 or mask_check_checked_volumes == mask_check_total_volumes,
            "unique_region_mask_pairs": {
                "requested": requested_unique_pairs,
                "matched": matched_unique_pairs,
                "missing": missing_unique_pairs,
                "coverage": matched_unique_pairs / max(requested_unique_pairs, 1),
            },
            "text_rows": {
                "requested": requested_text_rows,
                "matched": matched_text_rows,
                "missing": missing_text_rows,
                "coverage": matched_text_rows / max(requested_text_rows, 1),
            },
            "by_organ_group": {
                group: {
                    "unique_requested": requested_unique_by_group[group],
                    "unique_matched": matched_unique_by_group[group],
                    "unique_missing": missing_unique_by_group[group],
                    "unique_coverage": matched_unique_by_group[group] / max(requested_unique_by_group[group], 1),
                    "text_rows_requested": requested_rows_by_group[group],
                    "text_rows_matched": matched_rows_by_group[group],
                    "text_rows_missing": missing_rows_by_group[group],
                    "text_rows_coverage": matched_rows_by_group[group] / max(requested_rows_by_group[group], 1),
                }
                for group in sorted(requested_unique_by_group)
            },
            "missing_examples": missing_masks[:sample_missing],
        },
        "sample_embedding_norm": {
            "mean": float(norms.mean()) if norms.size else 0.0,
            "min": float(norms.min()) if norms.size else 0.0,
            "max": float(norms.max()) if norms.size else 0.0,
        },
    }


def main() -> int:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    mask_dirs = parse_mask_dirs(args.mask_artifact_dirs)
    summary = {"artifact_root": str(artifact_root), "sources": sources, "splits": splits, "results": {}}
    for source in sources:
        summary["results"][source] = {}
        for split in splits:
            summary["results"][source][split] = validate_split(
                source,
                split,
                artifact_root,
                mask_dirs,
                args.sample_missing,
                args.skip_mask_check,
                args.max_mask_volumes_per_split,
                args.progress_every,
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
