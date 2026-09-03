"""Summarize normal/abnormal text-pair distribution and sampling effects."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dtbd3d.diagnostic_token_learning.text_pair_sampling import (
    is_normal_no_finding_pair,
    select_abnormal_first_pairs,
)
from dtbd3d.diagnostic_token_learning.volume_id import normalize_volume_id


@dataclass(frozen=True)
class Pair:
    embedding_row: int
    organ_group: str
    mask_name: str
    text_hash: str
    text: str
    row: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def pct(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def summarize_split(artifact_dir: Path, split: str, max_pairs_per_volume: int) -> dict[str, Any]:
    index_path = artifact_dir / split / "index.jsonl"
    rows = read_jsonl(index_path)
    pairs_by_volume: dict[str, list[Pair]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        volume_id = normalize_volume_id(str(row.get("volume_id", "")))
        if not volume_id:
            raise ValueError(f"missing volume_id at {index_path}:{row_idx + 1}")
        pairs_by_volume[volume_id].append(
            Pair(
                embedding_row=int(row.get("embedding_row", row.get("row_id", row_idx))),
                organ_group=str(row.get("organ_group", "")),
                mask_name=str(row.get("mask_name", "")),
                text_hash=str(row.get("text_hash", "")),
                text=str(row.get("text", row.get("target_text", row.get("sentence", row.get("prompt", ""))))),
                row=row,
            )
        )

    per_volume_total: list[int] = []
    per_volume_normal: list[int] = []
    per_volume_abnormal: list[int] = []
    selected_total: list[int] = []
    selected_normal: list[int] = []
    selected_abnormal: list[int] = []
    organ_abnormal = Counter()
    organ_normal = Counter()

    for volume_id, pairs in pairs_by_volume.items():
        normal_count = sum(1 for pair in pairs if is_normal_no_finding_pair(pair))
        abnormal_count = len(pairs) - normal_count
        selected = select_abnormal_first_pairs(
            pairs,
            max_pairs=max_pairs_per_volume,
            volume_id=volume_id,
            source_name=artifact_dir.name,
        )
        selected_normal_count = sum(1 for pair in selected if is_normal_no_finding_pair(pair))
        selected_abnormal_count = len(selected) - selected_normal_count

        per_volume_total.append(len(pairs))
        per_volume_normal.append(normal_count)
        per_volume_abnormal.append(abnormal_count)
        selected_total.append(len(selected))
        selected_normal.append(selected_normal_count)
        selected_abnormal.append(selected_abnormal_count)
        for pair in pairs:
            if is_normal_no_finding_pair(pair):
                organ_normal[pair.organ_group] += 1
            else:
                organ_abnormal[pair.organ_group] += 1

    return {
        "artifact_dir": str(artifact_dir),
        "split": split,
        "max_pairs_per_volume": int(max_pairs_per_volume),
        "rows": len(rows),
        "volumes": len(pairs_by_volume),
        "all_pairs": {
            "normal_no_finding": int(sum(per_volume_normal)),
            "abnormal": int(sum(per_volume_abnormal)),
            "normal_fraction": float(sum(per_volume_normal) / max(sum(per_volume_total), 1)),
            "abnormal_fraction": float(sum(per_volume_abnormal) / max(sum(per_volume_total), 1)),
        },
        "selected_pairs": {
            "total": int(sum(selected_total)),
            "normal_no_finding": int(sum(selected_normal)),
            "abnormal": int(sum(selected_abnormal)),
            "normal_fraction": float(sum(selected_normal) / max(sum(selected_total), 1)),
            "abnormal_fraction": float(sum(selected_abnormal) / max(sum(selected_total), 1)),
        },
        "per_volume": {
            "total_mean": float(np.mean(per_volume_total)) if per_volume_total else 0.0,
            "normal_mean": float(np.mean(per_volume_normal)) if per_volume_normal else 0.0,
            "abnormal_mean": float(np.mean(per_volume_abnormal)) if per_volume_abnormal else 0.0,
            "abnormal_p50": pct(per_volume_abnormal, 50),
            "abnormal_p75": pct(per_volume_abnormal, 75),
            "abnormal_p90": pct(per_volume_abnormal, 90),
            "abnormal_p95": pct(per_volume_abnormal, 95),
            "abnormal_max": int(max(per_volume_abnormal) if per_volume_abnormal else 0),
            "volumes_with_abnormal": int(sum(1 for value in per_volume_abnormal if value > 0)),
            "volumes_without_abnormal": int(sum(1 for value in per_volume_abnormal if value == 0)),
            "selected_total_mean": float(np.mean(selected_total)) if selected_total else 0.0,
            "selected_abnormal_mean": float(np.mean(selected_abnormal)) if selected_abnormal else 0.0,
        },
        "top_organs": {
            "abnormal": organ_abnormal.most_common(20),
            "normal_no_finding": organ_normal.most_common(20),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Source artifact dir containing split/index.jsonl.")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--max-pairs-per-volume", type=int, default=5)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    summaries = [
        summarize_split(artifact_dir, split.strip(), args.max_pairs_per_volume)
        for split in args.splits.split(",")
        if split.strip()
    ]
    payload = {"summaries": summaries}
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
