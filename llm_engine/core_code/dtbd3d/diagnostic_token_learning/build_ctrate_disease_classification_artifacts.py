#!/usr/bin/env python
"""Build organ-conditioned CT-RATE disease classification artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_DATA_ROOT = Path("./data/DTBD3D_data")
GROUP_TO_MASK_NAME = {"pleura": "pleura_proxy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision-root", default=str(DEFAULT_DATA_ROOT / "diagnostic_supervision"))
    parser.add_argument("--out-dir", default=str(DEFAULT_DATA_ROOT / "diagnostic_supervision" / "ctrate_organ_disease_classification"))
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--include-global", action="store_true", help="Keep global labels that do not have an organ mask.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, array)
    tmp_npy = tmp.with_suffix(tmp.suffix + ".npy")
    if tmp_npy.exists():
        tmp_npy.replace(path)
    else:
        tmp.replace(path)


def load_group_mapping(path: Path, include_global: bool) -> tuple[list[str], dict[str, list[str]]]:
    raw = yaml.safe_load(path.read_text())
    groups_raw = raw.get("groups", {}) or {}
    groups: list[str] = []
    group_to_diseases: dict[str, list[str]] = {}
    for group, diseases in groups_raw.items():
        group = str(group)
        if group == "global" and not include_global:
            continue
        groups.append(group)
        group_to_diseases[group] = [str(disease) for disease in diseases or []]
    return groups, group_to_diseases


def build_split(
    *,
    src_dir: Path,
    out_dir: Path,
    split: str,
    organ_groups: list[str],
    disease_names: list[str],
    group_to_diseases: dict[str, list[str]],
) -> dict[str, Any]:
    ids = read_ids(src_dir / split / "ids.txt")
    source_labels = np.load(src_dir / split / "labels.npy", mmap_mode="r")
    if source_labels.shape != (len(ids), len(disease_names)):
        raise ValueError(
            f"{split}: source labels shape={source_labels.shape}, expected {(len(ids), len(disease_names))}"
        )

    disease_to_index = {name: idx for idx, name in enumerate(disease_names)}
    valid_mask = np.zeros((len(organ_groups), len(disease_names)), dtype=bool)
    for group_index, group in enumerate(organ_groups):
        for disease in group_to_diseases[group]:
            if disease not in disease_to_index:
                raise KeyError(f"{split}: disease {disease!r} from mapping is not in label_names.json")
            valid_mask[group_index, disease_to_index[disease]] = True

    labels = np.zeros((len(ids), len(organ_groups), len(disease_names)), dtype=np.int8)
    for group_index in range(len(organ_groups)):
        labels[:, group_index, :] = np.asarray(source_labels, dtype=np.int8) * valid_mask[group_index].astype(np.int8)

    split_dir = out_dir / split
    write_text_atomic(split_dir / "ids.txt", "\n".join(ids) + "\n")
    save_npy_atomic(split_dir / "labels.npy", labels)
    save_npy_atomic(split_dir / "valid_mask.npy", valid_mask)

    positives_by_group = {
        group: {
            disease: int(labels[:, group_index, disease_to_index[disease]].sum())
            for disease in group_to_diseases[group]
        }
        for group_index, group in enumerate(organ_groups)
    }
    return {
        "ids": len(ids),
        "labels_shape": list(labels.shape),
        "valid_mask": valid_mask.astype(bool).tolist(),
        "valid_pairs": int(valid_mask.sum()),
        "positives_by_group": positives_by_group,
    }


def main() -> int:
    args = parse_args()
    src_dir = Path(args.supervision_root) / "ctrate_disease_labels"
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists; pass --overwrite to rebuild")

    disease_names = json.loads((src_dir / "label_names.json").read_text())
    organ_groups, group_to_diseases = load_group_mapping(src_dir / "disease_to_organ_groups.yaml", args.include_global)
    group_to_mask_name = {group: GROUP_TO_MASK_NAME.get(group, group) for group in organ_groups}

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    summary = {
        "artifact": "ctrate_organ_disease_classification",
        "created_at": utc_now(),
        "source_artifact": str(src_dir),
        "include_global": bool(args.include_global),
        "organ_groups": organ_groups,
        "disease_names": disease_names,
        "group_to_mask_name": group_to_mask_name,
        "storage_contract": {
            "<split>/labels.npy": "[N,G,L] int8; invalid organ-disease pairs are zero and masked by valid_mask.npy",
            "<split>/valid_mask.npy": "[G,L] bool; true means disease label is valid for organ group",
            "<split>/ids.txt": "N volume ids, line-aligned with labels.npy",
        },
        "splits": {},
    }
    for split in splits:
        summary["splits"][split] = build_split(
            src_dir=src_dir,
            out_dir=out_dir,
            split=split,
            organ_groups=organ_groups,
            disease_names=disease_names,
            group_to_diseases=group_to_diseases,
        )

    write_json_atomic(out_dir / "organ_groups.json", organ_groups)
    write_json_atomic(out_dir / "disease_names.json", disease_names)
    write_json_atomic(out_dir / "group_to_mask_name.json", group_to_mask_name)
    write_json_atomic(out_dir / "metadata.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
