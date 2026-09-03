#!/usr/bin/env python
"""Build final-stage base diagnostic supervision artifacts.

This script only exports stable raw supervision: labels and text. It does not
run any text encoder and does not create derived embeddings.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dtbd3d.diagnostic_token_learning.volume_id import normalize_volume_id


DEFAULT_DATA_ROOT = Path("./data/DTBD3D_data")


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = project_root()
    data_links = root / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(DEFAULT_DATA_ROOT / "diagnostic_supervision"))
    parser.add_argument("--cache-index-root", default=str(DEFAULT_DATA_ROOT / "ct_rate_cache" / "index"))
    parser.add_argument("--ctrate-labels-train", default=str(data_links / "ct_rate/dataset/multi_abnormality_labels/train_predicted_labels.csv"))
    parser.add_argument("--ctrate-labels-valid", default=str(data_links / "ct_rate/dataset/multi_abnormality_labels/valid_predicted_labels.csv"))
    parser.add_argument("--ctrate-reports-train", default=str(data_links / "ct_rate/dataset/radiology_text_reports/train_reports.csv"))
    parser.add_argument("--ctrate-reports-valid", default=str(data_links / "ct_rate/dataset/radiology_text_reports/validation_reports.csv"))
    parser.add_argument("--radgenome-files-dir", default=str(data_links / "radgenome/dataset/radgenome_files"))
    parser.add_argument("--disease-organ-yaml", default=str(root / "Experiment/configs/ctrate_disease_to_organ_groups.yaml"))
    parser.add_argument("--radgenome-catalog-yaml", default=str(root / "Experiment/configs/radgenome_mask_catalog_curated.yaml"))
    parser.add_argument("--radgenome-anatomy-yaml", default=str(root / "Experiment/configs/radgenome_anatomy_to_group.yaml"))
    parser.add_argument("--splits", default="train,valid")
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


def write_json_atomic(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_text_atomic(path, text)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, array)
    tmp_npy = tmp.with_suffix(tmp.suffix + ".npy")
    if tmp_npy.exists():
        tmp_npy.replace(path)
    else:
        tmp.replace(path)


def copy_yaml(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def split_path(paths: dict[str, Path], split: str) -> Path:
    if split not in paths:
        raise KeyError(split)
    return paths[split]


def load_ctrate_table(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(path)
    return {normalize_volume_id(row["VolumeName"]): row for row in df.to_dict(orient="records")}


def report_text(row: dict[str, Any]) -> str:
    parts = []
    for key in ("Findings_EN", "Impressions_EN"):
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() != "nan":
            parts.append(value)
    return " ".join(parts).strip()


def build_ctrate_disease_labels(
    out_root: Path,
    splits: list[str],
    ids_by_split: dict[str, list[str]],
    labels_csv_by_split: dict[str, Path],
    disease_organ_yaml: Path,
) -> dict[str, Any]:
    artifact_dir = out_root / "ctrate_disease_labels"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copy_yaml(disease_organ_yaml, artifact_dir / "disease_to_organ_groups.yaml")
    summary: dict[str, Any] = {"artifact": "ctrate_disease_labels", "splits": {}}
    label_names: list[str] | None = None
    for split in splits:
        df = pd.read_csv(labels_csv_by_split[split])
        names = [col for col in df.columns if col != "VolumeName"]
        if label_names is None:
            label_names = names
        elif label_names != names:
            raise ValueError(f"label columns differ for split={split}")
        table = {normalize_volume_id(row["VolumeName"]): row for row in df.to_dict(orient="records")}
        ids = ids_by_split[split]
        missing = [volume_id for volume_id in ids if volume_id not in table]
        if missing:
            raise KeyError(f"{split}: missing CT-RATE labels for {len(missing)} ids, first={missing[:5]}")
        labels = np.asarray([[int(table[volume_id][name]) for name in names] for volume_id in ids], dtype=np.int8)
        split_dir = artifact_dir / split
        write_text_atomic(split_dir / "ids.txt", "\n".join(ids) + "\n")
        save_npy_atomic(split_dir / "labels.npy", labels)
        summary["splits"][split] = {
            "ids": len(ids),
            "labels_shape": list(labels.shape),
            "positive_counts": {name: int(labels[:, index].sum()) for index, name in enumerate(names)},
        }
    write_json_atomic(artifact_dir / "label_names.json", label_names or [])
    write_json_atomic(
        artifact_dir / "metadata.json",
        {
            "artifact": "ctrate_disease_labels",
            "created_at": utc_now(),
            "source_csv": {split: str(labels_csv_by_split[split]) for split in splits},
            "label_names": label_names or [],
            "storage_contract": {"labels.npy": "[N,18] int8", "ids.txt": "N volume ids, line-aligned with labels.npy"},
            "summary": summary,
        },
    )
    return summary


def build_ctrate_reports(
    out_root: Path,
    splits: list[str],
    ids_by_split: dict[str, list[str]],
    reports_csv_by_split: dict[str, Path],
) -> dict[str, Any]:
    artifact_dir = out_root / "ctrate_reports"
    summary: dict[str, Any] = {"artifact": "ctrate_reports", "splits": {}}
    for split in splits:
        table = load_ctrate_table(reports_csv_by_split[split])
        ids = ids_by_split[split]
        rows: list[dict[str, Any]] = []
        missing = 0
        for volume_id in ids:
            row = table.get(volume_id)
            if row is None:
                missing += 1
                continue
            text = report_text(row)
            rows.append(
                {
                    "volume_id": volume_id,
                    "report_text": text,
                    "findings": str(row.get("Findings_EN", "") or "").strip(),
                    "impressions": str(row.get("Impressions_EN", "") or "").strip(),
                    "clinical_information": str(row.get("ClinicalInformation_EN", "") or "").strip(),
                    "technique": str(row.get("Technique_EN", "") or "").strip(),
                    "source": "ct_rate",
                }
            )
        split_dir = artifact_dir / split
        write_text_atomic(split_dir / "ids.txt", "\n".join(row["volume_id"] for row in rows) + ("\n" if rows else ""))
        write_jsonl_atomic(split_dir / "reports.jsonl", rows)
        summary["splits"][split] = {"requested_ids": len(ids), "rows": len(rows), "missing": missing}
    write_json_atomic(
        artifact_dir / "metadata.json",
        {
            "artifact": "ctrate_reports",
            "created_at": utc_now(),
            "source_csv": {split: str(reports_csv_by_split[split]) for split in splits},
            "text_policy": "report_text = Findings_EN + Impressions_EN",
            "summary": summary,
        },
    )
    return summary


def load_radgenome_label_maps(catalog_yaml: Path, anatomy_yaml: Path) -> tuple[dict[str, dict[str, str]], set[str], dict[str, str]]:
    catalog = yaml.safe_load(catalog_yaml.read_text())
    keep: dict[str, dict[str, str]] = {}
    for group in catalog.get("groups", []):
        group_name = str(group["name"])
        for label in group.get("mask_labels", []):
            label = str(label)
            keep[label] = {"mask_name": label, "organ_group": group_name}
    skip = {str(row["mask_label"]) for row in catalog.get("skip_labels", [])}
    anatomy_config = yaml.safe_load(anatomy_yaml.read_text())
    fallbacks = {str(key): str(value) for key, value in (anatomy_config.get("fallbacks", {}) or {}).items()}
    return keep, skip, fallbacks


def map_anatomy(
    anatomy: Any,
    keep: dict[str, dict[str, str]],
    skip: set[str],
    fallbacks: dict[str, str],
) -> dict[str, Any] | None:
    if anatomy is None or pd.isna(anatomy):
        return None
    anatomy_text = str(anatomy).strip()
    if not anatomy_text:
        return None
    parts = [part.strip() for part in anatomy_text.split("/") if part.strip()]
    for part in reversed(parts):
        if part in keep:
            mapped = keep[part]
            return {"anatomy": anatomy_text, "mask_name": mapped["mask_name"], "organ_group": mapped["organ_group"], "mapping_status": "exact"}
    if any(part in skip for part in parts):
        return {"anatomy": anatomy_text, "mask_name": "", "organ_group": "", "mapping_status": "skip_label"}
    for part in reversed(parts):
        if part in fallbacks:
            return {"anatomy": anatomy_text, "mask_name": part, "organ_group": fallbacks[part], "mapping_status": "fallback"}
    return {"anatomy": anatomy_text, "mask_name": "", "organ_group": "", "mapping_status": "unmapped"}


def load_radgenome_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Volumename" not in df.columns:
        raise ValueError(f"RadGenome CSV missing Volumename: {path}")
    return df


def radgenome_csv_paths(files_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "train": files_dir / f"train_{stem}.csv",
        "valid": files_dir / f"validation_{stem}.csv",
    }


def write_radgenome_case_disorders(out_root: Path, splits: list[str], files_dir: Path) -> dict[str, Any]:
    artifact_dir = out_root / "radgenome_case_disorders"
    paths = radgenome_csv_paths(files_dir, "case_disorders")
    summary: dict[str, Any] = {"artifact": "radgenome_case_disorders", "splits": {}}
    for split in splits:
        rows: list[dict[str, Any]] = []
        df = load_radgenome_csv(split_path(paths, split))
        for row in df.to_dict(orient="records"):
            volume_id = normalize_volume_id(row["Volumename"])
            text = str(row.get("Disorders", "") or "").strip()
            rows.append({"volume_id": volume_id, "disorders_text": text, "source": "radgenome_case_disorders"})
        split_dir = artifact_dir / split
        write_text_atomic(split_dir / "ids.txt", "\n".join(row["volume_id"] for row in rows) + ("\n" if rows else ""))
        write_jsonl_atomic(split_dir / "disorders.jsonl", rows)
        summary["splits"][split] = {"rows": len(rows), "unique_volumes": len({row["volume_id"] for row in rows})}
    write_json_atomic(artifact_dir / "metadata.json", {"artifact": "radgenome_case_disorders", "created_at": utc_now(), "source_csv": {s: str(paths[s]) for s in splits}, "summary": summary})
    return summary


def write_radgenome_region_artifact(
    *,
    out_root: Path,
    splits: list[str],
    files_dir: Path,
    artifact_name: str,
    csv_stem: str,
    output_filename: str,
    text_column: str,
    keep: dict[str, dict[str, str]],
    skip: set[str],
    fallbacks: dict[str, str],
) -> dict[str, Any]:
    artifact_dir = out_root / artifact_name
    paths = radgenome_csv_paths(files_dir, csv_stem)
    summary: dict[str, Any] = {"artifact": artifact_name, "splits": {}}
    for split in splits:
        df = load_radgenome_csv(split_path(paths, split))
        rows: list[dict[str, Any]] = []
        status_counts: Counter[str] = Counter()
        for row in df.to_dict(orient="records"):
            mapped = map_anatomy(row.get("Anatomy"), keep, skip, fallbacks)
            status = "global" if mapped is None else str(mapped["mapping_status"])
            status_counts[status] += 1
            if mapped is None or status in {"skip_label", "unmapped"}:
                continue
            volume_id = normalize_volume_id(row["Volumename"])
            text = str(row.get(text_column, "") or "").strip()
            if not text:
                continue
            out = {
                "volume_id": volume_id,
                "anatomy": mapped["anatomy"],
                "mask_name": mapped["mask_name"],
                "organ_group": mapped["organ_group"],
                "mapping_status": mapped["mapping_status"],
                "text": text,
                "source": f"radgenome_{csv_stem}",
            }
            if "Presence" in row:
                presence_text = str(row.get("Presence", "") or "").strip().lower()
                out["presence"] = presence_text.startswith("yes")
                out["presence_raw"] = str(row.get("Presence", "") or "").strip()
                out["finding"] = text
            if text_column == "Sentence":
                out["sentence"] = text
            if text_column == "Abnormality":
                out["abnormality"] = text
            rows.append(out)
        split_dir = artifact_dir / split
        write_text_atomic(split_dir / "ids.txt", "\n".join(sorted({row["volume_id"] for row in rows})) + ("\n" if rows else ""))
        write_jsonl_atomic(split_dir / output_filename, rows)
        summary["splits"][split] = {
            "source_rows": int(len(df)),
            "rows": len(rows),
            "unique_volumes": len({row["volume_id"] for row in rows}),
            "mapping_status_counts": dict(sorted(status_counts.items())),
            "organ_group_counts": dict(sorted(Counter(row["organ_group"] for row in rows).items())),
        }
    write_json_atomic(
        artifact_dir / "metadata.json",
        {
            "artifact": artifact_name,
            "created_at": utc_now(),
            "source_csv": {s: str(paths[s]) for s in splits},
            "mapping_policy": "slash-separated Anatomy mapped via curated RadGenome mask catalog; global/skip/unmapped rows omitted",
            "summary": summary,
        },
    )
    return summary


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if any(split not in {"train", "valid"} for split in splits):
        raise ValueError(f"--splits must contain train/valid, got {splits}")

    ids_by_split = {
        "train": read_ids(Path(args.cache_index_root) / "train_ids.txt"),
        "valid": read_ids(Path(args.cache_index_root) / "valid_ids.txt"),
    }
    labels_csv_by_split = {"train": Path(args.ctrate_labels_train), "valid": Path(args.ctrate_labels_valid)}
    reports_csv_by_split = {"train": Path(args.ctrate_reports_train), "valid": Path(args.ctrate_reports_valid)}
    keep, skip, fallbacks = load_radgenome_label_maps(Path(args.radgenome_catalog_yaml), Path(args.radgenome_anatomy_yaml))
    files_dir = Path(args.radgenome_files_dir)

    summary = {
        "created_at": utc_now(),
        "out_root": str(out_root),
        "splits": splits,
        "artifacts": {},
    }
    summary["artifacts"]["ctrate_disease_labels"] = build_ctrate_disease_labels(out_root, splits, ids_by_split, labels_csv_by_split, Path(args.disease_organ_yaml))
    summary["artifacts"]["ctrate_reports"] = build_ctrate_reports(out_root, splits, ids_by_split, reports_csv_by_split)
    summary["artifacts"]["radgenome_case_disorders"] = write_radgenome_case_disorders(out_root, splits, files_dir)
    summary["artifacts"]["radgenome_region_abnormality"] = write_radgenome_region_artifact(
        out_root=out_root,
        splits=splits,
        files_dir=files_dir,
        artifact_name="radgenome_region_abnormality",
        csv_stem="vqa_abnormality",
        output_filename="region_abnormality.jsonl",
        text_column="Abnormality",
        keep=keep,
        skip=skip,
        fallbacks=fallbacks,
    )
    summary["artifacts"]["radgenome_region_presence"] = write_radgenome_region_artifact(
        out_root=out_root,
        splits=splits,
        files_dir=files_dir,
        artifact_name="radgenome_region_presence",
        csv_stem="vqa_presence",
        output_filename="region_presence.jsonl",
        text_column="Finding",
        keep=keep,
        skip=skip,
        fallbacks=fallbacks,
    )
    summary["artifacts"]["radgenome_region_reports"] = write_radgenome_region_artifact(
        out_root=out_root,
        splits=splits,
        files_dir=files_dir,
        artifact_name="radgenome_region_reports",
        csv_stem="region_report",
        output_filename="region_reports.jsonl",
        text_column="Sentence",
        keep=keep,
        skip=skip,
        fallbacks=fallbacks,
    )
    write_json_atomic(out_root / "metadata.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
