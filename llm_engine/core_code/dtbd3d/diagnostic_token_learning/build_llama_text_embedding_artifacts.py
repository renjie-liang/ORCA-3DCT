#!/usr/bin/env python
"""Build Llama-space text embedding artifacts for diagnostic token learning.

The base supervision artifacts store raw labels and text. This script derives
stable text targets and, unless --index-only is used, encodes them with a frozen
Llama-family text model. Encoder training then reads the saved embeddings
instead of loading Llama inside the CT-token training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from dtbd3d.diagnostic_token_learning.volume_id import normalize_volume_id


DEFAULT_DATA_ROOT = Path("./data/DTBD3D_data")
DEFAULT_LLAMA = Path("./checkpoints/Llama-3.1-8B-Instruct")
SOURCE_TO_FILE = {
    "radgenome_region_abnormality": "region_abnormality.jsonl",
    "ctrate_organ_disease_text": "",
}
DISEASE_GROUP_TO_MASK_NAME = {
    "pleura": "pleura_proxy",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision-root", default=str(DEFAULT_DATA_ROOT / "diagnostic_supervision"))
    parser.add_argument("--out-root", default=str(DEFAULT_DATA_ROOT / "diagnostic_text_embeddings" / "llama_hidden"))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_LLAMA))
    parser.add_argument("--embedding-backend", default="llama_hidden")
    parser.add_argument("--sources", default="radgenome_region_abnormality,ctrate_organ_disease_text")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--ids-file", default="", help="Optional volume-id allowlist for smoke artifacts.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--pooling", choices=["mean", "last_nonpad"], default="mean")
    parser.add_argument("--index-only", action="store_true", help="Write index.jsonl/metadata only; do not load Llama or write embeddings.npy.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a source/split when its requested output already exists.")
    parser.add_argument("--resume", action="store_true", help="Resume a partial embeddings.npy using encoded_mask.npy.")
    parser.add_argument("--encode-limit", type=int, default=0, help="Encode at most this many missing rows per source/split; 0 means all missing rows.")
    parser.add_argument("--include-global-disease", action="store_true", help="Include CT-RATE labels mapped to the global pseudo-region.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_json_atomic(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text.rstrip(" .")


def text_hash(text: str) -> str:
    canonical = normalize_text(text).lower()
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def make_target_text(organ_group: str, finding: str) -> str:
    organ = normalize_text(organ_group).replace("_", " ")
    finding_text = normalize_text(finding)
    return f"Region: {organ}. Finding: {finding_text}."


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_disease_to_groups(path: Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(path.read_text())
    out: dict[str, list[str]] = {}
    for group, labels in (raw.get("groups", {}) or {}).items():
        for label in labels or []:
            out.setdefault(str(label), []).append(str(group))
    return out


def dedupe_rows(rows: Iterable[dict[str, Any]], max_rows: int = 0) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str, str]] = []
    for source_row_id, row in enumerate(rows):
        key = (
            str(row["volume_id"]),
            str(row["organ_group"]),
            str(row["mask_name"]),
            str(row["target_text"]).lower(),
        )
        if key not in keyed:
            out = dict(row)
            out["source_row_id"] = source_row_id
            out["duplicate_count"] = 1
            keyed[key] = out
            order.append(key)
            if max_rows > 0 and len(order) >= max_rows:
                break
        else:
            keyed[key]["duplicate_count"] += 1
    final_rows = []
    for row_id, key in enumerate(order):
        row = keyed[key]
        row["row_id"] = row_id
        row["embedding_row"] = row_id
        row["pair_key"] = "|".join(key)
        final_rows.append(row)
    return final_rows


def build_radgenome_region_abnormality_rows(supervision_root: Path, split: str) -> list[dict[str, Any]]:
    path = supervision_root / "radgenome_region_abnormality" / split / "region_abnormality.jsonl"
    rows = []
    for row in read_jsonl(path):
        volume_id = normalize_volume_id(str(row["volume_id"]))
        organ_group = str(row["organ_group"])
        mask_name = str(row["mask_name"])
        finding = normalize_text(str(row.get("text", "")))
        if not volume_id or not organ_group or not mask_name or not finding:
            continue
        target_text = make_target_text(organ_group, finding)
        rows.append(
            {
                "volume_id": volume_id,
                "source": "radgenome_region_abnormality",
                "organ_group": organ_group,
                "mask_name": mask_name,
                "anatomy": row.get("anatomy", ""),
                "mapping_status": row.get("mapping_status", ""),
                "raw_text": finding,
                "target_text": target_text,
                "text": target_text,
                "text_hash": text_hash(target_text),
            }
        )
    return rows


def build_ctrate_organ_disease_rows(
    supervision_root: Path,
    split: str,
    *,
    include_global_disease: bool,
) -> list[dict[str, Any]]:
    artifact_dir = supervision_root / "ctrate_disease_labels"
    ids = read_ids(artifact_dir / split / "ids.txt")
    label_names = json.loads((artifact_dir / "label_names.json").read_text())
    labels = np.load(artifact_dir / split / "labels.npy", mmap_mode="r")
    label_to_groups = load_disease_to_groups(artifact_dir / "disease_to_organ_groups.yaml")
    if labels.shape != (len(ids), len(label_names)):
        raise ValueError(f"labels shape mismatch for split={split}: {labels.shape}, ids={len(ids)}, labels={len(label_names)}")

    rows: list[dict[str, Any]] = []
    for volume_index, volume_id in enumerate(ids):
        normalized_id = normalize_volume_id(volume_id)
        positive_indices = np.flatnonzero(labels[volume_index] > 0)
        for label_index in positive_indices:
            label_name = str(label_names[int(label_index)])
            for organ_group in label_to_groups.get(label_name, []):
                if organ_group == "global" and not include_global_disease:
                    continue
                mask_name = DISEASE_GROUP_TO_MASK_NAME.get(organ_group, organ_group)
                target_text = make_target_text(organ_group, label_name)
                rows.append(
                    {
                        "volume_id": normalized_id,
                        "source": "ctrate_organ_disease_text",
                        "organ_group": organ_group,
                        "mask_name": mask_name,
                        "label_name": label_name,
                        "raw_text": label_name,
                        "target_text": target_text,
                        "text": target_text,
                        "text_hash": text_hash(target_text),
                    }
                )
    return rows


def build_rows_for_source(
    source: str,
    supervision_root: Path,
    split: str,
    *,
    max_rows: int,
    include_global_disease: bool,
    allowed_ids: set[str] | None,
) -> list[dict[str, Any]]:
    if source == "radgenome_region_abnormality":
        rows = build_radgenome_region_abnormality_rows(supervision_root, split)
    elif source == "ctrate_organ_disease_text":
        rows = build_ctrate_organ_disease_rows(supervision_root, split, include_global_disease=include_global_disease)
    else:
        raise ValueError(f"unsupported source: {source}")
    if allowed_ids is not None:
        rows = [row for row in rows if normalize_volume_id(str(row["volume_id"])) in allowed_ids]
    return dedupe_rows(rows, max_rows=max_rows)


def write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_text_atomic(path, text)


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def numpy_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype("float16")
    if name == "float32":
        return np.dtype("float32")
    raise ValueError(name)


def pooled_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        mask = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (last_hidden * mask).sum(dim=1) / denom
    if pooling == "last_nonpad":
        lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        return last_hidden[batch, lengths]
    raise ValueError(pooling)


def encode_embeddings(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    model_name_or_path: str,
    device: str,
    batch_size: int,
    max_length: int,
    dtype_name: str,
    embedding_dtype_name: str,
    pooling: str,
    resume: bool = False,
    encode_limit: int = 0,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path = output_path.with_name("encoded_mask.npy")

    dtype = torch_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(torch.device(device))
    model.eval()

    hidden_size = int(getattr(model.config, "hidden_size"))
    embedding_dtype = numpy_dtype(embedding_dtype_name)
    if resume and output_path.exists():
        embeddings = np.load(output_path, mmap_mode="r+")
        if embeddings.shape != (len(rows), hidden_size):
            raise ValueError(f"existing embedding shape mismatch: {embeddings.shape} vs {(len(rows), hidden_size)}")
        if embeddings.dtype != embedding_dtype:
            raise ValueError(f"existing embedding dtype mismatch: {embeddings.dtype} vs {embedding_dtype}")
    else:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.npy")
        if tmp_path.exists():
            tmp_path.unlink()
        embeddings = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=embedding_dtype,
            shape=(len(rows), hidden_size),
        )
        embeddings[:] = 0
        embeddings.flush()
        tmp_path.replace(output_path)
        embeddings = np.load(output_path, mmap_mode="r+")

    if resume and mask_path.exists():
        encoded_mask = np.load(mask_path, mmap_mode="r+")
        if encoded_mask.shape != (len(rows),):
            raise ValueError(f"existing encoded mask shape mismatch: {encoded_mask.shape} vs {(len(rows),)}")
    else:
        tmp_mask_path = mask_path.with_suffix(mask_path.suffix + ".tmp.npy")
        if tmp_mask_path.exists():
            tmp_mask_path.unlink()
        encoded_mask = np.lib.format.open_memmap(tmp_mask_path, mode="w+", dtype=np.bool_, shape=(len(rows),))
        encoded_mask[:] = False
        encoded_mask.flush()
        tmp_mask_path.replace(mask_path)
        encoded_mask = np.load(mask_path, mmap_mode="r+")

    missing_indices = np.flatnonzero(~np.asarray(encoded_mask, dtype=bool))
    if encode_limit > 0:
        missing_indices = missing_indices[: int(encode_limit)]
    if len(missing_indices) == 0:
        return {
            "hidden_size": hidden_size,
            "peak_gpu_mem_gb": 0.0,
            "encoded_rows": int(np.asarray(encoded_mask, dtype=bool).sum()),
            "total_rows": len(rows),
            "complete": bool(np.asarray(encoded_mask, dtype=bool).all()),
            "encoded_mask": str(mask_path),
        }

    peak_mem_gb = 0.0
    with torch.inference_mode():
        for start in range(0, len(missing_indices), batch_size):
            batch_indices = missing_indices[start : start + batch_size]
            texts = [rows[int(row_index)]["target_text"] for row_index in batch_indices]
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens, use_cache=False)
            emb = pooled_hidden(output.last_hidden_state, tokens["attention_mask"], pooling)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            embeddings[batch_indices] = emb.detach().cpu().numpy().astype(embedding_dtype, copy=False)
            encoded_mask[batch_indices] = True
            embeddings.flush()
            encoded_mask.flush()
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated(torch.device(device)) / (1024**3))
            encoded_rows = int(np.asarray(encoded_mask, dtype=bool).sum())
            if start == 0 or start + len(batch_indices) >= len(missing_indices) or ((start // batch_size) + 1) % 10 == 0:
                print(f"encoded {encoded_rows}/{len(rows)} rows -> {output_path}", flush=True)
    embeddings.flush()
    encoded_mask.flush()
    mask_array = np.asarray(encoded_mask, dtype=bool)
    return {
        "hidden_size": hidden_size,
        "peak_gpu_mem_gb": peak_mem_gb,
        "encoded_rows": int(mask_array.sum()),
        "total_rows": len(rows),
        "complete": bool(mask_array.all()),
        "encoded_mask": str(mask_path),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_volumes": len({row["volume_id"] for row in rows}),
        "organ_group_counts": dict(sorted(Counter(row["organ_group"] for row in rows).items())),
        "duplicate_rows_collapsed": int(sum(max(int(row.get("duplicate_count", 1)) - 1, 0) for row in rows)),
    }


def existing_split_summary(index_path: Path, embedding_path: Path, *, index_only: bool, max_rows: int, ids_file: str) -> dict[str, Any]:
    rows = list(read_jsonl(index_path))
    summary = summarize_rows(rows)
    summary.update(
        {
            "index": str(index_path),
            "embeddings": None if index_only else str(embedding_path),
            "index_only": bool(index_only),
            "max_rows_per_split": max_rows,
            "ids_file": str(ids_file) if ids_file else "",
            "skipped_existing": True,
        }
    )
    if not index_only and embedding_path.exists():
        embeddings = np.load(embedding_path, mmap_mode="r")
        mask_path = embedding_path.with_name("encoded_mask.npy")
        if mask_path.exists():
            encoded_mask = np.load(mask_path, mmap_mode="r")
            encoded_rows = int(np.asarray(encoded_mask, dtype=bool).sum())
            complete = bool(np.asarray(encoded_mask, dtype=bool).all())
        else:
            encoded_rows = int(embeddings.shape[0])
            complete = True
        summary.update(
            {
                "hidden_size": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
                "embedding_shape": list(embeddings.shape),
                "embedding_dtype": str(embeddings.dtype),
                "encoded_rows": encoded_rows,
                "total_rows": int(embeddings.shape[0]),
                "complete": complete,
                "encoded_mask": str(mask_path) if mask_path.exists() else "",
            }
        )
    return summary


def existing_metadata_matches_request(
    split_dir: Path,
    *,
    model_name_or_path: str,
    pooling: str,
    index_only: bool,
    max_rows: int,
    ids_file: str,
) -> bool:
    metadata_path = split_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    summary = metadata.get("summary", {})
    return (
        str(metadata.get("model_name_or_path", "")) == str(model_name_or_path)
        and str(metadata.get("embedding_contract", {}).get("pooling", "")) == str(pooling)
        and bool(summary.get("index_only", False)) == bool(index_only)
        and int(summary.get("max_rows_per_split", -1)) == int(max_rows)
        and str(summary.get("ids_file", "")) == str(ids_file or "")
    )


def main() -> int:
    args = parse_args()
    if args.overwrite and (args.skip_existing or args.resume):
        raise ValueError("--overwrite cannot be combined with --skip-existing or --resume")
    supervision_root = Path(args.supervision_root)
    out_root = Path(args.out_root)
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    max_rows = max(0, int(args.max_rows_per_split))
    allowed_ids = None
    if args.ids_file:
        allowed_ids = {normalize_volume_id(volume_id) for volume_id in read_ids(Path(args.ids_file))}
    summaries: dict[str, Any] = {}

    for source in sources:
        if source not in SOURCE_TO_FILE:
            raise ValueError(f"unsupported source={source!r}; supported={sorted(SOURCE_TO_FILE)}")
        summaries[source] = {"splits": {}}
        for split in splits:
            if split not in {"train", "valid"}:
                raise ValueError(f"unsupported split={split!r}")
            split_dir = out_root / source / split
            index_path = split_dir / "index.jsonl"
            embedding_path = split_dir / "embeddings.npy"
            mask_path = split_dir / "encoded_mask.npy"
            requested_outputs_exist = index_path.exists() if args.index_only else index_path.exists() and embedding_path.exists()
            metadata_matches = existing_metadata_matches_request(
                split_dir,
                model_name_or_path=str(args.model_name_or_path),
                pooling=str(args.pooling),
                index_only=bool(args.index_only),
                max_rows=max_rows,
                ids_file=str(args.ids_file) if args.ids_file else "",
            )
            if (args.skip_existing or args.resume) and requested_outputs_exist and metadata_matches:
                existing_summary = existing_split_summary(
                    index_path,
                    embedding_path,
                    index_only=bool(args.index_only),
                    max_rows=max_rows,
                    ids_file=str(args.ids_file) if args.ids_file else "",
                )
                if args.index_only or existing_summary.get("complete", False) or args.skip_existing:
                    existing_summary["skipped_existing"] = True
                    summaries[source]["splits"][split] = existing_summary
                    print(json.dumps({"source": source, "split": split, "summary": existing_summary}, ensure_ascii=False), flush=True)
                    continue
            if args.skip_existing and requested_outputs_exist and not metadata_matches:
                raise FileExistsError(
                    f"{split_dir} exists but metadata does not match this request; "
                    "use --overwrite, change --out-root, or remove the stale partial artifact"
                )
            if index_path.exists() and not (args.overwrite or args.resume):
                hint = "pass --overwrite or --skip-existing"
                raise FileExistsError(f"{index_path} exists; {hint}")
            if embedding_path.exists() and not (args.overwrite or args.resume) and not args.index_only:
                hint = "pass --overwrite or --skip-existing"
                raise FileExistsError(f"{embedding_path} exists; {hint}")

            rows = build_rows_for_source(
                source,
                supervision_root,
                split,
                max_rows=max_rows,
                include_global_disease=bool(args.include_global_disease),
                allowed_ids=allowed_ids,
            )
            if not (args.resume and index_path.exists()):
                write_index(index_path, rows)
            split_summary = summarize_rows(rows)
            split_summary.update(
                {
                    "index": str(index_path),
                    "embeddings": None if args.index_only else str(embedding_path),
                    "index_only": bool(args.index_only),
                    "max_rows_per_split": max_rows,
                    "ids_file": str(args.ids_file) if args.ids_file else "",
                    "resume": bool(args.resume),
                    "encode_limit": int(args.encode_limit),
                }
            )
            if not args.index_only:
                embed_summary = encode_embeddings(
                    rows=rows,
                    output_path=embedding_path,
                    model_name_or_path=str(args.model_name_or_path),
                    device=str(args.device),
                    batch_size=int(args.batch_size),
                    max_length=int(args.max_length),
                    dtype_name=str(args.torch_dtype),
                    embedding_dtype_name=str(args.embedding_dtype),
                    pooling=str(args.pooling),
                    resume=bool(args.resume),
                    encode_limit=max(0, int(args.encode_limit)),
                )
                split_summary.update(embed_summary)
            write_json_atomic(
                split_dir / "metadata.json",
                {
                    "created_at": utc_now(),
                    "source": source,
                    "split": split,
                    "model_name_or_path": str(args.model_name_or_path),
                    "embedding_backend": str(args.embedding_backend),
                    "embedding_contract": {
                        "index_jsonl": "line-aligned with embeddings.npy when embeddings are present",
                        "embeddings_npy": "[N,hidden_size] L2-normalized pooled text encoder hidden states",
                        "pooling": args.pooling,
                        "text_template": "Region: {organ_group}. Finding: {finding_text}.",
                    },
                    "summary": split_summary,
                },
            )
            summaries[source]["splits"][split] = split_summary
            print(json.dumps({"source": source, "split": split, "summary": split_summary}, ensure_ascii=False), flush=True)

    write_json_atomic(
        out_root / "metadata.json",
        {
            "created_at": utc_now(),
            "supervision_root": str(supervision_root),
            "model_name_or_path": str(args.model_name_or_path),
            "embedding_backend": str(args.embedding_backend),
            "index_only": bool(args.index_only),
            "skip_existing": bool(args.skip_existing),
            "resume": bool(args.resume),
            "encode_limit": int(args.encode_limit),
            "sources": sources,
            "splits": splits,
            "torch_dtype": args.torch_dtype,
            "embedding_dtype": args.embedding_dtype,
            "max_length": int(args.max_length),
            "ids_file": str(args.ids_file) if args.ids_file else "",
            "pooling": args.pooling,
            "summary": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
