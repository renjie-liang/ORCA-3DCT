#!/usr/bin/env python3
"""Final eval bundle for learned LFQ tables.

The bundle always exports the report-generation codebook, compact token
artifacts for train/valid, and reconstruction metrics for valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dtbd3d.core.btb3d_model import expected_token_count
from dtbd3d.encoder_training.pipeline_config import load_pipeline_config
from dtbd3d.encoder_training.tiny_recon_train import (
    load_model,
    maybe_suppress_model_stdout,
    read_valid_ids,
    score_recon,
)
from dtbd3d.training.data.cached_ct import make_valid_dataloader
from dtbd3d.learned_vq import (
    LFQDeltaTableAdapter,
    export_codebook_artifact,
    load_delta_checkpoint,
    metadata_float,
    read_safetensors_metadata,
)

CODEBOOK_MODE = "learned_codebook"
CODEBOOK_DTYPE = "float16"
CODEBOOK_CHANNEL_ORDER = "msb_reverse_channels"
TOKEN_SPLITS = ("train", "valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Sub-task 6 YAML config.")
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--delta-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True, help="Output directory, usually run_dir/final_eval/step_xxxxxx.")
    parser.add_argument("--label", default="")
    parser.add_argument("--step", type=int, default=-1, help="Step label. Default parses tokenizer checkpoint name.")
    parser.add_argument("--device", default="", help="Override config validation.device, e.g. cuda:0.")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="0 uses config validation.eval_batch_size.")
    parser.add_argument("--n-train", type=int, default=0, help="Number of train volumes. 0 uses all train ids.")
    parser.add_argument("--n-valid", type=int, default=0, help="Number of valid volumes. 0 uses all valid ids.")
    return parser.parse_args()


def parse_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(ids) + ("\n" if ids else ""))
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_items(config: Any, split: str, n_train: int, n_valid: int) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    if split == "train":
        ids_file = config.train.ids_file
        cache_root = config.train.cache_root
        cache_split = config.train.cache_split
        n = n_train
        limit_source = "cli.n_train" if n_train > 0 else "full_split"
    elif split == "valid":
        ids_file = config.validation.ids_file
        cache_root = config.validation.cache_root
        cache_split = config.validation.cache_split
        if n_valid < 0:
            n = 0
            limit_source = "full_valid"
        else:
            n = n_valid if n_valid > 0 else config.validation.n_valid
            limit_source = "cli.n_valid" if n_valid > 0 else "config.validation.n_valid"
    else:
        raise ValueError(f"unsupported split {split}; expected train or valid")

    all_ids = read_valid_ids(ids_file, 0)
    ids = all_ids[:n] if n > 0 else all_ids
    items = [(volume_id, cache_root / cache_split / f"{volume_id}.npy") for volume_id in ids]
    for _volume_id, path in items:
        if not path.exists():
            raise FileNotFoundError(path)
    return items, {
        "split": split,
        "ids_file": str(ids_file),
        "cache_root": str(cache_root),
        "cache_split": cache_split,
        "ids_total": len(all_ids),
        "requested_n": int(n),
        "limit_source": limit_source,
        "n": len(items),
        "exported_n": len(items),
        "is_full_split": len(items) == len(all_ids),
        "subset_policy": "full_split" if len(items) == len(all_ids) else "first_n_in_ids_file_order",
    }


def build_model_args(config: Any, tokenizer_checkpoint: Path, device: str) -> argparse.Namespace:
    root = project_root()
    return argparse.Namespace(
        compression=config.train.compression,
        tokenizer_checkpoint=str(tokenizer_checkpoint),
        btb3d_repo=str(root / "Experiment/core_code/btb3d_baseline/encoder-decoder"),
        weights_root=str(root / "./data/btb3d_weights"),
        commitment_cost=config.train.commitment_cost,
        diversity_gamma=config.train.diversity_gamma,
        device=device,
        show_model_stdout=False,
    )


def summarize_recon(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["ssim", "psnr", "mse", "mae"]
    summary: dict[str, Any] = {"n": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if field in row and math.isfinite(float(row[field]))]
        summary[f"mean_{field}"] = float(np.mean(values)) if values else float("nan")
    return summary


def export_split_tokens_and_recon(
    model: torch.nn.Module,
    adapter: LFQDeltaTableAdapter,
    items: list[tuple[str, Path]],
    split_metadata: dict[str, Any],
    out_dir: Path,
    device: str,
    eval_batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    expected_tokens: int,
    compression: str,
    run_recon_eval: bool,
) -> dict[str, Any]:
    split = str(split_metadata["split"])
    token_dir = out_dir / "reportgen_artifact" / "token_artifact" / split
    recon_dir = out_dir / "reconstruction" / split

    ids = [volume_id for volume_id, _path in items]
    token_dir.mkdir(parents=True, exist_ok=True)
    write_ids(token_dir / "ids.txt", ids)
    token_matrix = np.lib.format.open_memmap(
        token_dir / "tokens_int.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(len(items), expected_tokens),
    )

    rows: list[dict[str, Any]] = []
    id_to_row = {volume_id: index for index, volume_id in enumerate(ids)}
    model.eval()
    adapter.eval()

    loader = make_valid_dataloader(
        [path for _volume_id, path in items],
        batch_size=eval_batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    with torch.no_grad():
        for batch in loader:
            volume_ids = list(batch["volume_id"])
            inp = batch["ct"].to(device, dtype=torch.bfloat16, non_blocking=pin_memory)
            with maybe_suppress_model_stdout(False):
                out = adapter(
                    model.tokenizer,
                    inp,
                    entropy_loss_weight=0.0,
                    calculate_quantize_loss=False,
                    use_distributed_batch_entropy=False,
                )

            token_rows = out.token_ids.detach().cpu().numpy().reshape(len(volume_ids), -1).astype(np.uint32)
            if token_rows.shape[1] != expected_tokens:
                raise ValueError(f"{split} token row width {token_rows.shape[1]}; expected {expected_tokens}")
            for batch_index, volume_id in enumerate(volume_ids):
                token_matrix[id_to_row[volume_id]] = token_rows[batch_index]

            if run_recon_eval:
                inp_batch = inp[:, 0].float().cpu().numpy()
                recon_batch = np.clip(out.decoded[:, 0].float().cpu().numpy(), -1.0, 1.0)
                for batch_index, volume_id in enumerate(volume_ids):
                    inp_np = inp_batch[batch_index]
                    recon_np = recon_batch[batch_index]
                    min_d = min(inp_np.shape[0], recon_np.shape[0])
                    min_h = min(inp_np.shape[1], recon_np.shape[1])
                    min_w = min(inp_np.shape[2], recon_np.shape[2])
                    inp_np = inp_np[:min_d, :min_h, :min_w]
                    recon_np = recon_np[:min_d, :min_h, :min_w]
                    row = {
                        "volume_id": volume_id,
                        "token_shape": tuple(int(x) for x in out.z_quantized[batch_index].shape),
                    }
                    row.update(score_recon(inp_np, recon_np, tuple(slice(0, int(dim)) for dim in inp_np.shape)))
                    rows.append(row)

    token_matrix.flush()
    ids_path = token_dir / "ids.txt"
    tokens_int_path = token_dir / "tokens_int.npy"
    token_metadata = {
        "artifact_type": "dtbd3d_compact_token_artifact",
        "format": "compact_tokens_int_matrix",
        "row_contract": "ids.txt line i corresponds to tokens_int.npy row i",
        "split": split,
        "n": len(items),
        "expected_tokens": expected_tokens,
        "tokens_per_volume": expected_tokens,
        "tokens_shape": [len(items), expected_tokens],
        "tokens_dtype": "uint32",
        "ids_path": str(ids_path),
        "tokens_int_path": str(tokens_int_path),
        "ids_relpath": "ids.txt",
        "tokens_int_relpath": "tokens_int.npy",
        "compression": compression,
        **split_metadata,
    }
    json_dump(token_dir / "metadata.json", token_metadata)

    recon_summary = summarize_recon(rows) if run_recon_eval else {"n": 0}
    if run_recon_eval:
        recon_summary = {
            **recon_summary,
            "metric_scope": "full_cached_preprocessed_tensor",
            "split": split,
            **split_metadata,
        }
        json_dump(recon_dir / "summary.json", recon_summary)
        write_csv(recon_dir / "per_volume_metrics.csv", rows)

    return {
        "split": split,
        "n": len(items),
        "token_artifact_dir": str(token_dir),
        "tokens_int": str(token_dir / "tokens_int.npy"),
        "tokens_shape": [len(items), expected_tokens],
        "tokens_per_volume": expected_tokens,
        "tokens_dtype": "uint32",
        "reconstruction_dir": str(recon_dir) if run_recon_eval else None,
        "reconstruction": recon_summary if run_recon_eval else None,
    }


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    config_path: Path,
    tokenizer_checkpoint: Path,
    delta_checkpoint: Path,
    step: int,
    codebook_metadata: dict[str, Any] | None,
    split_outputs: list[dict[str, Any]],
    elapsed_sec: float,
) -> None:
    reportgen_dir = out_dir / "reportgen_artifact"
    token_artifacts = {
        item["split"]: {
            "dir": os.path.relpath(str(item["token_artifact_dir"]), str(reportgen_dir)),
            "ids": os.path.relpath(str(Path(item["token_artifact_dir"]) / "ids.txt"), str(reportgen_dir)),
            "tokens_int": os.path.relpath(str(Path(item["token_artifact_dir"]) / "tokens_int.npy"), str(reportgen_dir)),
            "tokens_shape": item.get("tokens_shape"),
            "tokens_dtype": "uint32",
            "row_contract": "ids.txt line i corresponds to tokens_int.npy row i",
        }
        for item in split_outputs
        if item.get("token_artifact_dir")
    }
    manifest = {
        "artifact_type": "dtbd3d_final_eval_export",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "step": step,
        "config": str(config_path),
        "tokenizer_checkpoint": str(tokenizer_checkpoint),
        "delta_checkpoint": str(delta_checkpoint),
        "codebook_mode": CODEBOOK_MODE,
        "codebook_channel_order": codebook_metadata.get("codebook_channel_order") if codebook_metadata else None,
        "reportgen_expected_channel_order": (
            codebook_metadata.get("reportgen_expected_channel_order") if codebook_metadata else "msb_reverse_channels"
        ),
        "requires_channel_reverse_for_btb3d_pretrained_reportgen": (
            codebook_metadata.get("requires_channel_reverse_for_btb3d_pretrained_reportgen") if codebook_metadata else None
        ),
        "codebook_artifact_dir": "codebook_artifact" if codebook_metadata is not None else None,
        "token_artifacts": token_artifacts,
        "splits": split_outputs,
        "feature_contract": "token_id -> codebook[token_id] -> visual feature",
        "reportgen_load_contract": {
            "codebook": "load codebook_artifact/codebook.npy",
            "tokens": "for each split, load token_artifacts[split].tokens_int with ids row order",
            "lookup": (
                codebook_metadata.get("lookup_contract")
                if codebook_metadata
                else "visual_features = codebook[tokens_int]"
            ),
        },
        "elapsed_sec": elapsed_sec,
    }
    json_dump(reportgen_dir / "manifest.json", manifest)
    json_dump(out_dir / "metadata.json", manifest)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    tokenizer_checkpoint = Path(args.tokenizer_checkpoint).resolve()
    delta_checkpoint = Path(args.delta_checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not tokenizer_checkpoint.exists():
        raise FileNotFoundError(tokenizer_checkpoint)
    if not delta_checkpoint.exists():
        raise FileNotFoundError(delta_checkpoint)

    config = load_pipeline_config(config_path)
    step = args.step if args.step >= 0 else parse_step(tokenizer_checkpoint)
    device = args.device or config.validation.device
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else config.validation.eval_batch_size
    if eval_batch_size < 1:
        raise ValueError(f"--eval-batch-size must be >= 1, got {eval_batch_size}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    np.random.seed(0)
    torch.manual_seed(0)

    codebook_metadata = export_codebook_artifact(
        delta_checkpoint=delta_checkpoint,
        tokenizer_checkpoint=tokenizer_checkpoint,
        out_dir=out_dir / "reportgen_artifact" / "codebook_artifact",
        codebook_mode=CODEBOOK_MODE,
        dtype=CODEBOOK_DTYPE,
        label=args.label,
        compression=config.train.compression,
        step=step,
        codebook_channel_order=CODEBOOK_CHANNEL_ORDER,
    )

    split_outputs: list[dict[str, Any]] = []
    checkpoint_metadata = read_safetensors_metadata(delta_checkpoint)
    delta_scale = metadata_float(
        checkpoint_metadata,
        "delta_scale",
        float(config.raw.get("learned_table", {}).get("delta_scale", 0.1)),
    )
    model_args = build_model_args(config, tokenizer_checkpoint, device)
    model = load_model(model_args)
    model.eval()
    adapter = LFQDeltaTableAdapter(
        model.tokenizer.quantize,
        vocab_size=model.codebook_size,
        embedding_dim=model.config.model.quantize_model.token_size,
        delta_scale=delta_scale,
        preserve_encoder_ste=bool(config.raw.get("learned_table", {}).get("preserve_encoder_ste", True)),
    ).to(device)
    load_delta_checkpoint(adapter, delta_checkpoint)
    adapter.eval()
    expected_tokens = expected_token_count(config.train.compression)

    for split in TOKEN_SPLITS:
        effective_n_valid = -1 if args.n_valid == 0 else args.n_valid
        items, metadata = split_items(config, split, args.n_train, effective_n_valid)
        split_outputs.append(
            export_split_tokens_and_recon(
                model=model,
                adapter=adapter,
                items=items,
                split_metadata=metadata,
                out_dir=out_dir,
                device=device,
                eval_batch_size=eval_batch_size,
                num_workers=config.train.num_workers,
                prefetch_factor=config.train.prefetch_factor,
                pin_memory=config.train.pin_memory,
                expected_tokens=expected_tokens,
                compression=config.train.compression,
                run_recon_eval=(split == "valid"),
            )
        )

    elapsed_sec = time.perf_counter() - start
    write_manifest(
        out_dir=out_dir,
        args=args,
        config_path=config_path,
        tokenizer_checkpoint=tokenizer_checkpoint,
        delta_checkpoint=delta_checkpoint,
        step=step,
        codebook_metadata=codebook_metadata,
        split_outputs=split_outputs,
        elapsed_sec=elapsed_sec,
    )
    print(f"wrote final eval export: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
