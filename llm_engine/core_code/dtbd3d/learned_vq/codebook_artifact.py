"""Codebook artifact export helpers for learned LFQ delta tables."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

CODEBOOK_CHANNEL_ORDERS = ("msb_identity", "msb_reverse_channels")
REPORTGEN_EXPECTED_CHANNEL_ORDER = "msb_reverse_channels"
SOURCE_CODEBOOK_CHANNEL_ORDER = "msb_identity"


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def metadata_float(metadata: dict[str, str], key: str, default: float) -> float:
    raw = metadata.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        try:
            return float(json.loads(raw))
        except Exception:
            return default


def load_delta_state(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    return state, metadata


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def build_codebook_array(
    delta_checkpoint: str | Path,
    codebook_mode: str,
    dtype: str,
    delta_scale_override: float | None = None,
    codebook_channel_order: str = REPORTGEN_EXPECTED_CHANNEL_ORDER,
) -> tuple[np.ndarray, float, dict[str, str]]:
    if codebook_channel_order not in CODEBOOK_CHANNEL_ORDERS:
        raise ValueError(f"unsupported codebook_channel_order {codebook_channel_order}; expected one of {CODEBOOK_CHANNEL_ORDERS}")
    state, checkpoint_metadata = load_delta_state(delta_checkpoint)
    if "base_code" not in state:
        raise KeyError(f"{delta_checkpoint} does not contain base_code")

    base_code = state["base_code"].float()
    if codebook_mode == "learned_codebook":
        if "delta" not in state:
            raise KeyError(f"{delta_checkpoint} does not contain delta")
        delta_scale = (
            float(delta_scale_override)
            if delta_scale_override is not None
            else metadata_float(checkpoint_metadata, "delta_scale", 0.1)
        )
        codebook = base_code + delta_scale * state["delta"].float()
    elif codebook_mode == "lfq_binary":
        delta_scale = 0.0
        codebook = base_code
    else:
        raise ValueError(f"unsupported codebook_mode {codebook_mode}")

    if codebook.ndim != 2:
        raise ValueError(f"codebook must be 2D, got {tuple(codebook.shape)}")
    array = codebook.cpu().numpy()
    if codebook_channel_order == "msb_reverse_channels":
        array = array[..., ::-1]
    array = array.astype(np.float16 if dtype == "float16" else np.float32)
    return array, delta_scale, checkpoint_metadata


def export_codebook_artifact(
    delta_checkpoint: str | Path,
    out_dir: str | Path,
    codebook_mode: str,
    dtype: str = "float16",
    label: str = "",
    compression: str = "16x16x8",
    tokenizer_checkpoint: str | Path | None = None,
    token_artifact_dir: str | Path | None = None,
    step: int | None = None,
    delta_scale_override: float | None = None,
    codebook_channel_order: str = REPORTGEN_EXPECTED_CHANNEL_ORDER,
    reportgen_expected_channel_order: str = REPORTGEN_EXPECTED_CHANNEL_ORDER,
) -> dict[str, Any]:
    delta_checkpoint = Path(delta_checkpoint).resolve()
    out_dir = Path(out_dir).resolve()
    if not delta_checkpoint.exists():
        raise FileNotFoundError(delta_checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    array, delta_scale, checkpoint_metadata = build_codebook_array(
        delta_checkpoint=delta_checkpoint,
        codebook_mode=codebook_mode,
        dtype=dtype,
        delta_scale_override=delta_scale_override,
        codebook_channel_order=codebook_channel_order,
    )
    np.save(out_dir / "codebook.npy", array)
    requires_reverse = codebook_channel_order != reportgen_expected_channel_order

    metadata: dict[str, Any] = {
        "artifact_type": "dtbd3d_codebook_artifact",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "codebook_mode": codebook_mode,
        "source_codebook_channel_order": SOURCE_CODEBOOK_CHANNEL_ORDER,
        "codebook_channel_order": codebook_channel_order,
        "reportgen_expected_channel_order": reportgen_expected_channel_order,
        "requires_channel_reverse_for_btb3d_pretrained_reportgen": requires_reverse,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "compression": compression,
        "delta_checkpoint": str(delta_checkpoint),
        "tokenizer_checkpoint": str(tokenizer_checkpoint or ""),
        "token_artifact_dir": str(token_artifact_dir or ""),
        "delta_scale": delta_scale,
        "source_checkpoint_metadata": checkpoint_metadata,
        "lookup_contract": (
            "token_id -> codebook[token_id] -> visual feature"
            if not requires_reverse
            else "token_id -> codebook[token_id][..., ::-1] -> visual feature"
        ),
    }
    if step is not None:
        metadata["step"] = int(step)
    write_json_atomic(out_dir / "metadata.json", metadata)
    return metadata
