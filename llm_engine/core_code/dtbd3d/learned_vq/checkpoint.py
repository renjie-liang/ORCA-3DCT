"""Checkpoint helpers for Sub-task 6 delta tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from .lfq_table_adapter import LFQDeltaTableAdapter


def _string_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    if not metadata:
        return {}
    out: dict[str, str] = {}
    for key, value in metadata.items():
        out[str(key)] = value if isinstance(value, str) else json.dumps(value)
    return out


def save_delta_checkpoint(
    adapter: LFQDeltaTableAdapter,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save the delta table and base LFQ codebook."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items()}
    save_file(state, path, metadata=_string_metadata(metadata))
    return path


def load_delta_checkpoint(
    adapter: LFQDeltaTableAdapter,
    path: str | Path,
    strict: bool = True,
) -> None:
    """Load a delta-table checkpoint into an existing adapter."""
    state = load_file(str(path), device="cpu")
    adapter.load_state_dict(state, strict=strict)
