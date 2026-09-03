"""Checkpoint-loading helpers for BTB3D report-generation components."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


def load_mm_projector_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load projector weights saved by the DTBD3D training wrapper."""

    state = load_file(checkpoint_dir / "mm_projector.safetensors")
    return {key.removeprefix("mm_projector."): value for key, value in state.items()}


def load_new_token_embedding_rows(checkpoint_dir: Path) -> torch.Tensor:
    """Load the trainable special-token embedding rows."""

    return load_file(checkpoint_dir / "new_token_embeddings.safetensors")["model.embed_tokens.new_rows"]

