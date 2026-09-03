"""Shared checkpoint helpers for DTBD3D training runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def tensor_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}


def string_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            out[key] = value
        elif value is None:
            out[key] = ""
        else:
            out[key] = json.dumps(value, sort_keys=True)
    return out


def optimizer_state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: optimizer_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [optimizer_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(optimizer_state_to_cpu(item) for item in value)
    return value


def full_checkpoint_paths(checkpoint_dir: Path, step: int) -> tuple[Path, Path, Path]:
    state_path = checkpoint_dir / f"training_state_step_{step}.safetensors"
    optimizer_path = checkpoint_dir / f"training_optimizer_step_{step}.pt"
    metadata_path = checkpoint_dir / f"training_state_step_{step}.json"
    return state_path, optimizer_path, metadata_path


def optimizer_path_for_state_checkpoint(state_path: Path) -> Path:
    stem = state_path.stem
    if stem.startswith("training_state_"):
        return state_path.with_name(stem.replace("training_state_", "training_optimizer_", 1) + ".pt")
    return state_path.with_name(stem + "_optimizer.pt")


def metadata_path_for_state_checkpoint(state_path: Path) -> Path:
    return state_path.with_suffix(".json")


def save_full_training_checkpoint(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    state_path, optimizer_path, metadata_path = full_checkpoint_paths(checkpoint_dir, step)
    payload = dict(metadata)
    payload.update(
        {
            "checkpoint_kind": "full_training_state",
            "full_training_checkpoint": str(state_path),
            "optimizer_checkpoint": str(optimizer_path),
            "step": step,
        }
    )
    save_file(tensor_state_dict(module), state_path, metadata=string_metadata(payload))
    torch.save(
        {
            "step": step,
            "optimizer": optimizer_state_to_cpu(optimizer.state_dict()),
        },
        optimizer_path,
    )
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return state_path


def read_full_checkpoint_step(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    if "step" in metadata and str(metadata["step"]).strip():
        return int(json.loads(metadata["step"]))
    sidecar = metadata_path_for_state_checkpoint(path)
    if sidecar.exists():
        payload = json.loads(sidecar.read_text())
        return int(payload["step"])
    raise ValueError(f"full training checkpoint is missing step metadata: {path}")


def load_full_training_checkpoint(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    _device: str | torch.device,
) -> int:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    state = load_file(str(checkpoint_path), device="cpu")
    module.load_state_dict(state, strict=True)
    optimizer_path = optimizer_path_for_state_checkpoint(checkpoint_path)
    if not optimizer_path.exists():
        raise FileNotFoundError(f"optimizer checkpoint not found for resume: {optimizer_path}")
    optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(optimizer_payload["optimizer"])
    checkpoint_step = int(optimizer_payload.get("step", read_full_checkpoint_step(checkpoint_path)))
    metadata_step = read_full_checkpoint_step(checkpoint_path)
    if checkpoint_step != metadata_step:
        raise ValueError(f"resume checkpoint step mismatch: state={metadata_step} optimizer={checkpoint_step}")
    return metadata_step
