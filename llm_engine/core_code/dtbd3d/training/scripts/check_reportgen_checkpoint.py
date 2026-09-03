"""Validate a saved BTB3D report-generation checkpoint on disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "base_vocab_size.json",
    "mm_projector.safetensors",
    "new_token_embeddings.safetensors",
    "training_model_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    return parser.parse_args()


def _inspect_safetensors(path: Path) -> dict:
    tensors = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensors.append(
                {
                    "name": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                }
            )
    empty = [item for item in tensors if item["numel"] == 0]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "n_tensors": len(tensors),
        "n_empty_tensors": len(empty),
        "empty_tensor_names": [item["name"] for item in empty[:20]],
        "examples": tensors[:5],
    }


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise SystemExit(f"missing checkpoint directory: {checkpoint_dir}")

    missing = [name for name in REQUIRED_FILES if not (checkpoint_dir / name).exists()]
    if missing:
        raise SystemExit(f"missing checkpoint files: {missing}")

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "files": {},
        "base_vocab_size": json.loads((checkpoint_dir / "base_vocab_size.json").read_text())["base_vocab_size"],
    }
    for name in ["adapter_model.safetensors", "mm_projector.safetensors", "new_token_embeddings.safetensors"]:
        report["files"][name] = _inspect_safetensors(checkpoint_dir / name)

    failures = []
    for name, info in report["files"].items():
        if info["n_tensors"] == 0:
            failures.append(f"{name} contains no tensors")
        if info["n_empty_tensors"] > 0:
            failures.append(f"{name} contains {info['n_empty_tensors']} empty tensors")

    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("checkpoint integrity failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
