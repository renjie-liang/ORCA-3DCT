"""Load/save helpers for learned-table reconstruction training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from dtbd3d.core.btb3d_model import CHECKPOINT_DIRS, CONFIGS
from dtbd3d.learned_vq import save_delta_checkpoint
from dtbd3d.training.checkpoint import save_full_training_checkpoint


DEFAULT_BTB3D_REPO = "Experiment/core_code/btb3d_baseline/encoder-decoder"
DEFAULT_BTB3D_WEIGHTS_ROOT = "./data/btb3d_weights"


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def build_btb3d_model(args: argparse.Namespace) -> torch.nn.Module:
    root = project_root()
    btb3d_repo = root / DEFAULT_BTB3D_REPO
    sys.path.insert(0, str(btb3d_repo))
    from modeling.magvit_model import VisionTokenizer
    from src.utils import get_config

    config_file = btb3d_repo / "configs" / CONFIGS[args.compression]
    if not config_file.exists():
        raise FileNotFoundError(config_file)

    model_config = get_config(str(config_file))
    return VisionTokenizer(
        config=model_config,
        commitment_cost=args.commitment_cost,
        diversity_gamma=args.diversity_gamma,
        use_gan=False,
        use_lecam_ema=False,
        use_perceptual=False,
    )


def load_author_tokenizer_checkpoint(model: torch.nn.Module, args: argparse.Namespace) -> Path:
    root = project_root()
    weights_root = root / DEFAULT_BTB3D_WEIGHTS_ROOT
    ckpt_file = weights_root / "encoder-decoder" / CHECKPOINT_DIRS[args.compression] / "3rd_stage.ckpt"
    if not ckpt_file.exists():
        raise FileNotFoundError(ckpt_file)
    states = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    model.tokenizer.load_state_dict(states, strict=True)
    return ckpt_file


def load_tokenizer_override(model: torch.nn.Module, args: argparse.Namespace) -> Path | None:
    if args.tokenizer_checkpoint:
        tokenizer_state = load_file(args.tokenizer_checkpoint, device="cpu")
        model.tokenizer.load_state_dict(tokenizer_state, strict=True)
        return Path(args.tokenizer_checkpoint)
    return None


def load_model(args: argparse.Namespace, *, load_fresh_weights: bool = True) -> torch.nn.Module:
    model = build_btb3d_model(args)
    if load_fresh_weights:
        load_author_tokenizer_checkpoint(model, args)
        load_tokenizer_override(model, args)
    model.train()
    model.to(args.device).to(torch.bfloat16)
    return model


def save_tokenizer_checkpoint(
    model: torch.nn.Module,
    out_dir: Path,
    step: int,
    checkpoint_dir: Path | None = None,
) -> Path:
    ckpt_dir = checkpoint_dir if checkpoint_dir is not None else out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.tokenizer.state_dict().items()}
    path = ckpt_dir / f"tokenizer_step_{step}.safetensors"
    save_file(state, path)
    return path


def diagnostic_training_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    if any(key in raw_config for key in ("reconstruction", "input_conditioning", "auxiliary_losses")):
        return {
            "reconstruction": raw_config.get("reconstruction", {}),
            "input_conditioning": raw_config.get("input_conditioning", {}),
            "auxiliary_losses": raw_config.get("auxiliary_losses", {}),
        }
    return {"supervision": raw_config.get("supervision", {})}


def checkpoint_metadata(args: argparse.Namespace, step: int, tokenizer_checkpoint: str | None) -> dict[str, Any]:
    return {
        "step": step,
        "checkpoint_format": "dtbd3d_learned_table_recon",
        "tokenizer_variant": "subtask6_v12_delta_table",
        "base_lfq_checkpoint": args.tokenizer_checkpoint,
        "tokenizer_checkpoint": tokenizer_checkpoint,
        "trainable_modules": args.trainable_mode,
        "loss_variant": args.loss_variant,
        "token_ids_semantics": "lfq_ids_with_learned_decoder_table",
        "delta_scale": args.delta_scale,
        "delta_l2_weight": args.delta_l2_weight,
        "delta_lr": args.delta_lr,
        "decoder_lr": args.decoder_lr,
        "encoder_lr": args.encoder_lr,
        "preserve_encoder_ste": args.preserve_encoder_ste,
        "diagnostic_training": diagnostic_training_config(args.raw_config),
    }


def save_checkpoints(
    train_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
) -> tuple[Path, Path, Path]:
    if hasattr(train_module, "module"):
        train_module = train_module.module
    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    tokenizer_path = save_tokenizer_checkpoint(train_module.model, out_dir, step, checkpoint_dir)
    delta_path = checkpoint_dir / f"delta_table_step_{step}.safetensors"
    save_delta_checkpoint(train_module.adapter, delta_path, metadata=checkpoint_metadata(args, step, str(tokenizer_path)))
    metadata = checkpoint_metadata(args, step, str(tokenizer_path))
    metadata["delta_checkpoint"] = str(delta_path)
    full_state_path = save_full_training_checkpoint(train_module, optimizer, checkpoint_dir, step, metadata)
    if train_module.supervision is not None and any(param.requires_grad for param in train_module.supervision.parameters()):
        torch.save(
            {
                "step": step,
                "state_dict": train_module.supervision.state_dict(),
                "config": diagnostic_training_config(args.raw_config),
            },
            checkpoint_dir / f"diagnostic_supervision_step_{step}.pt",
        )
    return tokenizer_path, delta_path, full_state_path
