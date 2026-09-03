"""BTB3D encoder-decoder model loading helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


CONFIGS = {
    "16x16x8": "magvit2_3d_model_config.yaml",
    "8x8x8": "magvit2_3d_model_config_8x8x8.yaml",
}

CHECKPOINT_DIRS = {
    "16x16x8": "16_16_8",
    "8x8x8": "8_8_8",
}

TOKEN_LAYOUTS = {
    "16x16x8": (31, 32, 32),
    "8x8x8": (31, 64, 64),
}


def load_btb3d_tokenizer(
    compression: str,
    btb3d_repo: str | Path,
    weights_root: str | Path,
    device: str,
    tokenizer_checkpoint: str | Path | None = None,
):
    """Load the BTB3D `VisionTokenizer` wrapper and return it in eval mode."""

    if compression not in CONFIGS:
        raise ValueError(f"unsupported compression: {compression}")

    btb3d_repo = Path(btb3d_repo)
    weights_root = Path(weights_root)
    config_file = btb3d_repo / "configs" / CONFIGS[compression]
    ckpt_file = weights_root / "encoder-decoder" / CHECKPOINT_DIRS[compression] / "3rd_stage.ckpt"

    sys.path.insert(0, str(btb3d_repo))
    from modeling.magvit_model import VisionTokenizer
    from src.utils import get_config

    model_config = get_config(str(config_file))
    model = VisionTokenizer(
        config=model_config,
        commitment_cost=0,
        diversity_gamma=0,
        use_gan=False,
        use_lecam_ema=False,
        use_perceptual=False,
    )
    states = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    model.tokenizer.load_state_dict(states, strict=True)
    if tokenizer_checkpoint:
        tokenizer_checkpoint = Path(tokenizer_checkpoint)
        if not tokenizer_checkpoint.exists():
            raise FileNotFoundError(tokenizer_checkpoint)
        if tokenizer_checkpoint.suffix == ".safetensors":
            tuned_states = load_file(str(tokenizer_checkpoint), device="cpu")
        else:
            tuned_states = torch.load(tokenizer_checkpoint, map_location="cpu", weights_only=True)
        model.tokenizer.load_state_dict(tuned_states, strict=True)
    model.eval()
    model.to(device).to(torch.bfloat16)
    print(f"Model loaded: {ckpt_file}")
    if tokenizer_checkpoint:
        print(f"Tokenizer override loaded: {tokenizer_checkpoint}")
    return model


def expected_token_count(compression: str) -> int:
    token_t, token_h, token_w = TOKEN_LAYOUTS[compression]
    return token_t * token_h * token_w
