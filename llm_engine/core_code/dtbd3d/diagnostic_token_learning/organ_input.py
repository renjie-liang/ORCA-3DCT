"""Input-side anatomy conditioning."""

from __future__ import annotations

import torch

from dtbd3d.diagnostic_token_learning.config import OrganInputConfig


def apply_organ_input_conditioning(
    input_data: torch.Tensor,
    importance_weights: torch.Tensor | None,
    config: OrganInputConfig,
) -> torch.Tensor:
    """Apply conservative input-side organ conditioning.

    The pretrained tokenizer expects a single CT channel, so the first supported
    mode keeps the input shape unchanged and applies a small residual intensity
    gate derived from the organ importance map.
    """
    if not config.enabled or config.strength == 0.0:
        return input_data
    if config.mode != "importance_residual":
        raise ValueError(f"unsupported organ_input.mode={config.mode!r}; expected importance_residual")
    if importance_weights is None:
        raise ValueError("organ input conditioning requires importance weights")
    if importance_weights.shape != input_data.shape:
        raise ValueError(f"importance weights {tuple(importance_weights.shape)} do not match input {tuple(input_data.shape)}")

    residual = (importance_weights.float() - 1.0) * float(config.strength)
    conditioned = input_data.float() * (1.0 + residual)
    if config.clamp_min is not None or config.clamp_max is not None:
        conditioned = conditioned.clamp(min=config.clamp_min, max=config.clamp_max)
    return conditioned.to(dtype=input_data.dtype)

