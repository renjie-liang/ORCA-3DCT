"""LFQ-id delta table adapter.

This module keeps BTB3D's LFQ id generation intact, then replaces the fixed
decoder-side {-1,+1} code with a learned delta table:

    embedding[id] = base_lfq_code[id] + delta_scale * delta[id]

For encoder-unfrozen experiments, the adapter preserves LFQ's straight-through
gradient path by using the hard learned table in the forward pass while routing
the reconstruction gradient through the original LFQ quantized tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


@dataclass
class DeltaTableOutput:
    decoded: torch.Tensor
    z_quantized: torch.Tensor
    token_ids: torch.Tensor
    quantized_output: object
    quantize_loss_breakdown: object
    delta_l2_loss: torch.Tensor


class LFQDeltaTableAdapter(nn.Module):
    """Learn a decoder-side delta table over existing LFQ token ids."""

    def __init__(
        self,
        quantizer: nn.Module,
        vocab_size: int,
        embedding_dim: int,
        delta_scale: float = 0.1,
        preserve_encoder_ste: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if delta_scale <= 0:
            raise ValueError(f"delta_scale must be positive, got {delta_scale}")

        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.delta_scale = float(delta_scale)
        self.preserve_encoder_ste = bool(preserve_encoder_ste)

        quantizer_device = getattr(quantizer, "mask", torch.empty((), device="cpu")).device
        with torch.no_grad():
            ids = torch.arange(self.vocab_size, dtype=torch.long, device=quantizer_device)
            base_code = quantizer.indices_to_codes(ids, project_out=True).float()
        if tuple(base_code.shape) != (self.vocab_size, self.embedding_dim):
            raise ValueError(
                f"base LFQ code shape {tuple(base_code.shape)} does not match "
                f"{(self.vocab_size, self.embedding_dim)}"
            )
        self.register_buffer("base_code", base_code, persistent=True)
        self.delta = nn.Parameter(torch.zeros_like(base_code))

    def lookup(self, token_ids: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Return learned embeddings for LFQ token ids."""
        if token_ids.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
            token_ids = token_ids.long()
        token_ids = token_ids.long()
        base = F.embedding(token_ids, self.base_code)
        delta = F.embedding(token_ids, self.delta)
        out = base + self.delta_scale * delta
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out

    def delta_l2_loss(self, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Mean squared delta over used ids, or over the full table if omitted."""
        if token_ids is None:
            return self.delta.float().pow(2).mean()
        unique_ids = torch.unique(token_ids.detach().long())
        if unique_ids.numel() == 0:
            return self.delta.new_zeros(())
        return self.delta.index_select(0, unique_ids).float().pow(2).mean()

    def forward(
        self,
        tokenizer_module: nn.Module,
        input_data: torch.Tensor,
        entropy_loss_weight: float,
        calculate_quantize_loss: bool,
        use_distributed_batch_entropy: bool | None,
    ) -> DeltaTableOutput:
        """Encode with LFQ ids, replace decoder latent with learned table, decode."""
        z_base, quantized_output, breakdown = tokenizer_module.encode(
            input_data,
            entropy_loss_weight=entropy_loss_weight,
            use_distributed_batch_entropy=use_distributed_batch_entropy,
            calculate_quantize_loss=calculate_quantize_loss,
        )
        token_ids = quantized_output.indices
        learned_flat = self.lookup(token_ids, dtype=quantized_output.quantized.dtype)
        base_flat = quantized_output.quantized
        if learned_flat.shape != base_flat.shape:
            raise ValueError(f"learned/base latent shape mismatch: {learned_flat.shape} vs {base_flat.shape}")

        if self.training and self.preserve_encoder_ste:
            decoder_flat = base_flat + (learned_flat - base_flat.detach())
        else:
            decoder_flat = learned_flat

        if decoder_flat.ndim != 3:
            raise ValueError(f"expected flat decoder latent B,N,C, got {decoder_flat.shape}")
        if z_base.ndim == 4:
            _, _, h, w = z_base.shape
            z_delta = rearrange(decoder_flat, "b (h w) c -> b c h w", h=h, w=w)
        elif z_base.ndim == 5:
            _, _, t, h, w = z_base.shape
            z_delta = rearrange(decoder_flat, "b (t h w) c -> b c t h w", t=t, h=h, w=w)
        else:
            raise ValueError(f"unsupported quantized latent shape: {z_base.shape}")

        decoded = tokenizer_module.decode(z_delta)
        return DeltaTableOutput(
            decoded=decoded,
            z_quantized=z_delta,
            token_ids=token_ids,
            quantized_output=quantized_output,
            quantize_loss_breakdown=breakdown,
            delta_l2_loss=self.delta_l2_loss(token_ids),
        )
