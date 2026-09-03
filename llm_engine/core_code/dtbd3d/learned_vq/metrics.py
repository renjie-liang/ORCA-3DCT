"""Metrics for Sub-task 6 learned LFQ-id tables."""

from __future__ import annotations

import math
from typing import Any

import torch

from .lfq_table_adapter import LFQDeltaTableAdapter


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


@torch.no_grad()
def delta_table_metrics(adapter: LFQDeltaTableAdapter, token_ids: torch.Tensor | None = None) -> dict[str, float]:
    """Summarize delta magnitude globally and, optionally, over touched ids."""
    delta = adapter.delta.detach().float()
    embedding = adapter.base_code.detach().float() + adapter.delta_scale * delta
    metrics = {
        "delta_abs_mean": _as_float(delta.abs().mean()),
        "delta_abs_max": _as_float(delta.abs().max()),
        "delta_l2_mean": _as_float(delta.pow(2).sum(dim=1).sqrt().mean()),
        "embedding_mean": _as_float(embedding.mean()),
        "embedding_std": _as_float(embedding.std(unbiased=False)),
    }
    if token_ids is not None:
        unique_ids = torch.unique(token_ids.detach().long())
        touched = delta.index_select(0, unique_ids).float() if unique_ids.numel() else delta[:0]
        metrics.update(
            {
                "batch_unique_codes": float(unique_ids.numel()),
                "batch_unique_code_ratio": float(unique_ids.numel() / max(1, token_ids.numel())),
                "touched_delta_abs_mean": _as_float(touched.abs().mean()) if touched.numel() else 0.0,
                "touched_delta_abs_max": _as_float(touched.abs().max()) if touched.numel() else 0.0,
            }
        )
    return metrics


@torch.no_grad()
def code_usage_metrics(token_ids: torch.Tensor, vocab_size: int) -> dict[str, Any]:
    """Compute per-batch code usage metrics."""
    flat = token_ids.detach().long().reshape(-1)
    if flat.numel() == 0:
        return {
            "num_tokens": 0,
            "unique_codes": 0,
            "unique_code_ratio": 0.0,
            "code_usage_entropy": 0.0,
            "code_usage_perplexity": 0.0,
            "top1_code_fraction": 0.0,
        }
    unique, counts = torch.unique(flat, return_counts=True)
    probs = counts.float() / counts.sum().float()
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return {
        "num_tokens": int(flat.numel()),
        "unique_codes": int(unique.numel()),
        "unique_code_ratio": float(unique.numel() / min(vocab_size, flat.numel())),
        "code_usage_entropy": _as_float(entropy),
        "code_usage_perplexity": float(math.exp(_as_float(entropy))),
        "top1_code_fraction": _as_float(probs.max()),
    }
