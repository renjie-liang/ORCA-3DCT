"""Pluggable anatomical importance-map builder.

This module owns the invariant shared by all Sub-task 5 variants:

1. variant code returns an unnormalized non-negative token-grid map;
2. the builder sum-normalizes it so the average token weight is 1;
3. `lambda_uniform` mixes the normalized map with a uniform floor.

Training code should consume the saved maps as data. It should not duplicate
these normalization rules.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dtbd3d.importance_maps.ontology import ClinicalOntology


TOKEN_SHAPES = {
    "16x16x8": (31, 32, 32),
    "8x8x8": (31, 64, 64),
}

VariantFn = Callable[..., np.ndarray]


def _sum_normalize(weight: np.ndarray) -> np.ndarray:
    """Scale a map so `mean(weight) == 1`."""
    out = np.asarray(weight, dtype=np.float32)
    total = float(out.sum())
    if total <= 0.0:
        raise ValueError("importance map sum must be positive before normalization")
    return out * (out.size / total)


def _apply_uniform_floor(weight: np.ndarray, lambda_uniform: float) -> np.ndarray:
    """Mix the anatomical map with a uniform map.

    `lambda_uniform=0` keeps the anatomical weights.
    `lambda_uniform=1` recovers the uniform baseline.
    """
    if lambda_uniform == 0.0:
        return weight
    if lambda_uniform == 1.0:
        return np.ones_like(weight, dtype=np.float32)
    return ((1.0 - lambda_uniform) * weight + lambda_uniform).astype(np.float32, copy=False)


class AnatomicalImportanceMapBuilder:
    """Build one saved per-token weight map for a CT-RATE volume."""
    _variant_registry: dict[str, VariantFn] = {}

    @classmethod
    def register_variant(cls, name: str, fn: VariantFn) -> None:
        if name in cls._variant_registry:
            raise KeyError(f"Variant already registered: {name}")
        cls._variant_registry[name] = fn

    def __init__(
        self,
        ontology: ClinicalOntology,
        variant: str,
        compression: str,
        lambda_uniform: float,
    ) -> None:
        if compression not in TOKEN_SHAPES:
            raise ValueError(f"Unsupported compression: {compression}")
        if not 0.0 <= lambda_uniform <= 1.0:
            raise ValueError(f"lambda_uniform must be in [0,1], got {lambda_uniform}")
        if variant not in self._variant_registry:
            raise KeyError(f"Unknown importance-map variant: {variant}")
        self.ontology = ontology
        self.variant = variant
        self.compression = compression
        self.lambda_uniform = float(lambda_uniform)
        self.target_shape = TOKEN_SHAPES[compression]
        self._build_fn = self._variant_registry[variant]

    def build(
        self,
        volume_id: str,
        ts_mask: np.ndarray | None,
        **kwargs,
    ) -> np.ndarray:
        if self.lambda_uniform == 1.0:
            return np.ones(self.target_shape, dtype=np.float32)
        raw = self._build_fn(
            ontology=self.ontology,
            target_shape=self.target_shape,
            volume_id=volume_id,
            ts_mask=ts_mask,
            **kwargs,
        )
        return self.finalize_raw(raw)

    def finalize_raw(self, raw: np.ndarray) -> np.ndarray:
        """Normalize and mix a precomputed raw token-grid map."""
        if raw.shape != self.target_shape:
            raise ValueError(f"{self.variant} returned {raw.shape}, expected {self.target_shape}")
        normalized = _sum_normalize(raw)
        return _apply_uniform_floor(normalized, self.lambda_uniform)
