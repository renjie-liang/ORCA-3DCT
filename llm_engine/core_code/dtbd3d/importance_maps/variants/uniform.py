"""Uniform reconstruction weights."""

from __future__ import annotations

import numpy as np

from dtbd3d.importance_maps.builder import AnatomicalImportanceMapBuilder


def build_uniform(
    ontology,
    target_shape: tuple[int, int, int],
    volume_id: str,
    ts_mask: np.ndarray | None,
    **unused,
) -> np.ndarray:
    return np.ones(target_shape, dtype=np.float32)


AnatomicalImportanceMapBuilder.register_variant("uniform", build_uniform)

