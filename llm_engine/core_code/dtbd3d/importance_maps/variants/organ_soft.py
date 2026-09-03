"""V3 organ-soft anatomical importance map.

For the first Sub-task 5 implementation, each CT-RATE finding contributes a
soft organ prior derived from TotalSegmentator classes. If multiple findings
map to the same organ region, their contributions add before token-grid
downsampling. The shared builder later normalizes and applies the uniform floor.
"""

from __future__ import annotations

import numpy as np

from dtbd3d.importance_maps.builder import AnatomicalImportanceMapBuilder
from dtbd3d.importance_maps.downsample import downsample_to_token_grid
from dtbd3d.importance_maps.ontology import ClinicalOntology


def build_organ_soft(
    ontology: ClinicalOntology,
    target_shape: tuple[int, int, int],
    volume_id: str,
    ts_mask: np.ndarray | None,
    **unused,
) -> np.ndarray:
    """Return the unnormalized V3 token-grid map for one TS mask."""
    if ts_mask is None:
        raise KeyError(f"organ_soft requires ts_mask for volume {volume_id}")
    tilde_w = np.zeros(ts_mask.shape, dtype=np.float32)
    for finding in ontology.findings:
        ids = list(finding.ts_class_ids)
        if not ids:
            continue
        organ_ids = np.asarray(ids, dtype=ts_mask.dtype)
        tilde_w += np.isin(ts_mask, organ_ids).astype(np.float32)
    if float(tilde_w.sum()) <= 0.0:
        raise ValueError(f"organ_soft: no relevant TS organs in volume {volume_id}")
    return downsample_to_token_grid(tilde_w, target_shape=target_shape, method="mean")


AnatomicalImportanceMapBuilder.register_variant("organ_soft", build_organ_soft)
