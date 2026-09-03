from __future__ import annotations

from pathlib import Path

import numpy as np

from dtbd3d.importance_maps import AnatomicalImportanceMapBuilder, load_ontology
from dtbd3d.importance_maps.downsample import downsample_to_token_grid


ROOT = Path(__file__).resolve().parents[5]
ONTOLOGY = ROOT / "Experiment" / "configs" / "clinical_ontology.yaml"
ONTOLOGY_TS_V2 = ROOT / "Experiment" / "configs" / "clinical_ontology_ts_v2.yaml"


def test_ontology_loads_canonical_findings() -> None:
    ontology = load_ontology(ONTOLOGY)
    assert len(ontology.findings) == 18
    assert ontology.finding_to_ts_ids("Lung nodule") == {10, 11, 12, 13, 14}


def test_ts_v2_ontology_loads_nested_aliases() -> None:
    ontology = load_ontology(ONTOLOGY_TS_V2)
    assert len(ontology.findings) == 18
    assert ontology.finding_to_ts_ids("Lung nodule") == {10, 11, 12, 13, 14}
    assert ontology.finding_to_ts_ids("Hiatal hernia") == {6, 15}
    assert ontology.finding_to_ts_ids("Medical material") == set()


def test_uniform_shapes_for_both_compressions() -> None:
    ontology = load_ontology(ONTOLOGY)
    b16 = AnatomicalImportanceMapBuilder(ontology, "uniform", "16x16x8", lambda_uniform=0.0)
    b8 = AnatomicalImportanceMapBuilder(ontology, "uniform", "8x8x8", lambda_uniform=0.0)
    assert b16.build("synthetic", ts_mask=None).shape == (31, 32, 32)
    assert b8.build("synthetic", ts_mask=None).shape == (31, 64, 64)


def test_downsample_mean_preserves_ones() -> None:
    field = np.ones((31, 32, 32), dtype=np.float32)
    out = downsample_to_token_grid(field, target_shape=(31, 32, 32), method="mean")
    assert np.allclose(out, 1.0)


def test_organ_soft_lung_exceeds_aorta() -> None:
    ontology = load_ontology(ONTOLOGY)
    lung_id = ontology.ts_class_id["lung_upper_lobe_left"]
    aorta_id = ontology.ts_class_id["aorta"]
    mask = np.full((31, 32, 32), lung_id, dtype=np.int16)
    mask[:, 16:, :] = aorta_id
    builder = AnatomicalImportanceMapBuilder(ontology, "organ_soft", "16x16x8", lambda_uniform=0.0)
    weight = builder.build("synthetic_split", ts_mask=mask)
    assert weight.shape == (31, 32, 32)
    assert abs(float(weight.mean()) - 1.0) < 1e-6
    assert float(weight[:, :16, :].mean()) > float(weight[:, 16:, :].mean())


def test_lambda_one_collapses_to_uniform() -> None:
    ontology = load_ontology(ONTOLOGY)
    lung_id = ontology.ts_class_id["lung_upper_lobe_left"]
    aorta_id = ontology.ts_class_id["aorta"]
    mask = np.full((31, 32, 32), lung_id, dtype=np.int16)
    mask[:, 16:, :] = aorta_id
    builder = AnatomicalImportanceMapBuilder(ontology, "organ_soft", "8x8x8", lambda_uniform=1.0)
    weight = builder.build("synthetic_split", ts_mask=mask)
    assert weight.shape == (31, 64, 64)
    assert np.allclose(weight, 1.0)
