from __future__ import annotations

from pathlib import Path

import numpy as np

from dtbd3d.importance_maps import load_ontology
from dtbd3d.importance_maps.build_unified_ts_artifacts import (
    default_group_specs,
    group_bitmask_from_basis,
    shard_ids,
)


ROOT = Path(__file__).resolve().parents[5]
ONTOLOGY_TS_V2 = ROOT / "Experiment" / "configs" / "clinical_ontology_ts_v2.yaml"


def test_default_group_specs_resolve_ts_ids() -> None:
    ontology = load_ontology(ONTOLOGY_TS_V2)
    specs = default_group_specs(ontology)
    names = [str(spec["name"]) for spec in specs]
    assert names == ["lung", "cardiomediastinal", "hiatal", "airway", "pleura_proxy"]
    lung = specs[0]
    assert lung["ts_class_ids"] == [10, 11, 12, 13, 14]
    assert lung["bit"] == 0


def test_group_bitmask_from_basis_allows_overlapping_groups() -> None:
    class_ids = [10, 16, 51]
    basis = np.zeros((3, 2, 2, 2), dtype=np.float32)
    basis[0, 0, 0, 0] = 1.0
    basis[1, 0, 0, 0] = 1.0
    basis[2, 1, 1, 1] = 1.0
    specs = [
        {"bit": 0, "name": "lung", "ts_class_ids": [10]},
        {"bit": 1, "name": "airway", "ts_class_ids": [16]},
        {"bit": 2, "name": "heart", "ts_class_ids": [51]},
    ]
    bitmask = group_bitmask_from_basis(basis, class_ids, specs, min_fraction=0.0)
    assert bitmask.dtype == np.uint8
    assert int(bitmask[0, 0, 0]) == 0b00000011
    assert int(bitmask[1, 1, 1]) == 0b00000100
    assert int(bitmask[0, 1, 1]) == 0


def test_shard_ids_round_robin_and_limit() -> None:
    ids = [f"id_{index}" for index in range(10)]
    assert shard_ids(ids, num_shards=3, shard_id=0, limit=0) == ["id_0", "id_3", "id_6", "id_9"]
    assert shard_ids(ids, num_shards=3, shard_id=1, limit=2) == ["id_1", "id_4"]
