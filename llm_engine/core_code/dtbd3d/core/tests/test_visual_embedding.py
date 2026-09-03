from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dtbd3d.core.visual_embedding import (
    REPORTGEN_CHANNEL_ORDER,
    load_reportgen_codebook,
    materialize_reportgen_features_from_codes,
    resolve_reportgen_artifact_manifest,
)


def write_reportgen_metadata(path: Path, channel_order: str = REPORTGEN_CHANNEL_ORDER) -> None:
    path.write_text(json.dumps({
        "codebook_mode": "learned_codebook",
        "codebook_channel_order": channel_order,
        "reportgen_expected_channel_order": REPORTGEN_CHANNEL_ORDER,
        "requires_channel_reverse_for_btb3d_pretrained_reportgen": False,
        "lookup_contract": "token_id -> codebook[token_id] -> visual feature",
    }))


def test_resolve_reportgen_artifact_manifest(tmp_path: Path):
    reportgen_dir = tmp_path / "reportgen_artifact"
    token_dir = reportgen_dir / "token_artifact" / "valid"
    codebook_dir = reportgen_dir / "codebook_artifact"
    token_dir.mkdir(parents=True)
    codebook_dir.mkdir()
    (token_dir / "ids.txt").write_text("valid_1_a_1\n")
    np.save(token_dir / "tokens_int.npy", np.zeros((1, 31 * 32 * 32), dtype=np.uint32))
    np.save(codebook_dir / "codebook.npy", np.zeros((1 << 18, 18), dtype=np.float16))
    write_reportgen_metadata(codebook_dir / "metadata.json")
    manifest_path = reportgen_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "codebook_artifact_dir": "codebook_artifact",
        "token_artifacts": {
            "valid": {
                "dir": "token_artifact/valid",
                "ids": "token_artifact/valid/ids.txt",
                "tokens_int": "token_artifact/valid/tokens_int.npy",
            }
        },
    }))

    artifact = resolve_reportgen_artifact_manifest(manifest_path, "valid")

    assert artifact.token_dir == token_dir
    assert artifact.ids_path == token_dir / "ids.txt"
    assert artifact.tokens_int_path == token_dir / "tokens_int.npy"
    assert artifact.codebook_path == codebook_dir / "codebook.npy"
    assert artifact.codebook_metadata_path == codebook_dir / "metadata.json"


def test_load_reportgen_codebook_rejects_identity_order(tmp_path: Path):
    codebook_path = tmp_path / "codebook.npy"
    metadata_path = tmp_path / "metadata.json"
    np.save(codebook_path, np.zeros((1 << 18, 18), dtype=np.float16))
    write_reportgen_metadata(metadata_path, channel_order="msb_identity")

    with pytest.raises(ValueError, match="codebook_channel_order"):
        load_reportgen_codebook(codebook_path, metadata_path)


def test_materialize_reportgen_features_from_codebook():
    codebook = np.zeros((1 << 18, 18), dtype=np.float32)
    codebook[7] = np.arange(18, dtype=np.float32)
    codes = np.full((31 * 32 * 32,), 7, dtype=np.uint32)

    features = materialize_reportgen_features_from_codes(codes, "16x16x8", codebook=codebook)

    assert features.shape == (1, 31, 32, 32, 18)
    np.testing.assert_array_equal(features[0, 0, 0, 0], np.arange(18, dtype=np.float32))
