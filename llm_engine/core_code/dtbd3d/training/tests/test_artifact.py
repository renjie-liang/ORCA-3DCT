from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dtbd3d.core.artifact import (
    load_token_row,
    open_matrix,
    read_ids,
    save_token_row,
    token_path,
    update_ids,
    write_tokens_matrix,
)


def test_canonical_artifact_roundtrip(tmp_path):
    tokens_a = np.array([1, 2, 3], dtype=np.uint32)
    tokens_b = np.array([4, 5, 6], dtype=np.uint32)

    save_token_row(tmp_path, "a", tokens_a)
    save_token_row(tmp_path, "b", tokens_b)
    update_ids(tmp_path / "ids.txt", ["a", "b", "a"])

    assert token_path(tmp_path, "a").exists()
    assert read_ids(tmp_path / "ids.txt") == ["a", "b"]
    assert np.array_equal(load_token_row(tmp_path, "a"), tokens_a)

    matrix_path = write_tokens_matrix(tmp_path, expected_tokens=3)
    assert matrix_path.exists()

    id_to_idx, matrix = open_matrix(tmp_path)
    assert id_to_idx == {"a": 0, "b": 1}
    assert matrix.shape == (2, 3)

    # Force matrix fallback by deleting one per-volume file.
    token_path(tmp_path, "b").unlink()
    assert np.array_equal(load_token_row(tmp_path, "b", id_to_idx, matrix), tokens_b)
