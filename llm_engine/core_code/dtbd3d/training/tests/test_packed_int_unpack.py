from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dtbd3d.core.token_codec import unpack_lfq_codes


def test_unpack_shape():
    codes = np.array([[0, 7, 262143]], dtype=np.uint32)
    pm1 = unpack_lfq_codes(codes)
    assert pm1.shape == (1, 3, 18)
    assert set(np.unique(pm1).tolist()).issubset({-1.0, 1.0})


def test_unpack_known_values():
    codes = np.array([[0, 262143]], dtype=np.uint32)
    pm1 = unpack_lfq_codes(codes)
    assert np.all(pm1[0, 0] == -1.0)
    assert np.all(pm1[0, 1] == 1.0)


def test_unpack_conventions_are_equivalent_in_expected_pairs():
    codes = np.array([[0, 1, 7, 262143]], dtype=np.uint32)
    assert np.array_equal(
        unpack_lfq_codes(codes, convention="msb_reverse_channels"),
        unpack_lfq_codes(codes, convention="lsb_identity"),
    )
    assert np.array_equal(
        unpack_lfq_codes(codes, convention="msb_identity"),
        unpack_lfq_codes(codes, convention="lsb_reverse_channels"),
    )


def test_unpack_msb_identity_vs_reportgen_default_distinguishable():
    codes = np.array([[1]], dtype=np.uint32)
    reportgen = unpack_lfq_codes(codes)
    recon = unpack_lfq_codes(codes, convention="msb_identity")
    assert reportgen[0, 0, 0] == 1.0
    assert recon[0, 0, -1] == 1.0
    assert not np.array_equal(reportgen, recon)
