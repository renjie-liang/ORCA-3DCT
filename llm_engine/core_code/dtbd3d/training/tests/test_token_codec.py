from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dtbd3d.core.token_codec import (
    merge_8x8_reportgen,
    pack_lfq_bits,
    unmerge_8x8_reportgen,
    unpack_lfq_codes,
)


def test_msb_identity_unpack_known_values():
    codes = np.array([0, 1, 262143], dtype=np.uint32)
    pm1 = unpack_lfq_codes(codes, convention="msb_identity")
    assert pm1.shape == (3, 18)
    assert np.all(pm1[0] == -1.0)
    assert pm1[1, -1] == 1.0
    assert np.all(pm1[2] == 1.0)


def test_msb_reverse_channels_matches_lsb_identity():
    codes = np.array([0, 1, 7, 262143], dtype=np.uint32)
    assert np.array_equal(
        unpack_lfq_codes(codes, convention="msb_reverse_channels"),
        unpack_lfq_codes(codes, convention="lsb_identity"),
    )


def test_pack_unpack_roundtrip_msb_identity():
    codes = np.array([0, 1, 7, 12345, 262143], dtype=np.uint32)
    pm1 = unpack_lfq_codes(codes, convention="msb_identity")
    packed = pack_lfq_bits(pm1, convention="msb_identity")
    assert np.array_equal(packed, codes)


def test_8x8_reportgen_merge_inverse_exact():
    rng = np.random.default_rng(0)
    canonical = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(1, 18, 31, 64, 64),
    )
    merged = merge_8x8_reportgen(canonical)
    restored = unmerge_8x8_reportgen(merged)
    assert merged.shape == (1, 72, 31, 32, 32)
    assert restored.shape == canonical.shape
    assert np.array_equal(restored, canonical)
