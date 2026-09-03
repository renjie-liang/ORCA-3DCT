"""LFQ token packing, unpacking, and report-generation layout transforms."""

from __future__ import annotations

from typing import Literal

import numpy as np
from einops import rearrange


CODEBOOK_DIM = 18
LFQConvention = Literal[
    "msb_identity",
    "msb_reverse_channels",
    "lsb_identity",
    "lsb_reverse_channels",
]


def _validate_convention(convention: str) -> None:
    if convention not in ("msb_identity", "msb_reverse_channels", "lsb_identity", "lsb_reverse_channels"):
        raise ValueError(f"unsupported LFQ convention: {convention}")


def unpack_lfq_codes(
    codes: np.ndarray,
    convention: LFQConvention = "msb_reverse_channels",
    num_bits: int = CODEBOOK_DIM,
) -> np.ndarray:
    """Unpack packed LFQ integer codes into {-1, +1} vectors.

    `msb_identity` is the decoder/reconstruction convention. The 16x16x8
    report-generation adapter currently performs best with
    `msb_reverse_channels`.
    """

    if codes.dtype != np.uint32:
        raise TypeError(f"codes must be uint32, got {codes.dtype}")
    if num_bits <= 0:
        raise ValueError(f"num_bits must be positive, got {num_bits}")
    _validate_convention(convention)

    if convention.startswith("msb"):
        bit_powers = 1 << np.arange(num_bits - 1, -1, -1, dtype=np.uint32)
    else:
        bit_powers = 1 << np.arange(num_bits, dtype=np.uint32)

    bits = ((codes[..., None] & bit_powers) > 0).astype(np.float32)
    if convention.endswith("reverse_channels"):
        bits = bits[..., ::-1]
    return (bits * 2.0 - 1.0).astype(np.float32)


def pack_lfq_bits(bits_or_pm1: np.ndarray, convention: LFQConvention = "msb_identity") -> np.ndarray:
    """Pack LFQ bit vectors or {-1,+1} vectors into uint32 codes.

    The input's last dimension is interpreted according to `convention`.
    Values greater than zero are treated as bit 1.
    """

    _validate_convention(convention)
    if bits_or_pm1.shape[-1] <= 0:
        raise ValueError("last dimension must contain LFQ bits")
    num_bits = bits_or_pm1.shape[-1]

    bits = (bits_or_pm1 > 0).astype(np.uint32)
    if convention.endswith("reverse_channels"):
        bits = bits[..., ::-1]
    if convention.startswith("msb"):
        bit_powers = 1 << np.arange(num_bits - 1, -1, -1, dtype=np.uint32)
    else:
        bit_powers = 1 << np.arange(num_bits, dtype=np.uint32)
    return (bits * bit_powers).sum(axis=-1).astype(np.uint32)


def merge_8x8_reportgen(arr: np.ndarray) -> np.ndarray:
    """Merge canonical 8x8 tensor `(B,18,T,64,64)` to reportgen `(B,72,T,32,32)`."""

    if arr.ndim != 5:
        raise ValueError(f"expected 5D tensor, got shape {arr.shape}")
    if arr.shape[1] != CODEBOOK_DIM:
        raise ValueError(f"expected channel dim {CODEBOOK_DIM}, got {arr.shape[1]}")
    if arr.shape[3] % 2 or arr.shape[4] % 2:
        raise ValueError(f"spatial dimensions must be divisible by 2, got {arr.shape[3:]}")
    return rearrange(arr, "b c t (h p1) (w p2) -> b (c p1 p2) t h w", p1=2, p2=2)


def unmerge_8x8_reportgen(arr: np.ndarray) -> np.ndarray:
    """Invert reportgen 8x8 merge: `(B,72,T,32,32)` -> `(B,18,T,64,64)`."""

    if arr.ndim != 5:
        raise ValueError(f"expected 5D tensor, got shape {arr.shape}")
    expected_channels = CODEBOOK_DIM * 4
    if arr.shape[1] != expected_channels:
        raise ValueError(f"expected channel dim {expected_channels}, got {arr.shape[1]}")
    return rearrange(arr, "b (c p1 p2) t h w -> b c t (h p1) (w p2)", c=CODEBOOK_DIM, p1=2, p2=2)
