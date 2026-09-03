from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import save_file

from dtbd3d.core.token_codec import unpack_lfq_codes
from dtbd3d.learned_vq.codebook_artifact import build_codebook_array, export_codebook_artifact


def _write_delta_checkpoint(path):
    vocab_size = 2**18
    ids = torch.arange(vocab_size, dtype=torch.int64)
    bit_powers = 1 << torch.arange(17, -1, -1, dtype=torch.int64)
    bits = ((ids[:, None] & bit_powers[None, :]) != 0).float()
    base_code = bits * 2.0 - 1.0
    delta = torch.zeros_like(base_code)
    save_file({"base_code": base_code, "delta": delta}, path, metadata={"delta_scale": "0.1"})


def test_lfq_binary_msb_reverse_matches_token_codec(tmp_path):
    checkpoint = tmp_path / "delta_table.safetensors"
    _write_delta_checkpoint(checkpoint)
    tokens = np.array([0, 1, 2, 3, 17, 255, 1024, 262143], dtype=np.uint32)

    codebook, delta_scale, _metadata = build_codebook_array(
        checkpoint,
        codebook_mode="lfq_binary",
        dtype="float32",
        codebook_channel_order="msb_reverse_channels",
    )

    np.testing.assert_array_equal(codebook[tokens], unpack_lfq_codes(tokens, convention="msb_reverse_channels"))
    assert delta_scale == 0.0


def test_lfq_binary_msb_identity_matches_reconstruction_order(tmp_path):
    checkpoint = tmp_path / "delta_table.safetensors"
    _write_delta_checkpoint(checkpoint)
    tokens = np.array([0, 1, 2, 3, 17, 255, 1024, 262143], dtype=np.uint32)

    codebook, _delta_scale, _metadata = build_codebook_array(
        checkpoint,
        codebook_mode="lfq_binary",
        dtype="float32",
        codebook_channel_order="msb_identity",
    )

    np.testing.assert_array_equal(codebook[tokens], unpack_lfq_codes(tokens, convention="msb_identity"))


def test_learned_codebook_preserves_shape_and_metadata_contract(tmp_path):
    checkpoint = tmp_path / "delta_table.safetensors"
    _write_delta_checkpoint(checkpoint)
    out_dir = tmp_path / "artifact"

    metadata = export_codebook_artifact(
        checkpoint,
        out_dir,
        codebook_mode="learned_codebook",
        dtype="float16",
        codebook_channel_order="msb_reverse_channels",
    )
    codebook = np.load(out_dir / "codebook.npy", mmap_mode="r")

    assert codebook.shape == (2**18, 18)
    assert codebook.dtype == np.float16
    assert metadata["codebook_channel_order"] == "msb_reverse_channels"
    assert metadata["reportgen_expected_channel_order"] == "msb_reverse_channels"
    assert metadata["requires_channel_reverse_for_btb3d_pretrained_reportgen"] is False
