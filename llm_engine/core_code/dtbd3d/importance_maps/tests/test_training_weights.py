from __future__ import annotations

import numpy as np
import torch

from dtbd3d.importance_maps.training import apply_uniform_mix, build_importance_weight_batch, weighted_l1_loss


def test_apply_uniform_mix_matches_offline_formula() -> None:
    base = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    mixed = apply_uniform_mix(base, lambda_uniform=0.3)
    expected = 0.7 * base + 0.3
    assert np.allclose(mixed, expected)


def test_importance_weight_batch_is_crop_normalized(tmp_path) -> None:
    weight = np.ones((3, 4, 4), dtype=np.float32)
    weight[:, :, :2] = 3.0
    np.save(tmp_path / "vol_a.npy", weight)
    batch = build_importance_weight_batch(
        ["vol_a"],
        tmp_path,
        full_shape_dhw=(9, 16, 16),
        crop_depth_value=5,
        step=1,
        device="cpu",
        dtype=torch.float32,
        lambda_uniform=0.0,
    )
    assert batch.shape == (1, 1, 5, 16, 16)
    assert abs(float(batch.mean()) - 1.0) < 1e-6
    assert float(batch.max()) > 1.0


def test_importance_weight_batch_can_mix_base_map(tmp_path) -> None:
    weight = np.zeros((3, 4, 4), dtype=np.float32)
    weight[:, :, :2] = 2.0
    np.save(tmp_path / "vol_a.npy", weight)
    batch = build_importance_weight_batch(
        ["vol_a"],
        tmp_path,
        full_shape_dhw=(9, 16, 16),
        crop_depth_value=0,
        step=1,
        device="cpu",
        dtype=torch.float32,
        lambda_uniform=1.0,
    )
    assert torch.allclose(batch, torch.ones_like(batch))


def test_uniform_weighted_l1_matches_l1() -> None:
    input_data = torch.zeros((1, 1, 3, 4, 4), dtype=torch.float32)
    recon_output = torch.ones_like(input_data)
    weights = torch.ones_like(input_data)
    assert torch.allclose(weighted_l1_loss(input_data, recon_output, weights), torch.tensor(1.0))


def test_nonuniform_weight_changes_loss() -> None:
    input_data = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    recon_output = torch.zeros_like(input_data)
    recon_output[:, :, :, :, 0] = 2.0
    weights = torch.ones_like(input_data)
    weights[:, :, :, :, 0] = 3.0
    weights = weights / weights.mean()
    uniform = torch.abs(input_data - recon_output).mean()
    weighted = weighted_l1_loss(input_data, recon_output, weights)
    assert float(weighted) > float(uniform)
