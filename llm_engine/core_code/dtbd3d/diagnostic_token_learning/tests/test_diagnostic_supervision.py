from __future__ import annotations

import json

import numpy as np
import torch
import yaml

from dtbd3d.diagnostic_token_learning.build_llama_text_embedding_artifacts import (
    build_ctrate_organ_disease_rows,
    build_radgenome_region_abnormality_rows,
    dedupe_rows,
)
from dtbd3d.diagnostic_token_learning.config import (
    DiseaseClassificationConfig,
    DiagnosticSupervisionConfig,
    OrganInputConfig,
    TextAlignmentConfig,
    TextEmbeddingSourceConfig,
)
from dtbd3d.diagnostic_token_learning.llama_space_alignment import merge_tokens_spatial_2x2
from dtbd3d.diagnostic_token_learning.organ_input import apply_organ_input_conditioning
from dtbd3d.diagnostic_token_learning.supervision import DiagnosticSupervision
from dtbd3d.diagnostic_token_learning.volume_id import normalize_volume_id


def test_normalize_volume_id() -> None:
    assert normalize_volume_id("train_1_a_1.nii.gz") == "train_1_a_1"
    assert normalize_volume_id("img_valid_2_b_1.nii.gz") == "valid_2_b_1"
    assert normalize_volume_id("seg_valid_2_b_1.npy") == "valid_2_b_1"


def test_organ_input_importance_residual() -> None:
    x = torch.ones(1, 1, 2, 2, 2, dtype=torch.bfloat16)
    weights = torch.full_like(x, 2.0)
    out = apply_organ_input_conditioning(x, weights, OrganInputConfig(enabled=True, strength=0.25))
    assert out.shape == x.shape
    assert torch.allclose(out.float(), torch.full_like(out.float(), 1.25))


def test_llama_text_embedding_index_rows(tmp_path) -> None:
    supervision = tmp_path / "diagnostic_supervision"
    radg_dir = supervision / "radgenome_region_abnormality" / "valid"
    radg_dir.mkdir(parents=True)
    radg_row = {
        "volume_id": "valid_1_a_1",
        "organ_group": "lung",
        "mask_name": "lung",
        "anatomy": "lung",
        "mapping_status": "exact",
        "text": "small nodule",
    }
    radg_dir.joinpath("region_abnormality.jsonl").write_text(
        json.dumps(radg_row) + "\n" + json.dumps(radg_row) + "\n"
    )
    radg_rows = dedupe_rows(build_radgenome_region_abnormality_rows(supervision, "valid"))
    assert len(radg_rows) == 1
    assert radg_rows[0]["target_text"] == "Region: lung. Finding: small nodule."
    assert radg_rows[0]["duplicate_count"] == 2

    label_dir = supervision / "ctrate_disease_labels"
    valid_dir = label_dir / "valid"
    valid_dir.mkdir(parents=True)
    valid_dir.joinpath("ids.txt").write_text("valid_1_a_1\nvalid_2_a_1\n")
    label_dir.joinpath("label_names.json").write_text(json.dumps(["Lung nodule", "Medical material"]))
    np.save(valid_dir / "labels.npy", np.asarray([[1, 1], [0, 1]], dtype=np.int8))
    label_dir.joinpath("disease_to_organ_groups.yaml").write_text(
        yaml.safe_dump({"groups": {"lung": ["Lung nodule"], "global": ["Medical material"]}})
    )
    disease_rows = build_ctrate_organ_disease_rows(supervision, "valid", include_global_disease=False)
    assert len(disease_rows) == 1
    assert disease_rows[0]["organ_group"] == "lung"
    assert disease_rows[0]["target_text"] == "Region: lung. Finding: Lung nodule."


def test_llama_space_text_alignment_forward_backward(tmp_path) -> None:
    artifact_dir = tmp_path / "embeddings" / "radgenome_region_abnormality" / "train"
    artifact_dir.mkdir(parents=True)
    artifact_dir.joinpath("index.jsonl").write_text(
        json.dumps(
            {
                "row_id": 0,
                "embedding_row": 0,
                "volume_id": "train_1_a_1",
                "organ_group": "lung",
                "mask_name": "lung",
                "text_hash": "abc",
                "target_text": "Region: lung. Finding: nodule.",
            }
        )
        + "\n"
    )
    np.save(artifact_dir / "embeddings.npy", np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float16))
    np.save(artifact_dir / "encoded_mask.npy", np.asarray([True], dtype=bool))

    mask_dir = tmp_path / "masks" / "train"
    mask_dir.mkdir(parents=True)
    mask = np.ones((1, 2, 64, 64), dtype=np.float16)
    np.savez(
        mask_dir / "train_1_a_1.npz",
        mask_token=mask,
        mask_ids=np.asarray([0], dtype=np.int32),
        mask_names=np.asarray(["lung"]),
        mask_groups=np.asarray(["lung"]),
        mask_sources=np.asarray(["test"]),
    )

    cfg = DiagnosticSupervisionConfig(
        text_alignment=TextAlignmentConfig(
            enabled=True,
            weight=0.25,
            backend="llama_hidden_artifact",
            mode="cosine",
            hidden_dim=4,
            reportgen_grid_dhw=(2, 2, 2),
            embedding_sources=[
                TextEmbeddingSourceConfig(
                    name="radgenome_region_abnormality",
                    artifact_dir=str(tmp_path / "embeddings" / "radgenome_region_abnormality"),
                    mask_artifact_dir=str(tmp_path / "masks"),
                    split="train",
                    max_pairs_per_volume=2,
                )
            ],
        )
    )
    module = DiagnosticSupervision(cfg, visual_dim=2)
    z = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
    merged = merge_tokens_spatial_2x2(z, (2, 2, 2))
    assert merged.shape == (1, 2, 2, 2, 2)
    out = module(
        z_quantized=z,
        volume_ids=["train_1_a_1"],
        importance_weights=None,
        full_shape_dhw=(4, 4, 4),
        crop_depth_value=0,
        step=1,
    )
    assert out.loss.ndim == 0
    assert out.metrics["text_pairs"] == 1
    out.loss.backward()
    assert z.grad is not None


def test_llama_space_text_alignment_uses_group_mask(tmp_path) -> None:
    artifact_dir = tmp_path / "embeddings" / "radgenome_region_abnormality" / "train"
    artifact_dir.mkdir(parents=True)
    artifact_dir.joinpath("index.jsonl").write_text(
        json.dumps(
            {
                "row_id": 0,
                "embedding_row": 0,
                "volume_id": "train_1_a_1",
                "organ_group": "aorta",
                "mask_name": "heart ascending aorta",
                "text_hash": "abc",
                "target_text": "Region: aorta. Finding: calcification.",
            }
        )
        + "\n"
    )
    np.save(artifact_dir / "embeddings.npy", np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float16))
    np.save(artifact_dir / "encoded_mask.npy", np.asarray([True], dtype=bool))

    mask_dir = tmp_path / "radgenome_masks" / "train"
    mask_dir.mkdir(parents=True)
    np.savez(
        mask_dir / "train_1_a_1.npz",
        mask_token=np.ones((1, 2, 64, 64), dtype=np.float16),
        mask_ids=np.asarray([0], dtype=np.int32),
        mask_names=np.asarray(["lung"]),
        mask_groups=np.asarray(["aorta"]),
        mask_sources=np.asarray(["test"]),
    )

    cfg = DiagnosticSupervisionConfig(
        text_alignment=TextAlignmentConfig(
            enabled=True,
            weight=0.25,
            backend="llama_hidden_artifact",
            mode="cosine",
            hidden_dim=4,
            reportgen_grid_dhw=(2, 2, 2),
            embedding_sources=[
                TextEmbeddingSourceConfig(
                    name="radgenome_region_abnormality",
                    artifact_dir=str(tmp_path / "embeddings" / "radgenome_region_abnormality"),
                    mask_artifact_dir=str(tmp_path / "radgenome_masks"),
                    split="train",
                    max_pairs_per_volume=2,
                )
            ],
        )
    )
    module = DiagnosticSupervision(cfg, visual_dim=2)
    z = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
    out = module(
        z_quantized=z,
        volume_ids=["train_1_a_1"],
        importance_weights=None,
        full_shape_dhw=(4, 4, 4),
        crop_depth_value=0,
        step=1,
    )
    assert out.metrics["text_pairs"] == 1
    out.loss.backward()
    assert z.grad is not None


def test_disease_classification_forward_backward(tmp_path) -> None:
    artifact_root = tmp_path / "diagnostic_supervision" / "ctrate_organ_disease_classification"
    split_dir = artifact_root / "train"
    split_dir.mkdir(parents=True)
    split_dir.joinpath("ids.txt").write_text("train_1_a_1\n")
    artifact_root.joinpath("organ_groups.json").write_text(json.dumps(["lung", "pleura"]))
    artifact_root.joinpath("disease_names.json").write_text(json.dumps(["Nodule", "Pleural effusion"]))
    artifact_root.joinpath("group_to_mask_name.json").write_text(json.dumps({"lung": "lung", "pleura": "pleura_proxy"}))
    np.save(split_dir / "labels.npy", np.asarray([[[1, 0], [0, 1]]], dtype=np.int8))
    np.save(split_dir / "valid_mask.npy", np.asarray([[True, False], [False, True]], dtype=bool))

    mask_dir = tmp_path / "ts_masks" / "train"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((2, 2, 64, 64), dtype=np.float16)
    mask[0, :, :, :32] = 1
    mask[1, :, :, 32:] = 1
    np.savez(
        mask_dir / "train_1_a_1.npz",
        mask_token=mask,
        mask_ids=np.asarray([0, 1], dtype=np.int32),
        mask_names=np.asarray(["lung", "pleura_proxy"]),
    )

    cfg = DiagnosticSupervisionConfig(
        disease_classification=DiseaseClassificationConfig(
            enabled=True,
            weight=0.5,
            artifact_dir=str(artifact_root),
            mask_artifact_dir=str(tmp_path / "ts_masks"),
            split="train",
            feature_grid_dhw=(2, 2, 2),
            organ_embedding=True,
            hidden_dim=4,
        )
    )
    module = DiagnosticSupervision(cfg, visual_dim=3)
    z = torch.randn(1, 3, 2, 4, 4, requires_grad=True)
    out = module(
        z_quantized=z,
        volume_ids=["train_1_a_1"],
        importance_weights=None,
        full_shape_dhw=(4, 8, 8),
        crop_depth_value=0,
        step=1,
    )
    assert out.loss.ndim == 0
    assert out.metrics["disease_pairs"] == 2
    assert out.metrics["disease_organs"] == 2
    assert out.metrics["disease_pos"] == 2
    assert out.metrics["disease_feature_grid"] == "2x2x2"
    out.loss.backward()
    assert z.grad is not None
