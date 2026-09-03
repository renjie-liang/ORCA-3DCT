"""Organ-conditioned CT-RATE disease classification supervision."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dtbd3d.diagnostic_token_learning.config import DiseaseClassificationConfig
from dtbd3d.diagnostic_token_learning.llama_space_alignment import downsample_mask_64_to_grid
from dtbd3d.diagnostic_token_learning.volume_id import normalize_volume_id


SMOKE_DEBUG_PRINTS_HARDCODED = True
SMOKE_DEBUG_BREAKPOINT_HARDCODED = True
SMOKE_DEBUG_BREAKPOINT_STEP = 1


def _rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _breakpoints_enabled() -> bool:
    return "PYTEST_CURRENT_TEST" not in os.environ


class _DiseaseArtifact:
    def __init__(self, artifact_dir: str, split: str) -> None:
        self.root = Path(artifact_dir)
        self.split = str(split)
        split_dir = self.root / self.split
        ids_path = split_dir / "ids.txt"
        labels_path = split_dir / "labels.npy"
        valid_mask_path = split_dir / "valid_mask.npy"
        if not ids_path.exists():
            raise FileNotFoundError(ids_path)
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        if not valid_mask_path.exists():
            raise FileNotFoundError(valid_mask_path)

        self.ids = [normalize_volume_id(line.strip()) for line in ids_path.read_text().splitlines() if line.strip()]
        self.row_by_id = {volume_id: row for row, volume_id in enumerate(self.ids)}
        if len(self.row_by_id) != len(self.ids):
            raise ValueError(f"{ids_path} contains duplicate normalized volume ids")

        self.labels = np.load(labels_path, mmap_mode="r")
        self.valid_mask = np.asarray(np.load(valid_mask_path, mmap_mode="r"), dtype=bool)
        self.organ_groups = [str(x) for x in json.loads((self.root / "organ_groups.json").read_text())]
        self.disease_names = [str(x) for x in json.loads((self.root / "disease_names.json").read_text())]
        group_to_mask_path = self.root / "group_to_mask_name.json"
        if group_to_mask_path.exists():
            self.group_to_mask_name = {str(k): str(v) for k, v in json.loads(group_to_mask_path.read_text()).items()}
        else:
            self.group_to_mask_name = {group: group for group in self.organ_groups}

        expected_shape = (len(self.ids), len(self.organ_groups), len(self.disease_names))
        if tuple(self.labels.shape) != expected_shape:
            raise ValueError(f"{labels_path} shape={self.labels.shape}, expected {expected_shape}")
        if tuple(self.valid_mask.shape) != (len(self.organ_groups), len(self.disease_names)):
            raise ValueError(
                f"{valid_mask_path} shape={self.valid_mask.shape}, expected "
                f"{(len(self.organ_groups), len(self.disease_names))}"
            )


class DiseaseClassificationSignal(nn.Module):
    """Pool organ-masked visual tokens and predict CT-RATE disease labels."""

    def __init__(self, config: DiseaseClassificationConfig, visual_dim: int) -> None:
        super().__init__()
        self.config = config
        self.visual_dim = int(visual_dim)
        self.artifact: _DiseaseArtifact | None = None
        self.mask_root: Path | None = None
        if not config.enabled:
            self.num_organs = 0
            self.num_diseases = 0
            return
        if config.backend != "organ_masked_bce_artifact":
            raise ValueError(f"unsupported disease_classification.backend={config.backend!r}")
        if config.loss != "bce_with_logits":
            raise ValueError(f"unsupported disease_classification.loss={config.loss!r}")
        if not config.artifact_dir:
            raise ValueError("disease_classification.artifact_dir is required when enabled")
        if not config.mask_artifact_dir:
            raise ValueError("reconstruction.mask_artifact_dir is required for disease_classification")

        self.artifact = _DiseaseArtifact(config.artifact_dir, config.split)
        self.mask_root = Path(config.mask_artifact_dir) / config.split
        if not self.mask_root.exists():
            raise FileNotFoundError(self.mask_root)
        self.num_organs = len(self.artifact.organ_groups)
        self.num_diseases = len(self.artifact.disease_names)
        self.register_buffer(
            "valid_mask",
            torch.as_tensor(self.artifact.valid_mask.copy(), dtype=torch.bool),
            persistent=False,
        )
        self.organ_embedding = nn.Embedding(self.num_organs, self.visual_dim) if config.organ_embedding else None
        hidden_dim = int(config.hidden_dim)
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(self.visual_dim),
                nn.Linear(self.visual_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_diseases),
            )
        else:
            self.head = nn.Sequential(nn.LayerNorm(self.visual_dim), nn.Linear(self.visual_dim, self.num_diseases))

    @staticmethod
    def _tokens_bdhwc(z_quantized: torch.Tensor, target_dhw: tuple[int, int, int] | None) -> torch.Tensor:
        if z_quantized.ndim != 5:
            raise ValueError(f"expected B,C,D,H,W latent tokens, got {tuple(z_quantized.shape)}")
        tokens = z_quantized.float()
        if target_dhw is not None and tuple(tokens.shape[2:]) != tuple(target_dhw):
            tokens = F.interpolate(tokens, size=tuple(target_dhw), mode="trilinear", align_corners=False)
        return tokens.permute(0, 2, 3, 4, 1).contiguous()

    def _load_masks(
        self,
        volume_id: str,
        target_dhw: tuple[int, int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        assert self.mask_root is not None
        path = self.mask_root / f"{normalize_volume_id(volume_id)}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        npz = np.load(path, allow_pickle=False)
        names = [str(x) for x in npz["mask_names"]]
        out: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(names):
            out[name] = downsample_mask_64_to_grid(npz["mask_token"][idx], target_dhw, device, torch.float32)
        return out

    def forward(
        self,
        *,
        z_quantized: torch.Tensor,
        volume_ids: list[str],
        full_shape_dhw: tuple[int, int, int],
        crop_depth_value: int,
        step: int,
        disease_batch: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del full_shape_dhw, crop_depth_value
        if not self.config.enabled:
            zero = z_quantized.new_zeros(())
            return zero, {"disease_loss": 0.0, "disease_pairs": 0, "disease_organs": 0, "disease_pos": 0}
        if len(volume_ids) != int(z_quantized.shape[0]):
            raise ValueError(f"volume_ids length={len(volume_ids)} does not match batch={z_quantized.shape[0]}")
        assert self.artifact is not None

        # z_quantized arrives from the CT tokenizer as [B,C,D,H,W].
        # For the first 16x16x8 disease experiments the expected grid is
        # [B,C,31,32,32], then this is permuted to [B,31,32,32,C].
        feature_grid = self.config.feature_grid_dhw
        target_feature_grid = (
            (int(z_quantized.shape[2]), int(feature_grid[1]), int(feature_grid[2]))
            if feature_grid is not None
            else None
        )
        tokens = self._tokens_bdhwc(z_quantized, target_feature_grid)
        target_dhw = tuple(int(x) for x in tokens.shape[1:4])
        device = z_quantized.device
        valid_mask = self.valid_mask.to(device=device)
        loss_terms: list[torch.Tensor] = []
        disease_pairs = 0
        disease_organs = 0
        disease_pos = 0
        missing_label_volumes = 0
        missing_masks = 0
        debug_examples: list[dict[str, Any]] = []

        for batch_index, volume_id in enumerate(volume_ids):
            if disease_batch is not None:
                sample_disease = disease_batch[batch_index] if batch_index < len(disease_batch) else {}
                labels_cpu = sample_disease.get("labels")
                if labels_cpu is None:
                    raise RuntimeError(f"disease_classification missing label row for volume_id={volume_id}")
                labels = labels_cpu.to(device=device, dtype=torch.float32)
                raw_masks = sample_disease.get("masks", {})
                masks = {
                    str(name): downsample_mask_64_to_grid(mask, target_dhw, device, torch.float32)
                    for name, mask in raw_masks.items()
                }
            else:
                row_index = self.artifact.row_by_id.get(normalize_volume_id(volume_id))
                if row_index is None:
                    raise RuntimeError(f"disease_classification missing label row for volume_id={volume_id}")
                labels_np = np.asarray(self.artifact.labels[row_index], dtype=np.float32)
                labels = torch.as_tensor(labels_np, device=device, dtype=torch.float32)
                masks = self._load_masks(volume_id, target_dhw, device)
            for organ_index, organ_group in enumerate(self.artifact.organ_groups):
                organ_valid = valid_mask[organ_index]
                if not bool(organ_valid.any().detach().cpu()):
                    continue
                mask_name = self.artifact.group_to_mask_name.get(organ_group, organ_group)
                mask = masks.get(mask_name)
                if mask is None:
                    raise KeyError(
                        f"disease_classification mask {mask_name!r} missing for volume_id={volume_id} "
                        f"organ_group={organ_group!r}; available={sorted(masks.keys())[:24]}"
                    )
                denom = mask.sum().clamp_min(float(self.config.min_mask_sum))
                if float(denom.detach().cpu()) <= float(self.config.min_mask_sum):
                    raise RuntimeError(
                        f"disease_classification empty mask for volume_id={volume_id} "
                        f"organ_group={organ_group!r} mask_name={mask_name!r}"
                    )
                feature = (tokens[batch_index] * mask.unsqueeze(-1)).sum(dim=(0, 1, 2)) / denom
                if self.organ_embedding is not None:
                    organ_id = torch.as_tensor(organ_index, device=device, dtype=torch.long)
                    feature = feature + self.organ_embedding(organ_id)
                logits = self.head(feature.to(dtype=next(self.head.parameters()).dtype))
                target = labels[organ_index]
                per_pair = F.binary_cross_entropy_with_logits(logits[organ_valid], target[organ_valid], reduction="none")
                loss_terms.append(per_pair)
                disease_pairs += int(organ_valid.sum().detach().cpu())
                disease_organs += 1
                disease_pos += int(target[organ_valid].sum().detach().cpu())
                if len(debug_examples) < 12:
                    debug_examples.append(
                        {
                            "volume_id": volume_id,
                            "organ_group": organ_group,
                            "mask_name": mask_name,
                            "status": "matched",
                            "denom": float(denom.detach().cpu()),
                            "valid_disease_pairs": int(organ_valid.sum().detach().cpu()),
                            "positive_labels": int(target[organ_valid].sum().detach().cpu()),
                            "logits_shape": list(logits.shape),
                        }
                    )

        if SMOKE_DEBUG_PRINTS_HARDCODED and step == SMOKE_DEBUG_BREAKPOINT_STEP and _rank0():
            print(
                "[debug:disease_classification] "
                + json.dumps(
                    {
                        "volume_ids": volume_ids,
                        "z_quantized_shape": list(z_quantized.shape),
                        "tokens_bdhwc_shape": list(tokens.shape),
                        "target_dhw": list(target_dhw),
                        "disease_pairs": disease_pairs,
                        "disease_organs": disease_organs,
                        "disease_pos": disease_pos,
                        "missing_label_volumes": missing_label_volumes,
                        "missing_masks": missing_masks,
                        "examples": debug_examples,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if (
            SMOKE_DEBUG_BREAKPOINT_HARDCODED
            and step == SMOKE_DEBUG_BREAKPOINT_STEP
            and _rank0()
            and _breakpoints_enabled()
        ):
            print("[debug:breakpoint] disease classification collected pairs; inspect loss_terms/debug_examples, then type c", flush=True)
            breakpoint()

        if not loss_terms:
            raise RuntimeError(
                "disease_classification produced zero organ-disease pairs; "
                f"missing_label_volumes={missing_label_volumes} missing_masks={missing_masks} "
                f"feature_grid={target_dhw}"
            )
        loss = torch.cat(loss_terms).mean()
        return loss, {
            "disease_loss": float(loss.detach().cpu()),
            "disease_pairs": disease_pairs,
            "disease_organs": disease_organs,
            "disease_pos": disease_pos,
            "disease_missing_label_volumes": missing_label_volumes,
            "disease_missing_masks": missing_masks,
            "disease_feature_grid": "x".join(str(x) for x in target_dhw),
        }
