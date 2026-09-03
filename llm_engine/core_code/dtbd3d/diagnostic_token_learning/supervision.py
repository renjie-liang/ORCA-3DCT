"""Top-level auxiliary supervision module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from dtbd3d.diagnostic_token_learning.config import DiagnosticSupervisionConfig
from dtbd3d.diagnostic_token_learning.disease_classification import DiseaseClassificationSignal
from dtbd3d.diagnostic_token_learning.llama_prefix_prealignment import LlamaPrefixPrealignment
from dtbd3d.diagnostic_token_learning.llama_space_alignment import LlamaSpaceTextAlignment


@dataclass
class SupervisionOutput:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    metrics: dict[str, Any]


class DiagnosticSupervision(nn.Module):
    """Compute diagnosis-aware auxiliary losses for CT token training."""

    def __init__(self, config: DiagnosticSupervisionConfig, visual_dim: int) -> None:
        super().__init__()
        self.config = config
        supported_text_backends = {"llama_hidden_artifact", "medical_bert_artifact"}
        if config.text_alignment.enabled and config.text_alignment.backend not in supported_text_backends:
            raise ValueError(f"unsupported text_alignment.backend={config.text_alignment.backend!r}")
        self.text = LlamaSpaceTextAlignment(config.text_alignment, visual_dim) if config.text_alignment.enabled else None
        self.disease = DiseaseClassificationSignal(config.disease_classification, visual_dim)
        self.llama_prefix = (
            LlamaPrefixPrealignment(config.llama_prefix_prealignment, visual_dim)
            if config.llama_prefix_prealignment.enabled
            else None
        )

    def forward(
        self,
        *,
        z_quantized: torch.Tensor,
        volume_ids: list[str],
        importance_weights: torch.Tensor | None,
        full_shape_dhw: tuple[int, int, int],
        crop_depth_value: int,
        step: int,
        text_batch: Any | None = None,
        disease_batch: Any | None = None,
    ) -> SupervisionOutput:
        total = z_quantized.new_zeros(())
        components: dict[str, torch.Tensor] = {}
        metrics: dict[str, Any] = {}

        del importance_weights

        if self.text is not None:
            text_loss, text_metrics = self.text(
                z_quantized=z_quantized,
                volume_ids=volume_ids,
                text_batch=text_batch,
                full_shape_dhw=full_shape_dhw,
                crop_depth_value=crop_depth_value,
                step=step,
            )
            if self.config.text_alignment.weight > 0:
                weighted = float(self.config.text_alignment.weight) * text_loss
                total = total + weighted
                components["text_alignment"] = weighted
            metrics.update(text_metrics)
        else:
            metrics.update({"text_loss": 0.0, "text_pairs": 0})

        if self.llama_prefix is not None:
            prefix_loss, prefix_metrics = self.llama_prefix(
                z_quantized=z_quantized,
                volume_ids=volume_ids,
                text_batch=text_batch,
                step=step,
            )
            if self.config.llama_prefix_prealignment.weight > 0:
                weighted = float(self.config.llama_prefix_prealignment.weight) * prefix_loss
                total = total + weighted
                components["llama_prefix_prealignment"] = weighted
            metrics.update(prefix_metrics)
        else:
            metrics.update({"llama_prefix_loss": 0.0, "llama_prefix_pairs": 0, "llama_prefix_target_tokens": 0})

        disease_loss, disease_metrics = self.disease(
            z_quantized=z_quantized,
            volume_ids=volume_ids,
            disease_batch=disease_batch,
            full_shape_dhw=full_shape_dhw,
            crop_depth_value=crop_depth_value,
            step=step,
        )
        if self.config.disease_classification.enabled and self.config.disease_classification.weight > 0:
            weighted = float(self.config.disease_classification.weight) * disease_loss
            total = total + weighted
            components["disease_classification"] = weighted
        metrics.update(disease_metrics)

        metrics["aux_loss"] = float(total.detach().cpu())
        return SupervisionOutput(loss=total, components=components, metrics=metrics)
