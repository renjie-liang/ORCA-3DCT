"""Configuration objects for diagnosis-aware token supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OrganInputConfig:
    enabled: bool = False
    mode: str = "importance_residual"
    strength: float = 0.0
    clamp_min: float | None = None
    clamp_max: float | None = None


@dataclass(frozen=True)
class AnatomyLossConfig:
    enabled: bool = False
    note: str = "Existing learned_table.loss_variant remains the active anatomy-loss path."


@dataclass(frozen=True)
class TextEmbeddingSourceConfig:
    name: str
    artifact_dir: str
    mask_artifact_dir: str
    split: str = "train"
    max_pairs_per_volume: int = 7


@dataclass(frozen=True)
class TextAlignmentConfig:
    enabled: bool = False
    weight: float = 0.0
    mode: str = "multi_positive_infonce"
    temperature: float = 0.07
    min_mask_sum: float = 1e-6
    backend: str = "llama_hidden_artifact"
    embedding_sources: list[TextEmbeddingSourceConfig] = field(default_factory=list)
    hidden_dim: int = 4096
    reportgen_grid_dhw: tuple[int, int, int] = (31, 8, 8)


@dataclass(frozen=True)
class DiseaseClassificationConfig:
    enabled: bool = False
    weight: float = 0.0
    backend: str = "organ_masked_bce_artifact"
    artifact_dir: str = ""
    mask_artifact_dir: str = ""
    split: str = "train"
    feature_grid_dhw: tuple[int, int, int] | None = None
    min_mask_sum: float = 1e-6
    organ_embedding: bool = True
    hidden_dim: int = 0
    loss: str = "bce_with_logits"


@dataclass(frozen=True)
class LlamaPrefixPrealignmentConfig:
    enabled: bool = False
    weight: float = 0.0
    model_name_or_path: str = ""
    hidden_dim: int = 4096
    reportgen_grid_dhw: tuple[int, int, int] = (31, 8, 8)
    num_prefix_tokens: int = 32
    max_pairs_per_batch: int = 2
    prompt_template: str = "Region: {region}. Finding:"
    target_template: str = " {finding}."
    freeze_llama: bool = True
    dtype: str = "bfloat16"
    local_files_only: bool = True
    trust_remote_code: bool = False


@dataclass(frozen=True)
class VisualFidelityConfig:
    enabled: bool = False
    gradient_weight: float = 0.0
    window_weight: float = 0.0
    roi_ssim_weight: float = 0.0
    decision_status: str = "not_started"
    note: str = "Gated by visualization audit; default is disabled."


@dataclass(frozen=True)
class DiagnosticSupervisionConfig:
    mask_artifact_dir: str = ""
    organ_input: OrganInputConfig = field(default_factory=OrganInputConfig)
    anatomy_loss: AnatomyLossConfig = field(default_factory=AnatomyLossConfig)
    text_alignment: TextAlignmentConfig = field(default_factory=TextAlignmentConfig)
    disease_classification: DiseaseClassificationConfig = field(default_factory=DiseaseClassificationConfig)
    llama_prefix_prealignment: LlamaPrefixPrealignmentConfig = field(default_factory=LlamaPrefixPrealignmentConfig)
    visual_fidelity: VisualFidelityConfig = field(default_factory=VisualFidelityConfig)
    supervision_lr: float | None = None

    @property
    def has_trainable_auxiliary(self) -> bool:
        return self.text_alignment.enabled or self.disease_classification.enabled or self.llama_prefix_prealignment.enabled


def _string_path(value: Any, root: Path) -> str:
    if value is None or str(value) == "":
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return str(path)


def _load_text_embedding_sources(
    raw_sources: list[dict[str, Any]],
    root: Path,
    default_split: str,
) -> list[TextEmbeddingSourceConfig]:
    sources = []
    for raw_source in raw_sources:
        sources.append(
            TextEmbeddingSourceConfig(
                name=str(raw_source["name"]),
                artifact_dir=_string_path(raw_source["artifact_dir"], root),
                mask_artifact_dir=_string_path(raw_source["mask_artifact_dir"], root),
                split=str(raw_source.get("split", default_split)),
                max_pairs_per_volume=int(raw_source.get("max_pairs_per_volume", 7)),
            )
        )
    return sources


def load_supervision_config(
    raw_config: dict[str, Any],
    root: Path,
    default_split: str = "train",
) -> DiagnosticSupervisionConfig:
    legacy_raw = raw_config.get("supervision", {}) or {}
    reconstruction_raw = raw_config.get("reconstruction", {}) or {}
    input_conditioning_raw = raw_config.get("input_conditioning", {}) or {}
    auxiliary_raw = raw_config.get("auxiliary_losses", {}) or {}
    raw = auxiliary_raw if auxiliary_raw else legacy_raw

    organ_input_raw = (input_conditioning_raw.get("region_mask", {}) or {}) if input_conditioning_raw else (legacy_raw.get("organ_input", {}) or {})
    text_raw = raw.get("text_alignment", {}) or {}
    disease_raw = raw.get("disease_classification", {}) or {}
    llama_prefix_raw = raw.get("llama_prefix_prealignment", {}) or {}
    visual_raw = raw.get("visual_fidelity", {}) or {}
    feature_grid_raw = disease_raw.get("feature_grid_dhw")
    shared_mask_artifact_dir = _string_path(reconstruction_raw.get("mask_artifact_dir", legacy_raw.get("mask_artifact_dir", "")), root)
    disease_mask_artifact_dir = _string_path(disease_raw.get("mask_artifact_dir", ""), root) or shared_mask_artifact_dir

    return DiagnosticSupervisionConfig(
        mask_artifact_dir=shared_mask_artifact_dir,
        organ_input=OrganInputConfig(
            enabled=bool(organ_input_raw.get("enabled", False)),
            mode=str(organ_input_raw.get("mode", "importance_residual")),
            strength=float(organ_input_raw.get("strength", 0.0)),
            clamp_min=organ_input_raw.get("clamp_min"),
            clamp_max=organ_input_raw.get("clamp_max"),
        ),
        anatomy_loss=AnatomyLossConfig(enabled=bool((raw.get("anatomy_loss", {}) or {}).get("enabled", False))),
        text_alignment=TextAlignmentConfig(
            enabled=bool(text_raw.get("enabled", False)),
            weight=float(text_raw.get("weight", 0.0)),
            mode=str(text_raw.get("mode", "multi_positive_infonce")),
            temperature=float(text_raw.get("temperature", 0.07)),
            min_mask_sum=float(text_raw.get("min_mask_sum", 1e-6)),
            backend=str(text_raw.get("backend", "llama_hidden_artifact")),
            embedding_sources=_load_text_embedding_sources(text_raw.get("embedding_sources", []) or [], root, default_split),
            hidden_dim=int(text_raw.get("hidden_dim", 4096)),
            reportgen_grid_dhw=tuple(int(x) for x in text_raw.get("reportgen_grid_dhw", [31, 8, 8])),
        ),
        disease_classification=DiseaseClassificationConfig(
            enabled=bool(disease_raw.get("enabled", False)),
            weight=float(disease_raw.get("weight", 0.0)),
            backend=str(disease_raw.get("backend", "organ_masked_bce_artifact")),
            artifact_dir=_string_path(disease_raw.get("artifact_dir", ""), root),
            mask_artifact_dir=disease_mask_artifact_dir,
            split=str(disease_raw.get("split", default_split)),
            feature_grid_dhw=tuple(int(x) for x in feature_grid_raw) if feature_grid_raw else None,
            min_mask_sum=float(disease_raw.get("min_mask_sum", 1e-6)),
            organ_embedding=bool(disease_raw.get("organ_embedding", True)),
            hidden_dim=int(disease_raw.get("hidden_dim", 0)),
            loss=str(disease_raw.get("loss", "bce_with_logits")),
        ),
        llama_prefix_prealignment=LlamaPrefixPrealignmentConfig(
            enabled=bool(llama_prefix_raw.get("enabled", False)),
            weight=float(llama_prefix_raw.get("weight", 0.0)),
            model_name_or_path=_string_path(llama_prefix_raw.get("model_name_or_path", ""), root),
            hidden_dim=int(llama_prefix_raw.get("hidden_dim", 4096)),
            reportgen_grid_dhw=tuple(int(x) for x in llama_prefix_raw.get("reportgen_grid_dhw", [31, 8, 8])),
            num_prefix_tokens=int(llama_prefix_raw.get("num_prefix_tokens", 32)),
            max_pairs_per_batch=int(llama_prefix_raw.get("max_pairs_per_batch", 2)),
            prompt_template=str(llama_prefix_raw.get("prompt_template", "Region: {region}. Finding:")),
            target_template=str(llama_prefix_raw.get("target_template", " {finding}.")),
            freeze_llama=bool(llama_prefix_raw.get("freeze_llama", True)),
            dtype=str(llama_prefix_raw.get("dtype", "bfloat16")),
            local_files_only=bool(llama_prefix_raw.get("local_files_only", True)),
            trust_remote_code=bool(llama_prefix_raw.get("trust_remote_code", False)),
        ),
        visual_fidelity=VisualFidelityConfig(
            enabled=bool(visual_raw.get("enabled", False)),
            gradient_weight=float(visual_raw.get("gradient_weight", 0.0)),
            window_weight=float(visual_raw.get("window_weight", 0.0)),
            roi_ssim_weight=float(visual_raw.get("roi_ssim_weight", 0.0)),
            decision_status=str(visual_raw.get("decision_status", "not_started")),
            note=str(visual_raw.get("note", "Gated by visualization audit; default is disabled.")),
        ),
        supervision_lr=float(raw["supervision_lr"]) if "supervision_lr" in raw and raw["supervision_lr"] is not None else None,
    )
