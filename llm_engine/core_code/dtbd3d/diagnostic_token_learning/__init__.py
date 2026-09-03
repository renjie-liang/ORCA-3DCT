"""Modular supervision for diagnosis-aware CT token learning."""

from dtbd3d.diagnostic_token_learning.config import DiagnosticSupervisionConfig, load_supervision_config
from dtbd3d.diagnostic_token_learning.disease_classification import DiseaseClassificationSignal
from dtbd3d.diagnostic_token_learning.organ_input import apply_organ_input_conditioning
from dtbd3d.diagnostic_token_learning.supervision import DiagnosticSupervision, SupervisionOutput

__all__ = [
    "DiagnosticSupervision",
    "DiagnosticSupervisionConfig",
    "DiseaseClassificationSignal",
    "SupervisionOutput",
    "apply_organ_input_conditioning",
    "load_supervision_config",
]
