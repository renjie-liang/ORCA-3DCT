"""disease family — CT-RATE multi-label report abnormalities (18 findings). CLASSIFY / AUROC. role=primary.

The VALIDATION ANCHOR: report-derived diagnoses (the reference the measurement families validate against),
NOT an image-grounded measurement. readout=abmil (1 gated-attention focus, multi-label detection). Labels are
the CT-RATE predicted_labels CSVs (unchanged across the 2026-07 re-extraction -> disease results stay valid)."""
from _paths import CTRATE

SPEC = {
    "task": "classify", "metric": "auroc", "readout": "abmil", "role": "primary",
    "train": f"{CTRATE}/train_predicted_labels.csv",
    "valid": f"{CTRATE}/valid_predicted_labels.csv",
}
