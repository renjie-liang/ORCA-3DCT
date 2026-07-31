"""size_merlin family — Merlin organ-size targets. REGRESS / R2. role=primary. readout=abmil_multi.

Targets selected 2026-07-18: spleen, kidney, aorta diameter. Each is carried in BOTH forms
(col order) = [spleen_ml, spleen_vert_ratio, kidney_ml, kidney_vert_ratio, aorta_diameter_mm, aorta_vert_diam].

Why both forms, deliberately: encoders resize the whole volume to fixed dims, so an ABSOLUTE mL/mm target is
only recoverable if the dataset's field of view is standardized. Merlin's FOV varies 5.1x (CV 0.31), which
should degrade the absolute form, while an organ/vertebra RATIO cancels the resize. We do NOT pre-select the
form -- the probe reports per-target R2 and the data answers it. (Selecting the form by which one the encoder
predicts better would be circular; here the two forms are both targets of the SAME neutral probe.)

Guards at merge (merge_all.py, PER-TARGET blanking): spleen<30 mL (the failure/splenectomy spike density
already cuts), n_kidneys<2 (one kidney makes a total volume a different quantity), non-positive values.
size_labels.csv = 25,477 volumes; one CSV serves both splits (train/valid from the manifest id lists).
No log1p: these are volumes/ratios/mm, not the zero-inflated fractions that CT-RATE's density needs it for."""
from _paths import MERLIN

SPEC = {
    "task": "regress", "metric": "r2", "readout": "abmil_multi", "role": "primary",
    "train": f"{MERLIN}/size_labels.csv",
    "valid": f"{MERLIN}/size_labels.csv",
}
