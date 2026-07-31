"""size family — organ-size measurements (mask-derived). REGRESS / R2. role=primary. readout=abmil_multi.

5 targets (col order): sz_heart_lung (CT cardiothoracic ratio), sz_aorta_heart (vascular), sz_ivc_aorta
(venous) = scale-invariant log volume-RATIOS; aorta_diameter_mm, heart_width_mm = absolute diameters. All
near-normal after mL-guards -> NO log1p. Per-target NaN where the organ mL-guard failed (masked in the probe).
Extractor data/size/; review docs/size_probing.md."""
from _paths import DATA

SPEC = {
    "task": "regress", "metric": "r2", "readout": "abmil_multi", "role": "primary",
    "train": f"{DATA}/size/labels_train.csv",
    "valid": f"{DATA}/size/labels_valid.csv",
}
