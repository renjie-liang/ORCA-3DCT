"""density_merlin family — Merlin HU-value density targets. REGRESS / R2. role=primary. readout=abmil_multi.

The Merlin counterpart of `density` (CT-RATE). 3 targets (col order) = [vert_L1T12_median, muscle_auto_median,
liver_spleen_diff]. All CONTINUOUS HU measures -> NO log1p (unlike CT-RATE density, whose fraction targets
aorta_calc/lung_haacon are zero-inflated and need log1p; Merlin's three are plain HU, ~normal already).

Chosen for contrast-robustness on portal-venous abdomen (r^2 vs aortic HU <=0.04) and vetted one at a time
(viz/merlin_vert, viz/merlin_muscle, viz/merlin_liver): vert=osteoporosis, muscle=myosteatosis (age-anchored,
no report finding), liver-spleen=steatosis. Guards applied at merge (merge_density.py, per-target blanking):
vert<20mL, muscle<40mL, spleen<30mL|liver<400mL -> the affected target blanked as NaN, masked in the R2.
labels_density.csv = 25,469 volumes; one CSV serves both splits (train/valid from the manifest id lists)."""
from _paths import MERLIN

SPEC = {
    "task": "regress", "metric": "r2", "readout": "abmil_multi", "role": "primary",
    "train": f"{MERLIN}/density_labels.csv",
    "valid": f"{MERLIN}/density_labels.csv",
}
