"""location_merlin family — Merlin landmark-position targets. REGRESS / R2. role=primary. readout=abmil_multi.

Targets selected 2026-07-18 (col order) = [kidney_z, kidney_lr_z_asym, bladder_z].

These are organ z-centroids normalized by the vertebral span, i.e. WHERE a landmark sits along the
cranio-caudal axis rather than how big it is. They are the family that a content-adaptive merge is expected to
DAMAGE (merging destroys the grid order that encodes position) and that ORCA's sinusoidal centroid
re-injection is designed to restore -- so this family is the direct test of that mechanism on Merlin, the way
CT-RATE `location` is on chest CT.

kidney_lr_z_asym is the left-right z offset between the two kidneys: undefined with a single kidney, so it is
blanked when n_kidneys<2 rather than silently computed from one.

Guards at merge (merge_all.py, PER-TARGET blanking): n_kidneys<2 for the two kidney targets, bladder<=5 mL
for bladder_z (empty/failed mask). location_labels.csv = 25,470 volumes; one CSV serves both splits.
No log1p: normalized positions, already ~symmetric."""
from _paths import MERLIN

SPEC = {
    "task": "regress", "metric": "r2", "readout": "abmil_multi", "role": "primary",
    "train": f"{MERLIN}/location_labels.csv",
    "valid": f"{MERLIN}/location_labels.csv",
}
