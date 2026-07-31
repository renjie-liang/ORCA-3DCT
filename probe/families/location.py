"""location family — normalized anatomical landmark POSITIONS (mask centroids). REGRESS / R2. role=aux
(spatial "canary": does compression preserve WHERE anatomy sits; weak R2, glance only). readout=abmil_multi.

3 targets (selected from 19; all test-retest 0.87-0.99 so std+redundancy drove selection) = 3 orthogonal spatial
axes: lung_LR_logratio (L/R lung-volume balance), heart_x (mediastinal laterality), ivc_z (cranio-caudal).
DROPPED diaphragm_asym (2026-07-12): R2 ~0.12/~0 on the full probe = no discrimination (probe predicts its mean).
Positions ~symmetric -> NO log1p; guards barely trim; per-target NaN masked.
Extractor data/location/; review docs/location_probing.md."""
from _paths import DATA

SPEC = {
    "task": "regress", "metric": "r2", "readout": "abmil_multi", "role": "aux",
    "train": f"{DATA}/location/labels_train.csv",
    "valid": f"{DATA}/location/labels_valid.csv",
}
