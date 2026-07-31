"""density family — HU-value measurements (mask-derived). REGRESS / R2. role=primary. readout=abmil_multi.

4 targets (col order) = [hu_aorta_calc, vert_median, lung_mean(MLD), hu_lung_haacon]; per-target NaN where the
organ guard failed (masked in the probe loss/R2). log1p_cols = the right-skewed zero-inflated FRACTIONS
(aorta_calc idx0, lung_haacon idx3): log1p(scale*x) makes them ~normal so R2 is meaningful (raw-space R2 is
dominated by the 0-spike). scale=1000 because the fractions are tiny (~1e-3) -> plain log1p(x)~x is useless
(skew 6.4->6.2); log1p(1000x) spreads the tail (skew 6.4->1.7). vert_median/lung_mean continuous -> no transform.
Extractor data/density/; review docs/density_probing.md."""
from _paths import DATA

SPEC = {
    "task": "regress", "metric": "r2", "readout": "abmil_multi", "role": "primary",
    "log1p_cols": [0, 3], "log1p_scale": 1000.0,
    "train": f"{DATA}/density/labels_train.csv",
    "valid": f"{DATA}/density/labels_valid.csv",
}
