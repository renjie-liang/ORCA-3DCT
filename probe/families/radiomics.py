"""radiomics family — texture / histogram-shape measurements (mask-derived, pyradiomics). REGRESS / R2.
role=primary. readout=abmil_multi.

FINAL targets (col order, chosen 2026-07-12 from the full-extraction selection_report.csv by test-retest +
disease-AUC + redundancy vs density) = 3 INDEPENDENT reliable axes:
  [0] lung_firstorder_Kurtosis  high-density lung texture (fibrosis/consolidation); retest 0.93, AUC 0.885, ~0.75 redundant w/ density
  [1] vert_firstorder_Kurtosis  vertebral trabecular texture (bone); retest 0.82, corr ~0.07 w/ lung -> independent axis; AUC 0.58 (no bone-disease label)
  [2] lung_Perc15               low-density/emphysema axis; retest 0.48 WEAK (aux) but the ONLY density-independent lung signal
The provisional low-density pair LAA950/Perc15 proved UNRELIABLE on full extraction (retest -0.32/0.48); kept
only Perc15 as a weak aux. GLCM Contrast recorded but UNRELIABLE -> never selected.

log1p_cols=[0,1]: both Kurtosis targets are extremely right-skewed (skew ~45/49, max ~1075/725, floor ~1) ->
log1p(x) tames the tail so R2 is meaningful (values are O(1-1000), so scale=1). Perc15 ~symmetric (skew 0.89) ->
no transform. Extractor data/radiomics/; merge/select merge_radiomics.py; review docs/radiomics_probing.md."""
from _paths import DATA

SPEC = {
    "task": "regress", "metric": "r2", "readout": "abmil_multi", "role": "primary",
    "log1p_cols": [0, 1], "log1p_scale": 1.0,
    "train": f"{DATA}/radiomics/labels_train.csv",
    "valid": f"{DATA}/radiomics/labels_valid.csv",
}
