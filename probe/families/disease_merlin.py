"""disease_merlin family — Merlin zero-shot report findings (30 abnormalities). CLASSIFY / AUROC. role=primary.

The Merlin-dataset counterpart of `disease` (CT-RATE): same role (report-derived validation anchor, not an
image-grounded measurement), same readout. Labels ship with the dataset (zero_shot_findings_disease_cls.csv,
positive/negative prompt matching over the report findings section).

Encoding matters: Merlin's README defines missing=-1, negative=0, positive=1 -- so -1 is NOT a negative, it is
"the prompt matcher found neither positive nor negative evidence". The derived CSV writes -1 as blank, which
load_labels reads as NaN = per-cell MISSING, so the masked BCE/AUROC path skips those cells. Treating -1 as
negative instead would silently relabel 77% of all cells (the missing majority) as confident negatives.
Only 22.4% of cells are labeled; every one of the 30 classes still has >=10 valid positives.

One CSV serves both splits: train/valid come from the manifest id lists, not from separate label files."""
from _paths import MERLIN

SPEC = {
    "task": "classify", "metric": "auroc", "readout": "abmil", "role": "primary",
    "train": f"{MERLIN}/disease_labels.csv",
    "valid": f"{MERLIN}/disease_labels.csv",
}
