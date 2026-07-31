"""probe2 experiment configuration (dicts, no yaml dependency)."""

from _paths import CTRATE, RESULTS, DATA   # re-exported for back-compat (from config import RESULTS/DATA/...)
# foreground/organ-coverage prior (upstream dir is named 'organ'; soft [0,1] anatomical-PRESENCE map,
# NOT per-organ labels and NOT diagnostic saliency). SPARSE: ~75% exactly 0 (air/bg), ~15-22% tissue --
# a discriminative body/tissue-vs-air mask (so weighted_pool genuinely differs from pool; cos~0.67).
FOREGROUND_ROOT = "./data/DTBD3D_data/token_importance/organ"
MANIFEST_DIR = "./vendor/npy_manifests"

# encoder -> manifest (the 3 embeddings)
ENCODERS = {
    "btb3d":       f"{MANIFEST_DIR}/btb3d.json",      # 16x16x8 grid 31x32x32 (legacy name; == btb3d_16x16)
    "btb3d_16x16": f"{MANIFEST_DIR}/btb3d.json",      # explicit alias for btb3d (use this for new experiments)
    "btb3d_8x8":   f"{MANIFEST_DIR}/btb3d_8x8.json",  # 8x8x8 grid 31x64x64 (finer; codebook_grid, msb)
    "ct_clip": f"{MANIFEST_DIR}/ct_clip.json",
    "colipri": f"{MANIFEST_DIR}/colipri.json",
    # --- Merlin dataset (2nd dataset; 8x16x16 grids over 25,489 real-HU abdominal CTs) ---
    "merlin_segvol": f"{MANIFEST_DIR}/merlin_segvol.json",
    "merlin_suprem": f"{MANIFEST_DIR}/merlin_suprem.json",
}

# encoder -> foreground/organ-prior root, or None when that dataset has no map yet. CT-RATE encoders
# intersect their ids against FOREGROUND_ROOT; Merlin has no map while its TotalSegmentator masks are
# still extracting, so its ids are filtered on grid availability instead. None => score/organ-needing
# methods (weighted_pool, agglo_organ, ...) cannot run on that encoder -- they fail loudly in ProbeSet.
FOREGROUND_BY_ENCODER = {
    "btb3d": FOREGROUND_ROOT, "btb3d_16x16": FOREGROUND_ROOT, "btb3d_8x8": FOREGROUND_ROOT,
    "ct_clip": FOREGROUND_ROOT, "colipri": FOREGROUND_ROOT,
    "merlin_segvol": None, "merlin_suprem": None,
}

# --- budget grids (per encoder, per method) -------------------------------------------------------
# We do NOT force one shared grid. Each (encoder, method) uses budgets that land on ITS natural token
# counts; the curve is plotted vs the TRUE token count (actual_n), so different grids are comparable.
# See probe2/docs/BUDGETS.md for how each list is derived. Default = the token counts pool lands on at
# integer compression factors r (2x2x2, 3x3x3, ...); other methods hit these exactly too (shared anchors).
BUDGETS_BY_ENCODER = {
    "btb3d":       [8, 27, 64, 125, 216, 512, 1210, 4096],   # 31x32x32 at r = 16,10,8,6,5,4,3,2
    "btb3d_16x16": [8, 27, 64, 125, 216, 512, 1210, 4096],   # alias of btb3d
    "ct_clip": [8, 27, 64, 125, 216, 512, 1728],         # 24^3 at r = 10,8,6,5,4,3,2
    "colipri": [8, 27, 64, 125, 216, 512, 1728],
    # 8x16x16 = 2048 tokens (anisotropic): pool lands on 4/32/108/256 at r = 8,4,3,2 (avgpack_r edge-pads).
    # Fewer rungs than the cubic encoders because the short axis (8) runs out first -- these ARE the natural
    # counts; do not interpolate to the CT-RATE grid, the curve is plotted vs true token count.
    "merlin_segvol": [4, 32, 108, 256],
    "merlin_suprem": [4, 32, 108, 256],
}
# method -> {encoder: [budgets]} override (else BUDGETS_BY_ENCODER). See probe2/docs/BUDGETS.md for why.
BUDGETS_BY_METHOD = {
    # tome/divprune pre-pool to 4096 (16/edge) -> budgets must be < 4096; drop btb3d's 4096 (=pre, no merge)
    "tome":     {"btb3d": [8, 27, 64, 125, 216, 512, 1210], "btb3d_16x16": [8, 27, 64, 125, 216, 512, 1210]},
    # divprune pre-pools to 1024 (FPS is O(budget*pre); pre=4096 was ~200x slower) -> budgets must be <=1024
    "divprune": {"btb3d": [8, 27, 64, 125, 216, 512], "btb3d_16x16": [8, 27, 64, 125, 216, 512], "ct_clip": [8, 27, 64, 125, 216, 512], "colipri": [8, 27, 64, 125, 216, 512]},
    # octree (coarse=4) output is quantized to [71..512]; use its meaningful range on all encoders
    "octree":   {"btb3d": [64, 125, 216, 512], "btb3d_16x16": [64, 125, 216, 512], "ct_clip": [64, 125, 216, 512], "colipri": [64, 125, 216, 512]},
}


def budgets_for(method, encoder):
    return BUDGETS_BY_METHOD.get(method, {}).get(encoder, BUDGETS_BY_ENCODER[encoder])

# expected sample count per (encoder, split) AFTER organ-map intersection (stage1 fail-fast check)
EXPECTED_N = {
    "btb3d":       {"train": 47146, "valid": 3039},
    "btb3d_16x16": {"train": 47146, "valid": 3039},   # alias of btb3d
    "btb3d_8x8":   {"train": 47146, "valid": 3039},   # same volumes/ids as btb3d, finer grid
    "ct_clip":     {"train": 47146, "valid": 3039},   # 3 train vols lack an organ map -> dropped
    "colipri":     {"train": 24128, "valid": 1564},
    # Merlin: official split lists minus ids without an embedding (5 train vols failed extraction).
    # Label coverage is NOT a filter here -- unlabeled rows are carried and masked per-cell downstream.
    "merlin_segvol": {"train": 15309, "valid": 5055},
    "merlin_suprem": {"train": 15309, "valid": 5055},
}

# Volumes whose ORGAN mask is corrupt (empty mask_token in DTBD3D_data/ts_mask_8x8x8) -- the grid + foreground
# map exist so they pass filtered_ids, but organ_vec crashes (adaptive_pool on 0 channels). Dropped ONLY for
# organ-needing methods; the EXPECTED_N assert still checks the RAW (pre-exclusion) count, so the dataset
# integrity guard is intact. Populated from viz/organ_alignment/code (scan_btb3d_empty). btb3d only.
BAD_ORGAN_VIDS = {
    # empty mask_token in DTBD3D_data/ts_mask_8x8x8 (header scan 2026-07-21: exactly 2 of 47146 train, 0 valid)
    "btb3d": {"train_10847_a_2", "train_12477_a_1"},
}

# family -> spec dict, now ONE FILE PER FAMILY in probe_families/ (mirrors compress_operators/). Each family is
# probed in its OWN separate run (no multi-task mixing). See probe2/probe_families/{disease,size,density,
# location,radiomics}.py for the specs + rationale. DROPPED (2026-07-11): biomarker (redundant; heart_calc_frac
# anatomically failed) + metadata (DICOM header, off-thesis; kept only as a confound-control covariate).
from families import FAMILY_REGISTRY as FAMILIES

# probe hyperparameters (lighter than the old 15-20 epochs; validate ranking stability before trusting)
PROBE = {"epochs": 10, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64,
         "readout": "abmil", "readout_dim": 256, "readout_heads": 4, "dropout": 0.1,
         "input_norm": "ln", "seeds": [2026, 2027, 2028]}  # multi-seed for small-gap significance
