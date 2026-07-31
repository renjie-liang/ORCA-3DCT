"""Shared building blocks used by multiple compression operators (no registration here).
Operator interface everywhere: fn(grid[T,H,W,C], budget, score) -> [N_tokens, C] float32."""
import numpy as np
import torch
import torch.nn.functional as F


def flatten_grid(grid):                                    # [T,H,W,C] -> [T*H*W, C]
    return grid.reshape(-1, grid.shape[-1]).astype(np.float32)


def pack_r(grid, r):                                       # [T,H,W,C] -> [N, C*r^3] channel-major (matches RG trainer)
    T, H, W, C = grid.shape
    pt, ph, pw = (-T) % r, (-H) % r, (-W) % r
    if pt or ph or pw:
        grid = np.pad(grid, ((0, pt), (0, ph), (0, pw), (0, 0)), mode="edge")
    T, H, W, C = grid.shape
    blocks = grid.reshape(T // r, r, H // r, r, W // r, r, C)
    return blocks.transpose(0, 2, 4, 6, 1, 3, 5).reshape((T // r) * (H // r) * (W // r), C * r ** 3).astype(np.float32)


def avgpack_r(grid, r):                                    # [T,H,W,C] -> [N/r^3, C] block-average (TRUE compression, channel-preserving)
    T, H, W, C = grid.shape                                # avg counterpart of pack_r: mean over each r^3 block instead of concat.
    pt, ph, pw = (-T) % r, (-H) % r, (-W) % r              # -> token count down r^3, channel stays C (no dim blow-up, no proj needed).
    if pt or ph or pw:
        grid = np.pad(grid, ((0, pt), (0, ph), (0, pw), (0, 0)), mode="edge")
    T, H, W, C = grid.shape
    blocks = grid.reshape(T // r, r, H // r, r, W // r, r, C)
    return blocks.mean(axis=(1, 3, 5)).reshape((T // r) * (H // r) * (W // r), C).astype(np.float32)


def _pool_shape(grid, budget):                             # target (t,h,w) ~budget tokens, keeping the grid aspect ratio
    T, H, W = grid.shape[:3]
    s = (budget / (T * H * W)) ** (1 / 3)
    return max(1, round(T * s)), max(1, round(H * s)), max(1, round(W * s))


def uniform_pool(grid, budget):                            # [T,H,W,C] -> [t*h*w, C] avg-pool to an ASPECT-RATIO-preserving grid
    t, h, w = _pool_shape(grid, budget)
    x = torch.as_tensor(grid).permute(3, 0, 1, 2)[None]
    pooled = F.adaptive_avg_pool3d(x, (t, h, w))[0].permute(1, 2, 3, 0)
    return pooled.reshape(t * h * w, grid.shape[-1]).numpy().astype(np.float32)


def pool_to_cube(grid, n):                                 # [T,H,W,C] -> [n^3, C] exact n x n x n cube (octree subdivision needs this)
    x = torch.as_tensor(grid).permute(3, 0, 1, 2)[None]
    pooled = F.adaptive_avg_pool3d(x, (n, n, n))[0].permute(1, 2, 3, 0)
    return pooled.reshape(n ** 3, grid.shape[-1]).numpy().astype(np.float32)


def trilinear_downsample(grid, budget):                    # [T,H,W,C] -> [t*h*w, C] trilinear downsample, aspect-ratio preserving
    t, h, w = _pool_shape(grid, budget)
    x = torch.as_tensor(grid).permute(3, 0, 1, 2)[None]
    out = F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)[0].permute(1, 2, 3, 0)
    return out.reshape(t * h * w, grid.shape[-1]).numpy().astype(np.float32)


def dim_crop(tokens, components, mean, D):                 # [N,C] -> [N,D] project onto top-D PCA axes (D<=C)
    # components = Vt [C,C] (rows = principal axes, desc variance); mean [C]. See nxd_fit_pca.py.
    return ((tokens.astype(np.float32) - mean) @ components[:D].T).astype(np.float32)


def topk_by_score(grid, score, k):                         # [k,C] top-k tokens by per-token score
    tokens = flatten_grid(grid); k = min(k, tokens.shape[0])
    idx = np.argpartition(-score, k - 1)[:k]
    return tokens[idx].astype(np.float32)


def resize_score_to_grid(importance, grid):                # [D,H,W] map -> per-token score [T*H*W] on the grid
    x = torch.as_tensor(np.asarray(importance, np.float32))[None, None]
    return F.adaptive_avg_pool3d(x, grid.shape[:3])[0, 0].numpy().reshape(-1)


def foreground_score(volume_id, split, grid):              # anatomical FOREGROUND-coverage prior (data-side; no model)
    from pathlib import Path
    from config import FOREGROUND_ROOT
    from io_utils import _strip
    importance = np.load(Path(FOREGROUND_ROOT) / split / f"{_strip(volume_id)}.npy")
    return resize_score_to_grid(importance, grid)


# Per-encoder organ-mask roots. Each encoder's masks are PRE-BAKED to that encoder's own grid frame at extraction
# (colipri: extract_organ_colipri.py grid_align = flip last axis; ct_clip: extract_organ_ctclip.py = identity, both
# verified by IoU vs the old mask), so organ_vec is a SINGLE path-driven IDENTITY load -- no per-encoder rotation.
# Adding an encoder = one row here + its exact-geometry extractor. An unlisted encoder raises KeyError (fail fast).
ORGAN_ROOT = {
    "colipri":     "./data/DTBD3D_data/organ_masks_colipri",
    "ct_clip":     "./data/DTBD3D_data/organ_masks_ctclip",
    "btb3d":       "./data/DTBD3D_data/ts_mask_8x8x8",   # fast_token mask_token (11,31,64,64)
    "btb3d_16x16": "./data/DTBD3D_data/ts_mask_8x8x8",   # alias of btb3d
    "btb3d_8x8":   "./data/DTBD3D_data/ts_mask_8x8x8",
    # Merlin: 12 ABDOMINAL groups baked per encoder (data_prep/mask_extract/extract_organ_merlin.py). The two
    # encoders do NOT share an axis convention (SegVol (Z,X,Y) no flip; SuPreM RAS = L/R flipped), so each has
    # its own directory -- verified with a flip-SENSITIVE test (liver right of spleen), not a symmetric one.
    "merlin_segvol": "./data/Merlin/organ_masks/merlin_segvol",
    "merlin_suprem": "./data/Merlin/organ_masks/merlin_suprem",
}


# On-disk layout, DECLARED per encoder rather than probed. CT-RATE masks are written under <root>/<split>/<vid>.npz;
# the Merlin extractor writes FLAT (<root>/<vid>.npz), which is unambiguous there because Merlin train/valid volume
# ids are disjoint (verified: overlap 0). Declaring it beats an `if not exists: try the other place` fallback -- that
# would turn "masks for this split were never extracted" into a silent read of the wrong file.
ORGAN_SPLIT_SUBDIR = {"colipri": True, "ct_clip": True, "merlin_segvol": False, "merlin_suprem": False}


def organ_vec(volume_id, split, grid, encoder):            # [T*H*W, 11] soft organ-occupancy per token, aligned to encoder grid
    from pathlib import Path
    T, H, W = grid.shape[:3]
    root = Path(ORGAN_ROOT[encoder])
    p = (root / split / f"{volume_id}.npz") if ORGAN_SPLIT_SUBDIR.get(encoder, True) else (root / f"{volume_id}.npz")
    d = np.load(p, allow_pickle=True)
    if encoder in ("colipri", "ct_clip", "merlin_segvol", "merlin_suprem"):
        m = d["masks"].astype(np.float32)                 # (11,T,H,W) pre-baked to the grid frame (identity)
    else:                                                 # btb3d*: fast_token mask_token (11,31,64,64) -> soft-pool to grid
        import torch, torch.nn.functional as F
        raw = torch.as_tensor(d["mask_token"].astype(np.float32))[None]     # (1,11,31,64,64)
        m = F.adaptive_avg_pool3d(raw, (T, H, W))[0].numpy()               # (11,T,H,W)
    m = np.moveaxis(m, 0, -1)                              # (T,H,W,11)
    assert m.shape[:3] == (T, H, W), f"organ {m.shape[:3]} != grid {(T, H, W)}"
    return np.ascontiguousarray(m.reshape(T * H * W, m.shape[-1]), np.float32)


# Encoder-attention roots for the MedPruner-DINS baseline. Only encoders with transformer self-attention can
# supply this; btb3d (MAGVIT-2 CNN + LFQ) has none and is excluded from that baseline rather than faked.
ATTN_ROOT = {
    "colipri": "./data/CT-RATE/embeddings/colipri_attn",
    "merlin_segvol": "./data/Merlin/embeddings/segvol_attn",
    # merlin_suprem: pending (Swin windowed-attention extractor); btb3d has no attention (CNN, excluded).
}
import os as _os
ATTN_VARIANT = _os.environ.get("ATTN_VARIANT", "all")   # all | last | last4 (measured equivalent for top-B selection)


def attn_score(volume_id, split, grid, encoder):        # [T*H*W] per-token attention received
    from pathlib import Path as _P
    p = _P(ATTN_ROOT[encoder]) / split / "attn_received" / ATTN_VARIANT / f"{volume_id}.npy"
    a = np.load(p).astype(np.float32)
    assert a.shape == grid.shape[:3], f"attn {a.shape} != grid {grid.shape[:3]} ({volume_id})"
    return a.reshape(-1)
