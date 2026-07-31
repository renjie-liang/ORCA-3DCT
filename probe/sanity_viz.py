#!/usr/bin/env python3
"""Per-run alignment canary. Renders the FIRST few volumes' embedding-structure + organ-mask overlay using the
EXACT production loaders (GridLoader + organ_vec) that feed the probe. Saved next to each run's results, so every
experiment self-certifies that the embedding and organ mask it consumed are spatially aligned. No raw CT (kept
cheap/encoder-agnostic). Caller wraps this in try/except -- a viz failure must never abort a training run."""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import ENCODERS
from io_utils import GridLoader
import compressors as co

_O = 64
ORG = [(0, "lung", "#3399ff"), (2, "heart", "#22dd22"), (3, "aorta", "#ff2222"), (9, "spine", "#ff22ff")]


def _rs(x):
    return F.adaptive_avg_pool3d(torch.as_tensor(np.ascontiguousarray(x, np.float32)[None, None]), (_O, _O, _O))[0, 0].numpy()


def _views(v):
    D, H, W = v.shape
    return [np.rot90(v[D // 2]), np.flipud(v[:, H // 2]), np.flipud(v[:, :, W // 2])]


def render_sanity(encoder, split, ids, out_dir, n=3):
    """Render up to n volumes; returns list of written paths. Raises on error (caller catches)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    gl = GridLoader(ENCODERS[encoder], split)
    vn = ["axial", "coronal", "sagittal"]
    written = []
    for vid in list(ids)[:n]:
        g = gl.load(vid)                                                     # PRODUCTION embedding loader
        T, H, W = g.shape[:3]
        organ = co.organ_vec(vid, split, g, encoder).reshape(T, H, W, -1)   # PRODUCTION mask loader
        emb = _views(_rs(np.linalg.norm(g - g[0, 0, 0], axis=-1)))
        mv = {nm: _views(_rs(organ[..., i])) for i, nm, _ in ORG}
        fig, ax = plt.subplots(1, 3, figsize=(11, 4.1))
        for c in range(3):
            a = ax[c]; a.imshow(emb[c], cmap="viridis", interpolation="bilinear"); a.set_xticks([]); a.set_yticks([])
            for i, nm, col in ORG:
                if mv[nm][c].max() > 0.25:
                    a.contour(mv[nm][c], levels=[0.3], colors=[col], linewidths=1.6)
            a.set_title(vn[c], fontsize=10, weight="bold")
        fig.legend(handles=[Line2D([0], [0], color=col, lw=2, label=nm) for _, nm, col in ORG],
                   loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(f"[canary] {encoder} {vid}: embedding structure + organ mask (production loaders)\n"
                     f"grid {T}x{H}x{W} — contours should track the embedding's anatomy", fontsize=10)
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        out = f"{out_dir}/{vid}.png"
        fig.savefig(out, dpi=110); plt.close(fig); written.append(out)
    return written
