import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph


def _conn_graph(T, H, W, connectivity):
    """Voxel adjacency for the Ward merge. connectivity in {6,18,26} = face / +edge / +corner neighbours;
    None (or 0) = no spatial constraint (global feature clustering -- the spatial-adjacency ablation).
    6 uses sklearn's grid_to_graph; 18/26 build the extra edges by shifted index pairs (symmetric)."""
    if connectivity in (None, 0, "none"):
        return None
    if connectivity == 6:
        return grid_to_graph(T, H, W)
    offs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]                                  # 3 face axes (both signs via symmetry)
    if connectivity in (18, 26):
        offs += [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]   # 12 edge
    if connectivity == 26:
        offs += [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]             # 8 corner
    if connectivity not in (18, 26):
        raise ValueError(f"connectivity must be 6/18/26/None, got {connectivity!r}")
    idx = np.arange(T * H * W).reshape(T, H, W)
    rows, cols = [], []
    for dt, dh, dw in offs:
        a = idx[max(dt, 0):T + min(dt, 0), max(dh, 0):H + min(dh, 0), max(dw, 0):W + min(dw, 0)]
        b = idx[max(-dt, 0):T + min(-dt, 0), max(-dh, 0):H + min(-dh, 0), max(-dw, 0):W + min(-dw, 0)]
        rows.append(a.ravel()); cols.append(b.ravel())
    r = np.concatenate(rows); c = np.concatenate(cols)
    n = T * H * W
    g = sparse.coo_matrix((np.ones(r.size, np.uint8), (r, c)), shape=(n, n)).tocsr()
    return (g + g.T).astype(bool).astype(np.uint8)                           # symmetric adjacency


_STD_FLOOR = 0.10   # binds ONLY at degenerate tiny budgets; every reported budget (>=27) has std > this


def _pos_encode(cent, encoding, freqs):
    """Turn region centroids [budget,3] in [0,1] into the position feature block, per-dim RMS-normalized to 1
    so `centroid_scale` means the same thing across encodings.
      raw        : the 3 coordinates themselves.
      sinusoidal : sin/cos of the 3 coords at `freqs` geometric frequencies (transformer-style) -> 6*freqs dims,
                   a basis a LINEAR projector can read position off far more easily than raw coords.

    The per-dim RMS is estimated from only `budget` regions, so at TINY budgets it is both unreliable and can
    approach 0 (4 regions have nearly the same centroid in some axis). Dividing by it then blows the position
    block up to ~30x the feature scale, so 24 position dims drown 768 content dims -- measured as ORCA+sin
    LOSING to avgpack at b4/b8 on Merlin, and producing NaN when a dim's std was exactly 0. Flooring the
    divisor fixes that failure mode. The floor is set BELOW the empirical std at every budget we actually
    report (>=27), so those results are bit-identical; it binds only in the degenerate small-budget regime."""
    if encoding == "raw":
        blk = cent
    elif encoding == "sinusoidal":
        fs = (2.0 ** np.arange(freqs)) * np.pi                      # pi, 2pi, 4pi, ...
        ang = cent[:, :, None] * fs[None, None, :]                  # [budget,3,freqs]
        blk = np.concatenate([np.sin(ang), np.cos(ang)], axis=2).reshape(cent.shape[0], -1)  # [budget,6*freqs]
    else:
        raise ValueError(f"unknown centroid_encoding {encoding!r}")
    # keep the original `+1e-6` so that whenever the floor does NOT bind the divisor is EXACTLY the
    # pre-fix one -- every budget we report (>=27) stays bit-identical; only the degenerate regime moves.
    return blk / np.maximum(blk.std(0, keepdims=True) + 1e-6, _STD_FLOOR)


def agglo_merge(grid, budget, score, standardize=False, centroid=False,
                centroid_scale=1.0, centroid_encoding="raw", centroid_freqs=4, extent=False,
                connectivity=6, linkage="ward", labels=None):
    """Distortion-optimal region merge (the content-adaptive, VARIABLE-size counterpart of avgpack). Start from
    all N tokens on the T*H*W grid (6-connectivity), then greedily merge the adjacent region-pair that adds the
    LEAST within-region L2 variance (Ward linkage) until `budget` regions remain. Unlike SLIC (equal-size, edge-
    snapping), this truly REALLOCATES budget: homogeneous background collapses into a few large regions; complex
    anatomy (vessels/lesions/boundaries) stays fine. Output = per-region MEAN feature (L2-optimal summary).

    Ward + spatial connectivity is O(N log N)-ish (sklearn heap); exact `budget` output tokens (channel-preserving).

    standardize=True z-scores each feature dim BEFORE Ward (cluster on the whitened geometry; still aggregate the
    ORIGINAL feature). Ward's L2 variance is otherwise dominated by high-variance dims -> on low-effective-dim
    encoders (ct_clip: ~18% eff dims) raw Ward clusters by a loud subspace and degrades. Standardization equalizes
    dim weight. Neutral on balanced encoders (colipri: ~85% eff dims). Clustering-only; output tokens unchanged.

    centroid=True appends each region's normalized centroid (x,y,z), scaled to the feature RMS, to its mean ->
    output [budget, C+3]. The region-mean discards absolute position (irregular partition has no fixed grid
    lattice), which is why merging loses on POSITION attributes; the centroid restores it (training-free).
    """
    T, H, W, C = grid.shape
    feat = grid.reshape(-1, C).astype(np.float32)
    fclust = (feat - feat.mean(0)) / (feat.std(0) + 1e-6) if standardize else feat
    if labels is not None:                                         # tree precomputed once, cut per budget (btb3d multi-budget builder)
        lab = np.asarray(labels)
    else:
        conn = _conn_graph(T, H, W, connectivity)                  # 6/18/26-neighbour adjacency, or None = unconstrained
        lab = AgglomerativeClustering(n_clusters=budget, linkage=linkage, connectivity=conn).fit_predict(fclust)
    N = feat.shape[0]
    M = sparse.csr_matrix((np.ones(N, np.float32), (lab, np.arange(N))), shape=(budget, N))
    cnt = np.asarray(M.sum(1)).ravel()
    means = (M @ feat) / np.maximum(cnt[:, None], 1.0)             # [budget, C] region means
    if centroid:
        gx, gy, gz = np.meshgrid(np.arange(T), np.arange(H), np.arange(W), indexing="ij")
        coords = np.stack([gx.ravel() / T, gy.ravel() / H, gz.ravel() / W], 1).astype(np.float32)  # normalized (T,H,W) order
        cent = (M @ coords) / np.maximum(cnt[:, None], 1.0)        # [budget, 3] region centroids in [0,1]
        blocks = [_pos_encode(cent, centroid_encoding, centroid_freqs)]
        if extent:
            # per-region spread: RMS distance of member voxels from their centroid, per axis. Captures how
            # DISTRIBUTED a region is (a small compact IVC vs a lung sprawling L-R), which a single centroid
            # cannot. A distributed attribute like lung_LR balance might need this where the centroid fails.
            var = (M @ (coords ** 2)) / np.maximum(cnt[:, None], 1.0) - cent ** 2
            ext = np.sqrt(np.maximum(var, 0.0))                    # [budget,3]
            blocks.append(ext / (ext.std(0, keepdims=True) + 1e-6))
        pos = np.concatenate(blocks, axis=1)                       # per-dim unit-RMS position features
        # centroid_scale sets position energy RELATIVE to the feature RMS, so the projector weighs the two
        # comparably. scale too small -> position ignored; too large -> the 768 content dims get drowned.
        means = np.concatenate([means, pos * (centroid_scale * (float(feat.std()) + 1e-6))], axis=1)
    return means.astype(np.float32)
