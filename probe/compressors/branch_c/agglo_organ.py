import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from .agglo_merge import _pos_encode


def agglo_organ(grid, budget, score, organ=None, lam=0.5, centroid=False,
                centroid_scale=1.0, centroid_encoding="raw", centroid_freqs=4, extent=False, labels=None):
    """ORCA-Organ: the SAME Ward budget-reallocating engine as agglo_merge, with a SOFT organ prior.

    We do NOT change Ward's distance (that would drop its variance-minimizing / budget-reallocation property).
    Instead we AUGMENT each token's feature with its lambda-scaled organ-occupancy vector and run the identical
    Ward: feat_aug = [ embedding_C , sqrt(lam)*s*organ_11 ], where s = embedding's own std (so lam is a scale-free
    embedding:organ ratio). Ward's within-cluster variance now also penalizes merging cells of different organ
    composition -> embedding-DOMINANT, organ-GUIDED. Output = per-region mean of the ORIGINAL embedding.

    lam=0 (or organ=None) is EXACTLY agglo_merge (= ORCA-base). Soft-by-design: if the encoder embedding does not
    align with the organ map (ct_clip/btb3d), the embedding term dominates and the organ prior degrades gracefully.
    """
    T, H, W, C = grid.shape
    feat = grid.reshape(-1, C).astype(np.float32)
    if labels is not None:                                           # tree precomputed once, cut per budget (btb3d builder)
        lab = np.asarray(labels)
    else:
        if organ is None or lam <= 0:
            aug = feat
        else:
            org = organ.reshape(feat.shape[0], -1).astype(np.float32)    # [N,K] occupancy in [0,1]
            # scale the K-dim organ BLOCK so its TOTAL variance = lam * feature total variance. Per-dim scaling would
            # drown K=11 organ dims against C=512-768 feature dims (organ weight ~ K*lam/C ~ 0.4%); balancing by total
            # variance makes lam a meaningful organ:feature importance ratio (lam=1 -> organ as important as ALL feats).
            fvar = float(feat.var(0).sum()); ovar = float(org.var(0).sum()) + 1e-9
            aug = np.concatenate([feat, np.sqrt(lam * fvar / ovar) * org], axis=1)
        conn = grid_to_graph(T, H, W)                                # 6-neighbour voxel adjacency
        lab = AgglomerativeClustering(n_clusters=budget, linkage="ward", connectivity=conn).fit_predict(aug)
    N = feat.shape[0]
    M = sparse.csr_matrix((np.ones(N, np.float32), (lab, np.arange(N))), shape=(budget, N))
    cnt = np.asarray(M.sum(1)).ravel()
    means = (M @ feat) / np.maximum(cnt[:, None], 1.0)             # [budget, C] region means (original embedding)
    if centroid:
        # Same position re-injection as agglo_merge (shared `_pos_encode`), so ORCA-Organ and ORCA-base differ
        # ONLY by the organ prior lam. Without this the lambda axis could never be tested against the tuned
        # sinusoidal setting -- the two halves of the method had no configuration in common.
        gx, gy, gz = np.meshgrid(np.arange(T), np.arange(H), np.arange(W), indexing="ij")
        coords = np.stack([gx.ravel() / T, gy.ravel() / H, gz.ravel() / W], 1).astype(np.float32)
        cent = (M @ coords) / np.maximum(cnt[:, None], 1.0)
        blocks = [_pos_encode(cent, centroid_encoding, centroid_freqs)]
        if extent:
            var = (M @ (coords ** 2)) / np.maximum(cnt[:, None], 1.0) - cent ** 2
            ext = np.sqrt(np.maximum(var, 0.0))
            blocks.append(ext / (ext.std(0, keepdims=True) + 1e-6))
        pos = np.concatenate(blocks, axis=1)
        means = np.concatenate([means, pos * (centroid_scale * (float(feat.std()) + 1e-6))], axis=1)
    return means.astype(np.float32)
