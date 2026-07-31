import numpy as np
from .primitives import avgpack_r
from .branch_c.agglo_merge import _pos_encode


def avgpack(grid, budget=None, score=None, r=2, centroid=False,
            centroid_scale=1.0, centroid_encoding="raw", centroid_freqs=4):
    """Average each r^3 block -> ONE token (TRUE compression). [T,H,W,C] -> [N/r^3, C], channel-preserving.

    centroid=True appends the SAME position block ORCA gets, computed the SAME way, at the SAME scale.

    Why this option has to exist: every probe readout is permutation-invariant (MeanPool / AbmilPool /
    AbmilMultiPool all pool over tokens with weights computed from each token's own features -- no order term).
    Fixed-grid pooling carries its position ONLY in token ORDER, so the probe throws that away, while ORCA
    carries position INSIDE the feature vector. On the location family that is not a comparison between two
    compressors, it is a comparison between "has coordinates" and "has none", and the merge mechanism gets
    credited for the difference. Turning this on gives pooling the same coordinates, so the contrast isolates
    what the paper actually claims: adaptive REGION FORMATION vs a fixed grid.

    centroid=False is the default and returns bit-identical output to the previous version, so every result
    already on disk stays valid and comparable.
    """
    out = avgpack_r(grid, r)                                    # [N, C]
    if not centroid:
        return out
    T, H, W, C = grid.shape
    pt, ph, pw = (-T) % r, (-H) % r, (-W) % r                   # avgpack_r edge-pads; centroids must use the SAME padded grid
    Tp, Hp, Wp = T + pt, H + ph, W + pw
    gx, gy, gz = np.meshgrid(np.arange(Tp), np.arange(Hp), np.arange(Wp), indexing="ij")
    coords = np.stack([gx / Tp, gy / Hp, gz / Wp], -1).astype(np.float32)     # [Tp,Hp,Wp,3] normalized, same convention
    # block-mean of the coordinates == the region centroid, the exact analogue of agglo_merge's (M @ coords)/cnt
    cent = coords.reshape(Tp // r, r, Hp // r, r, Wp // r, r, 3).mean(axis=(1, 3, 5)).reshape(-1, 3)
    assert cent.shape[0] == out.shape[0], f"centroids {cent.shape[0]} != tokens {out.shape[0]}"
    pos = _pos_encode(cent, centroid_encoding, centroid_freqs)               # SHARED with ORCA -- not a re-implementation
    feat = grid.reshape(-1, C).astype(np.float32)
    return np.concatenate([out, pos * (centroid_scale * (float(feat.std()) + 1e-6))], axis=1).astype(np.float32)
