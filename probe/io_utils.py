"""Self-contained I/O + metrics for probe2 (vendored so probe2 has no scripts_probing dependency).
_strip / load_labels / metrics mirror scripts_probing; GridLoader replaces ScreenDataset for the
npy_grid encoders (btb3d/ct_clip/colipri)."""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def _strip(v):
    for suf in (".nii.gz", ".nii", ".npz", ".npy"):
        if v.endswith(suf):
            return v[: -len(suf)]
    return v


def load_labels(csv_path):
    """VolumeName,<cols> CSV -> {stripped_vid: float32[K]}. Blank/nan -> np.nan = per-target MISSING (a guard
    failed for that organ). Consumers MASK per column: regression stats/loss/R2 skip NaN cells. Rows are
    dropped upstream only when ALL targets are blank, so every vid here has >=1 present target. (Was 0.0 --
    wrong: 0.0 is a valid value AND biased the mean/std/R2 of guard-failed cells.)"""
    out = {}
    with open(csv_path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            out[_strip(row[0])] = np.array(
                [float(x) if x not in ("", "nan", "NaN", "NA") else np.nan for x in row[1:]], dtype=np.float32)
    return out


def apply_log1p(Y, cols, scale=1.0):
    """log1p(scale*x) on the given target columns (right-skewed zero-inflated fractions -> ~normal so R2 is
    meaningful; raw-space R2 is dominated by the 0-spike). `scale` matters: for TINY fractions (aorta_calc
    ~1e-3) plain log1p(x)~x is useless; scale=1000 spreads the positive tail (skew 6.4->1.7). Zeros stay 0.
    NaN-safe (log1p(nan)=nan); asserts non-negativity. Returns Y."""
    if Y is None or not cols:
        return Y
    Y = Y.copy()
    for c in cols:
        assert np.nanmin(Y[:, c]) >= 0.0, f"log1p target col {c} has negatives"
        Y[:, c] = np.log1p(scale * Y[:, c])
    return Y


def metrics(y, p):
    """Per-label auroc/ap/f1 as length-K arrays (NaN for single-class or unlabeled labels). Caller nanmeans.

    Scored per column over that label's FINITE cells only: NaN in y means "this volume has no ground truth
    for this label" (Merlin marks 77% of cells missing), which is not a negative and must not be scored.
    sklearn raises on NaN targets, so the mask is required, not defensive. All-finite y (CT-RATE) is
    unchanged."""
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    K = y.shape[1]
    auroc, ap, f1 = (np.full(K, np.nan) for _ in range(3))
    for j in range(K):
        mj = np.isfinite(y[:, j])
        if mj.sum() < 2:
            continue
        yj, pj = y[mj, j], p[mj, j]
        if yj.min() == yj.max():
            continue
        auroc[j] = roc_auc_score(yj, pj)
        ap[j] = average_precision_score(yj, pj)
        f1[j] = f1_score(yj, (pj > 0.5).astype(int), zero_division=0)
    return auroc, ap, f1


class GridLoader:
    """Per-volume encoder-grid loader. Exposes .ids and .load(vid) -> [T,H,W,C] float32.
    Two manifest artifact_types:
      - "npy_grid": one dense .npy per volume (ct_clip/colipri).
      - "codebook_grid": compact LFQ tokens (one big mmap'd int file per split) + a shared codebook.
        load(vid) = codebook[tokens_row.reshape(grid_shape)]. Avoids the per-file cold-read penalty on /orange
        (~7x faster cold; the whole tokens file is one mmap shared across workers via the OS page cache).
        `codebook_channel_reverse` reverses the 18 LFQ channels so the DATA-colocated (msb_identity) codebook
        yields the authoritative msb_reverse_channels features == the old dense_grid (verified exact, max|diff|=0)."""
    def __init__(self, manifest, split):
        m = json.loads(Path(manifest).read_text())
        self.kind = m.get("artifact_type")
        self.tr = m.get("axis_transpose")
        se = m["splits"][split]
        self.ids = [l.strip() for l in open(se["ids"]) if l.strip()]
        if self.kind == "npy_grid":
            self.dir = Path(se["dir"])
        elif self.kind == "codebook_grid":
            cb = np.load(m["codebook"])
            if m.get("codebook_channel_reverse"):
                cb = cb[:, ::-1]
            self.cb = np.ascontiguousarray(cb, dtype=np.float32)          # (K, C)
            self.grid_shape = tuple(m["grid_shape"])                       # e.g. (31,32,32)
            self.tokens = np.load(se["tokens"], mmap_mode="r")            # (N, prod(grid_shape)) int; row order == ids
            self.row = {v: i for i, v in enumerate(self.ids)}
        else:
            raise ValueError(f"GridLoader: unknown artifact_type {self.kind} ({manifest})")

    def load(self, vid):
        if self.kind == "npy_grid":
            arr = np.load(self.dir / f"{vid}.npy")
        else:
            tok = np.asarray(self.tokens[self.row[vid]]).reshape(self.grid_shape)
            arr = self.cb[tok]                                            # (D,H,W,C) == dense_grid
        if self.tr is not None:
            arr = np.transpose(arr, self.tr)
        return np.ascontiguousarray(arr, dtype=np.float32)
