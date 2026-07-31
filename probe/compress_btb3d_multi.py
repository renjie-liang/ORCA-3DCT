#!/usr/bin/env python3
"""Multi-budget BTB3D cache builder.

The slow part of the Ward compressors on BTB3D (31,744-token grid) is building the agglomerative tree.
That tree does NOT depend on the target budget (sklearn cuts the SAME full tree at n_clusters) and it is
seed-independent. So we build it ONCE per volume and cut it at every budget we need, instead of running a
separate O(N log N) Ward per (method, budget). Two trees suffice:

  Tree A  = agglo_merge features (raw btb3d embedding)                  -> ward / sin  @ B=512
  Tree B  = agglo_organ features ([embedding, sqrt(lam*fvar/ovar)*organ]) -> orca      @ B=27 / 216 / 512

Each cut is fed back through the UNCHANGED agglo_merge / agglo_organ pooling+position code via their new
`labels=` argument, so every written token array is bit-identical to what `co.apply(...)` produces (verified
in --verify mode). Output goes to the SHARED persistent cache the GPU trainer already reads:
{PROBE_CACHE_ROOT}/{exp_id}/{split}/{vid}.npy .

  Populate one shard:  PROBE_CACHE_ROOT=/.../cache python compress_btb3d_multi.py --num-shards 36 --shard-id K --split both
  Verify a few vols :  PROBE_CACHE_ROOT=/.../cache python compress_btb3d_multi.py --verify 5
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.cluster import ward_tree
from sklearn.feature_extraction.image import grid_to_graph
try:
    from sklearn.cluster._agglomerative import _hc_cut
except Exception:
    from sklearn.cluster._hierarchical_fast import _hc_cut

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import GridLoader                                            # noqa: E402
from config import ENCODERS                                               # noqa: E402
import compressors as co                                                  # noqa: E402
from run import filtered_ids, resolve_cache_root                          # noqa: E402
from compressors.branch_c.agglo_merge import agglo_merge                  # noqa: E402
from compressors.branch_c.agglo_organ import agglo_organ                  # noqa: E402

ENC = "btb3d"
LAM = 0.5
SINP = dict(centroid=True, centroid_encoding="sinusoidal", centroid_freqs=4, centroid_scale=2.0)

# (exp_id, tree, budget, builder-callable). tree "A" = merge feats, "B" = organ feats.
# The callable takes (grid, budget, organ, lab) and returns the token array, EXACTLY as the exp's yaml config.
CELLS = [
    ("exp_ablfill_ward_btb3d_b512", "A", 512, lambda g, b, o, l: agglo_merge(g, b, None, labels=l)),
    ("exp_ablfill_sin_btb3d_b512",  "A", 512, lambda g, b, o, l: agglo_merge(g, b, None, labels=l, **SINP)),
    ("exp_ablfill_orca_btb3d_b512", "B", 512, lambda g, b, o, l: agglo_organ(g, b, None, organ=o, lam=LAM, labels=l, **SINP)),
    ("exp_rt3_orca_btb3d_b216",     "B", 216, lambda g, b, o, l: agglo_organ(g, b, None, organ=o, lam=LAM, labels=l, **SINP)),
    ("exp_rt3_orca_btb3d_b27",      "B", 27,  lambda g, b, o, l: agglo_organ(g, b, None, organ=o, lam=LAM, labels=l, **SINP)),
]
# yaml-faithful direct configs, ONLY for --verify (recompute via co.apply and compare):
VERIFY_CFG = {
    "exp_ablfill_ward_btb3d_b512": ("agglo_merge", 512, {}),
    "exp_ablfill_sin_btb3d_b512":  ("agglo_merge", 512, SINP),
    "exp_ablfill_orca_btb3d_b512": ("agglo_organ", 512, dict(lam=LAM, **SINP)),
    "exp_rt3_orca_btb3d_b216":     ("agglo_organ", 216, dict(lam=LAM, **SINP)),
    "exp_rt3_orca_btb3d_b27":      ("agglo_organ", 27,  dict(lam=LAM, **SINP)),
}


def _two_trees(grid, organ):
    """Build tree A (merge feats) and tree B (organ feats) ONCE. Returns (childrenA, childrenB, n_leaves)."""
    T, H, W, C = grid.shape
    feat = grid.reshape(-1, C).astype(np.float32)
    conn = grid_to_graph(T, H, W)                                          # 6-neighbour, shared by both trees
    N = feat.shape[0]
    chA = ward_tree(feat, connectivity=conn, n_clusters=None)[0]
    org = organ.reshape(N, -1).astype(np.float32)
    fvar = float(feat.var(0).sum()); ovar = float(org.var(0).sum()) + 1e-9  # EXACTLY agglo_organ's augmentation
    aug = np.concatenate([feat, np.sqrt(LAM * fvar / ovar) * org], axis=1)
    chB = ward_tree(aug, connectivity=conn, n_clusters=None)[0]
    return chA, chB, N


def _labs(children, n_leaves, budgets):
    return {b: _hc_cut(b, children, n_leaves) for b in budgets}


def _atomic_save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr.astype(np.float32))
    os.replace(tmp, path)


def build_vid(gl, vid, split, roots):
    # resume fast-path: if every cell is already cached, skip BEFORE the expensive two-tree build
    if all((roots[exp_id] / split / f"{vid}.npy").exists() for exp_id, *_ in CELLS):
        return
    grid = gl.load(vid)
    organ = co.organ_vec(vid, split, grid, ENC)
    chA, chB, N = _two_trees(grid, organ)
    labA = _labs(chA, N, {512})
    labB = _labs(chB, N, {27, 216, 512})
    for exp_id, tree, budget, fn in CELLS:
        cpath = roots[exp_id] / split / f"{vid}.npy"
        if cpath.exists():
            continue
        lab = labA[budget] if tree == "A" else labB[budget]
        _atomic_save(cpath, fn(grid, budget, organ, lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--split", default="both", choices=["train", "valid", "both"])
    ap.add_argument("--verify", type=int, default=0, help="recompute N vols via co.apply and assert bit-identical, then exit")
    a = ap.parse_args()
    assert os.environ.get("PROBE_CACHE_ROOT", "").strip(), "need PROBE_CACHE_ROOT (shared cache path)"
    roots = {exp_id: Path(resolve_cache_root(exp_id)) for exp_id, *_ in CELLS}

    if a.verify:
        gl = GridLoader(ENCODERS[ENC], "valid")
        ids = filtered_ids(ENC, "valid", 0)[: a.verify]
        ok = True
        for vid in ids:
            grid = gl.load(vid); organ = co.organ_vec(vid, "valid", grid, ENC)
            chA, chB, N = _two_trees(grid, organ)
            labA = _labs(chA, N, {512}); labB = _labs(chB, N, {27, 216, 512})
            for exp_id, tree, budget, fn in CELLS:
                mine = fn(grid, budget, organ, (labA if tree == "A" else labB)[budget])
                method, bud, params = VERIFY_CFG[exp_id]
                ref = co.apply(method, grid, budget=bud, score=None, organ=organ, **params)
                same = mine.shape == ref.shape and np.array_equal(mine, ref)
                print(f"  {vid} {exp_id:32} shape {mine.shape} == co.apply : {same}", flush=True)
                ok = ok and same
        print(f"[verify] {'ALL BIT-IDENTICAL' if ok else 'MISMATCH -- DO NOT USE'}", flush=True)
        sys.exit(0 if ok else 1)

    splits = ["train", "valid"] if a.split == "both" else [a.split]
    ulist = os.environ.get("UNCACHED_LIST", "").strip()   # balanced resume: shard ONLY the uncached vids
    for sp in splits:
        gl = GridLoader(ENCODERS[ENC], sp)
        if ulist:
            lf = f"{ulist}.{sp}.txt"
            ids = [l.strip() for l in open(lf)] if os.path.exists(lf) else []
        else:
            ids = filtered_ids(ENC, sp, 0)
        mine = ids[a.shard_id::a.num_shards]
        print(f"[btb3d-multi] {sp} shard {a.shard_id}/{a.num_shards}: {len(mine)}/{len(ids)} vids (uncached-list={'yes' if ulist else 'no'})", flush=True)
        t0 = time.time(); n = 0
        for vid in mine:
            build_vid(gl, vid, sp, roots)
            n += 1
            if n % 25 == 0:
                print(f"[btb3d-multi] {sp} {n}/{len(mine)}  ({(time.time()-t0)/n:.1f}s/vol)", flush=True)
        print(f"[btb3d-multi] {sp} shard {a.shard_id}: DONE {n} vids in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
