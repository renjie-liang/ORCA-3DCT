#!/usr/bin/env python3
"""Pre-generate the held-out test split as durable id-list files (run ONCE).

    python gen_heldout_split.py            # colipri + merlin_suprem + merlin_segvol, K=2000 each

The held-out volumes are a deterministic subset of each encoder's TRAIN volumes (seed-fixed, disjoint from the
rest of train). They are TRAIN volumes -- their embeddings/labels live under the `train` split -- so run_heldout.py
loads them with split="train"; nothing is duplicated. Persisting the split (vs. re-splitting in code each run)
makes the holdout reproducible, inspectable, and IDENTICAL across every method/budget/seed that reads it.

Writes probing/heldout_ids/{encoder}_{trainho,heldout}.txt (one vid per line, sorted).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from run import filtered_ids                                        # reuse the exact usable-id logic (grid + mask + bad-vid filter)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heldout_ids")
ENCODERS = ["colipri", "merlin_suprem", "merlin_segvol"]
K = 2000                                                            # held-out volumes per encoder
SEED = 20260722                                                     # fixed -> reproducible split forever


def gen(enc):
    ids = sorted(filtered_ids(enc, "train", 0))                    # sort first: order-independent of listdir()
    assert len(ids) > K, f"{enc}: only {len(ids)} train ids, cannot hold out {K}"
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ids))
    ho = sorted(ids[i] for i in perm[:K])
    trainho = sorted(ids[i] for i in perm[K:])
    assert not (set(ho) & set(trainho)) and len(ho) + len(trainho) == len(ids)
    os.makedirs(OUT, exist_ok=True)
    for name, lst in (("heldout", ho), ("trainho", trainho)):
        p = os.path.join(OUT, f"{enc}_{name}.txt")
        open(p, "w").write("\n".join(lst) + "\n")
        print(f"  {enc:14s} {name:8s}: {len(lst):6d} vids -> {p}")
    return ho


hos = {}
for enc in ENCODERS:
    hos[enc] = gen(enc)
# sanity: merlin_suprem and merlin_segvol should hold out the SAME Merlin volumes (same vid pool, same seed)
if set(hos["merlin_suprem"]) == set(hos["merlin_segvol"]):
    print("[ok] Merlin SuPreM/SegVol hold out identical volumes (consistent Merlin holdout)")
else:
    n = len(set(hos["merlin_suprem"]) & set(hos["merlin_segvol"]))
    print(f"[note] Merlin SuPreM/SegVol holdout overlap = {n}/{K} (their train vid pools differ)")
print(f"\nDONE. K={K} per encoder, seed={SEED}.")
