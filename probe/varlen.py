"""Variable-length token support -- an OPT-IN side path used ONLY by methods flagged `variable_len` in the
compressor REGISTRY (currently region_pool_r2_2d / MedRegion-CT R2). Fixed-length methods never touch this
file: their DataLoader keeps the default collate, and the abmil readouts get mask=None (bit-identical).

The two pieces that variable length needs:
  1. collate_varlen : pad a batch of [N_i, C] token sets to [B, Wmax, C] + a boolean mask [B, Wmax].
  2. the readouts (probe_model.py) apply that mask so padded slots get ZERO attention (a padded zero-vector
     would otherwise still absorb softmax mass and dilute the real tokens -> wrong pooling).

Deliberately assert-heavy: this is new code on the paper's core comparison, so we want a loud failure, not a
silently-wrong number.
"""
import torch


def collate_varlen(batch):
    """batch = list of (tok[N_i, C] (fp16), idx int) from ProbeSet.__getitem__.
    -> (padded[B, Wmax, C] fp16, mask[B, Wmax] bool, idx[B] long). Padded slots are 0 and mask=False."""
    toks = [b[0] for b in batch]
    idxs = [b[1] for b in batch]
    assert len(toks) > 0, "empty batch"
    assert all(t.dim() == 2 for t in toks), f"expected [N,C] per sample, got dims {[t.dim() for t in toks]}"
    C = toks[0].shape[1]
    assert all(t.shape[1] == C for t in toks), f"channel dim must match across batch, got {[t.shape[1] for t in toks]}"
    ns = [t.shape[0] for t in toks]
    assert all(n > 0 for n in ns), f"a sample has ZERO tokens (n={ns}) -- compressor produced nothing"
    B, W = len(toks), max(ns)
    out = torch.zeros(B, W, C, dtype=toks[0].dtype)
    mask = torch.zeros(B, W, dtype=torch.bool)
    for i, t in enumerate(toks):
        out[i, :ns[i]] = t
        mask[i, :ns[i]] = True
    idx = torch.tensor(idxs, dtype=torch.long)
    # loud invariants: every row has >=1 real token; mask bookkeeping matches exactly
    assert mask.any(dim=1).all(), "collate_varlen: a padded row has no real token"
    assert int(mask.sum()) == sum(ns), f"collate_varlen: mask count {int(mask.sum())} != real tokens {sum(ns)}"
    assert out.shape == (B, W, C) and mask.shape == (B, W) and idx.shape == (B,)
    return out, mask, idx
