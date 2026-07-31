"""Per-family readout probes + train/eval for the probe2 sweep.

DENSE only: X is an in-RAM fp16 array [N, W, C] (loaded once by stage2, see load_cache); each batch is
sliced in-RAM and cast to float32 on the fly (GPU sees one batch at a time). Ragged/var_n methods
(effrank) are skipped, so no padding/mask path is needed.

Readouts ([B,W,C] tokens -> pooled vector); choice is per family (config.FAMILIES[f]["readout"]):
  mean_linear : mean-pool                     -> [B,C]        global-aggregate tasks (metadata)
  abmil       : gated attention-MIL, 1 head   -> [B,C]        multi-label detection (disease)
  abmil_multi : gated attention-MIL, M heads  -> [B,M*C]      regression (strongest + perm-invariant)
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import metrics  # noqa: E402   -> (auroc, ap, f1) per label


# All readouts take an OPTIONAL mask[B,W] bool (True = real token). mask=None (every fixed-length method) is
# the ORIGINAL code path, bit-identical. mask given (variable_len methods, see varlen.py) zeros out padded
# slots -- for abmil by setting their pre-softmax logits to -inf (a padded zero-vector would otherwise steal
# softmax mass from real tokens and dilute the pooled vector).
def _mask_logits(logits, mask):                              # logits[B,W,K], mask[B,W] -> logits with pad=-inf
    if mask is None:
        return logits
    assert mask.shape == logits.shape[:2], f"mask {tuple(mask.shape)} vs logits {tuple(logits.shape[:2])}"
    return logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))


class MeanPool(nn.Module):                                   # permutation-invariant mean
    def __init__(self, d, hid, heads): super().__init__(); self.out_mult = 1
    def forward(self, x, mask=None):                          # [B,W,C] -> [B,C]
        if mask is None:
            return x.mean(1)
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp(min=1)


class AbmilPool(nn.Module):                                  # gated attention-MIL (Ilse 2018), 1 focus
    def __init__(self, d, hid, heads):
        super().__init__(); self.out_mult = 1
        self.V = nn.Linear(d, hid); self.U = nn.Linear(d, hid); self.w = nn.Linear(hid, 1)
    def forward(self, x, mask=None):                          # [B,W,C] -> [B,C]
        att = torch.softmax(_mask_logits(self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))), mask), dim=1)
        return (att * x).sum(1)


class AbmilMultiPool(nn.Module):                             # M gated-attention heads (M foci) -> concat
    def __init__(self, d, hid, heads):
        super().__init__(); self.out_mult = heads
        self.V = nn.Linear(d, hid); self.U = nn.Linear(d, hid); self.w = nn.Linear(hid, heads)
    def forward(self, x, mask=None):                         # [B,W,C] -> [B, M*C]
        att = torch.softmax(_mask_logits(self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))), mask), dim=1)  # [B,W,M]
        return torch.einsum("bwm,bwc->bmc", att, x).reshape(x.shape[0], -1)


READOUTS = {"mean_linear": MeanPool, "abmil": AbmilPool, "abmil_multi": AbmilMultiPool}


class Probe(nn.Module):
    def __init__(self, d, n_out, readout, hid=256, heads=4, dropout=0.1):
        super().__init__()
        self.innorm = nn.LayerNorm(d, elementwise_affine=False)
        self.pool = READOUTS[readout](d, hid, heads)
        self.head = nn.Sequential(nn.Linear(d * self.pool.out_mult, hid), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hid, n_out))

    def forward(self, x):
        return self.head(self.pool(self.innorm(x)))


def _iter(X, idx, Y, bs, shuffle):
    """Yield (xb float32 [B,W,C], yb [B,n_out]) by slicing the in-RAM fp16 array (X[idx[sel]]) and
    casting to float32 -- only the batch is float32; X itself stays fp16."""
    order = np.random.permutation(len(idx)) if shuffle else np.arange(len(idx))
    for k in range(0, len(order), bs):
        sel = order[k:k + bs]
        if shuffle and len(sel) < bs:
            break                                            # drop_last -> stable training
        yield np.asarray(X[idx[sel]], dtype=np.float32), Y[sel]


def train_eval(Xtr, idx_tr, Ytr, Xva, idx_va, Yva, task, readout, dev,
               epochs=8, lr=1e-3, wd=1e-4, bs=64, seed=2026, hid=256, heads=4, dropout=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Probe(Xtr.shape[-1], Ytr.shape[1], readout, hid, heads, dropout).to(dev)
    if task == "classify":
        pos = Ytr.mean(0).clip(1e-3, 1 - 1e-3)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(((1 - pos) / pos).clip(max=100.0),
                                                            dtype=torch.float32, device=dev))
    else:
        mu, sd = np.nanmean(Ytr, 0), np.nanstd(Ytr, 0) + 1e-6   # per-target stats over PRESENT cells only (skip NaN)
        muT = torch.tensor(mu, dtype=torch.float32, device=dev); sdT = torch.tensor(sd, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)

    best, bestm = -1e9, {}
    for _ in range(epochs):
        model.train()
        for xb, yb in _iter(Xtr, idx_tr, Ytr, bs, True):
            x = torch.from_numpy(xb).to(dev); y = torch.from_numpy(yb).to(dev)
            opt.zero_grad(); pred = model(x)
            if task == "regress":
                y = (y - muT) / sdT
                msk = torch.isfinite(y)                          # per-cell mask: NaN = target missing for this vol
                loss = (((pred - torch.nan_to_num(y)) ** 2) * msk).sum() / msk.sum().clamp(min=1)   # masked MSE
            else:
                loss = crit(pred, y)
            if torch.isfinite(loss):
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); preds = []
        with torch.no_grad():
            for xb, _ in _iter(Xva, idx_va, Yva, bs, False):
                preds.append(model(torch.from_numpy(xb).to(dev)).cpu().numpy())
        out = np.concatenate(preds)
        if task == "classify":
            au, ap, f1 = metrics(Yva, 1.0 / (1.0 + np.exp(-out)))
            score, m_ = float(np.nanmean(au)), {"auroc": float(np.nanmean(au)), "macro_f1": float(np.nanmean(f1))}
        else:
            ys = (Yva - mu) / sd; p = np.nan_to_num(out)     # ys keeps NaN where a target is missing (guard fail)
            K = ys.shape[1]; r2 = np.full(K, np.nan)
            for j in range(K):                               # per-target R2 over ITS present valid rows only
                mj = np.isfinite(ys[:, j])
                if mj.sum() < 2:
                    continue
                yj, pj = ys[mj, j], p[mj, j]
                r2[j] = 1 - ((pj - yj) ** 2).sum() / (((yj - yj.mean()) ** 2).sum() + 1e-9)
            r2 = np.clip(r2, -1.0, 1.0)                       # floor degenerate/near-constant columns (else R2 explodes)
            score, m_ = float(np.nanmean(r2)), {"r2": float(np.nanmean(r2))}
        if score > best:
            best, bestm = score, m_
    return bestm
