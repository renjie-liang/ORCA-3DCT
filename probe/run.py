#!/usr/bin/env python3
"""Config-driven probe runner. One YAML = one experiment: a compression method+params applied to an encoder's
grids, then EACH family probed with its OWN model (shared readout that predicts ALL its attributes for regress
families; abmil for disease). Cache-free: compression runs per-grid in the DataLoader workers (no token cache).

SHARED DATA PIPELINE (multi-family): the expensive part -- loading grids (~21MB each) + applying compression --
depends ONLY on (encoder, compression), NOT on the family. So for a run with N families we load+compress each
grid ONCE per epoch and feed the shared compressed tokens to N INDEPENDENT family models (own head, own optimizer,
own backward). This is NOT the old multi-task-mixing bug: there is no summed loss and no shared head across
families -- families stay statistically independent; we only amortize the grid I/O + compression (N x -> 1 x).

Regression = per-cell masked MSE + per-attribute masked R2 (NaN = target missing). Classify (disease) =
BCE(+pos_weight) + per-label/macro/micro AUROC. Raw per-volume preds saved EVERY epoch. Results ->
results/experiments/{exp_id}/.

  python run.py experiments/avgpack_r2_all5.yaml
"""
import argparse, csv, functools, os, sys, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from io_utils import GridLoader, load_labels, metrics, _strip, apply_log1p     # noqa: E402
from probe_model import READOUTS                                              # noqa: E402
from config import ENCODERS, FAMILIES, EXPECTED_N, FOREGROUND_BY_ENCODER, BAD_ORGAN_VIDS      # noqa: E402
import compressors as co                                                     # noqa: E402
from schema import load_config                                               # noqa: E402
from varlen import collate_varlen                                            # noqa: E402
import plots                                                                 # noqa: E402

DBG = bool(os.environ.get("DEBUG"))
_NOTQDM = bool(os.environ.get("TQDM_DISABLE"))


def _collate_for(method):   # variable_len methods (region_pool_r2_2d) -> pad+mask collate; else default (None)
    return collate_varlen if co.REGISTRY[method].get("variable_len", False) else None


@functools.lru_cache(maxsize=None)
def label_names(fam):
    with open(FAMILIES[fam]["valid"]) as f:
        return f.readline().strip().split(",")[1:]


def filtered_ids(encoder, split, limit):
    gl = GridLoader(ENCODERS[encoder], split)
    fg = FOREGROUND_BY_ENCODER[encoder]
    if fg is None:                                       # dataset has no foreground map (Merlin): the binding
        have = set(os.listdir(gl.dir))                   # constraint is just "does the grid exist"
    else:                                                # CT-RATE: an id is usable only if it also has an organ map
        have = set(os.listdir(Path(fg) / split))
    ids = [v for v in gl.ids if f"{_strip(v)}.npy" in have]
    assert len(ids) == EXPECTED_N[encoder][split], f"{encoder}/{split}: {len(ids)} vs {EXPECTED_N[encoder][split]}"
    bad = BAD_ORGAN_VIDS.get(encoder, set())               # drop volumes with a corrupt organ mask (assert above
    if bad:                                                 # still guards the RAW count -> dataset integrity intact)
        ids = [v for v in ids if v not in bad]
    return ids[:limit] if limit else ids


class ProbeSet(Dataset):                                          # load grid -> apply compression -> tokens [N',C']
    def __init__(self, encoder, split, ids, comp, cache_dir=None):
        self.gl = GridLoader(ENCODERS[encoder], split); self.ids = ids; self.split = split; self.encoder = encoder
        self.name = comp.method; self.budget = comp.budget; self.params = comp.params
        self.needs_score = co.REGISTRY[comp.method]["needs_score"]
        self.score_source = co.REGISTRY[comp.method].get("score_source", "foreground")
        self.needs_organ = co.REGISTRY[comp.method]["needs_organ"]
        self.cache_dir = None
        if cache_dir is not None:                                 # within-run compression cache (node-local /tmp)
            from pathlib import Path as _P
            self.cache_dir = _P(cache_dir) / split; self.cache_dir.mkdir(parents=True, exist_ok=True)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        vid = self.ids[i]
        cpath = self.cache_dir / f"{vid}.npy" if self.cache_dir is not None else None
        if cpath is not None and cpath.exists():                  # cached compression is deterministic -> reuse
            tok = np.load(cpath)
        else:
            grid = self.gl.load(vid)                              # [T,H,W,C]
            score = None
            if self.needs_score:                                  # foreground prior, or the encoder attention map (MedPruner-DINS)
                score = (co.attn_score if self.score_source == "attention" else co.foreground_score)(
                    vid, self.split, grid, self.encoder) if self.score_source == "attention" else \
                    co.foreground_score(vid, self.split, grid)
            organ = co.organ_vec(vid, self.split, grid, self.encoder) if self.needs_organ else None
            tok = co.apply(self.name, grid, budget=self.budget, score=score, organ=organ, **self.params)   # [N',C']
            if cpath is not None:
                import os
                tmp = f"{cpath}.{os.getpid()}.tmp"
                with open(tmp, "wb") as f: np.save(f, tok)        # file-obj save: no .npy suffix appended
                os.replace(tmp, cpath)                            # atomic; parallel workers write disjoint vids
        return torch.from_numpy(np.ascontiguousarray(tok)).half(), i                    # fp16 (halve transfer)


class ProbeModel(nn.Module):                                      # readout + head on the COMPRESSED tokens (no compressor)
    def __init__(self, d, n_out, readout, hid, heads, dropout, proj=None):
        super().__init__()
        self.proj = nn.Linear(d, proj) if proj else None          # optional dim reduction BEFORE readout (stabilizes high-dim)
        d_eff = proj or d
        self.innorm = nn.LayerNorm(d_eff, elementwise_affine=False)
        self.pool = READOUTS[readout](d_eff, hid, heads)
        self.head = nn.Sequential(nn.Linear(d_eff * self.pool.out_mult, hid), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hid, n_out))
    def forward(self, x, mask=None):
        if self.proj is not None:
            x = self.proj(x)
        return self.head(self.pool(self.innorm(x), mask))


def labels_family(ids, split, fam):
    lab = load_labels(FAMILIES[fam][split]); dim = len(next(iter(lab.values())))
    Y = np.full((len(ids), dim), np.nan, np.float32); mask = np.zeros(len(ids), bool)
    for i, v in enumerate(ids):
        k = _strip(v)
        if k in lab:
            Y[i] = lab[k]; mask[i] = True
    Y = apply_log1p(Y, FAMILIES[fam].get("log1p_cols", []), FAMILIES[fam].get("log1p_scale", 1.0))
    return Y, mask, dim


def masked_mse(out, yb):
    m = torch.isfinite(yb)
    return (((out - torch.nan_to_num(yb)) ** 2) * m).sum() / m.sum().clamp(min=1)


def masked_bce(out, yb, pos_weight):
    """Per-cell masked BCE. NaN = that (volume, label) is unlabeled, not a negative -- Merlin marks 77% of
    cells 'missing'. Plain BCEWithLogitsLoss over NaN targets yields a NaN loss, which the caller's
    `if torch.isfinite(loss)` guard then SILENTLY drops: training appears to run while learning nothing.
    Identical to reduction='mean' BCE when every cell is finite (CT-RATE), so `disease` is unaffected."""
    m = torch.isfinite(yb)
    l = F.binary_cross_entropy_with_logits(out, torch.nan_to_num(yb), pos_weight=pos_weight, reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1)


def per_target_r2(y, p):
    K = y.shape[1]; r2 = np.full(K, np.nan)
    for j in range(K):
        mj = np.isfinite(y[:, j])
        if mj.sum() < 2:
            continue
        yj, pj = y[mj, j], p[mj, j]
        r2[j] = 1 - ((pj - yj) ** 2).sum() / (((yj - yj.mean()) ** 2).sum() + 1e-9)
    return float(np.nanmean(np.clip(r2, -1, 1))), np.clip(r2, -1, 1)


def score_and_save(fam, p, y, task, norm, epoch, out_csv, va_ids, mva, seed, first_seed):
    """Given a family's valid preds `p` (over its labeled rows) + truth `y`: compute metrics, print, append the
    per-epoch CSV rows, and dump the raw per-volume preds npz. Shared by single- and multi-family paths."""
    rows = []; base = {"epoch": epoch, "seed": seed, "family": fam}
    if task == "classify":
        prob = 1 / (1 + np.exp(-p)); au, _, _ = metrics(y, prob)
        for j, nm in enumerate(label_names(fam)):
            rows.append({**base, "target": nm, "metric": "auroc", "score": (round(float(au[j]), 4) if np.isfinite(au[j]) else "")})
        mf = np.isfinite(y)                              # micro pools every (volume, label) cell into one ranking;
        yf, pf = y[mf].ravel(), prob[mf].ravel()         # unlabeled cells (Merlin's -1) must be dropped, not ranked
        micro = float(roc_auc_score(yf, pf)) if yf.size and yf.min() != yf.max() else float("nan")
        rows += [{**base, "target": "macro", "metric": "auroc", "score": round(float(np.nanmean(au)), 4)},
                 {**base, "target": "micro", "metric": "auroc", "score": (round(micro, 4) if np.isfinite(micro) else "")}]
        print(f"  [ep{epoch}] {fam:9s} macro={np.nanmean(au):.4f} micro={micro:.4f}", flush=True)
        ysave = p
    else:
        mu, sd = norm; mean_r2, per = per_target_r2(y, p * sd + mu)
        for j, v in enumerate(per):
            rows.append({**base, "target": j, "metric": "r2", "score": (round(float(v), 4) if np.isfinite(v) else "")})
        rows.append({**base, "target": "mean", "metric": "r2", "score": round(mean_r2, 4)})
        print(f"  [ep{epoch}] {fam:9s} r2_mean={mean_r2:.4f} per={np.round(per,3)}", flush=True)
        ysave = p * sd + mu
    pdir = Path(out_csv).parent / "preds"; pdir.mkdir(parents=True, exist_ok=True)
    tnames = label_names(fam) if task == "classify" else [str(j) for j in range(y.shape[1])]
    np.savez_compressed(pdir / f"{Path(out_csv).stem}_ep{epoch}.npz",
                        vids=np.array([va_ids[k] for k in np.nonzero(mva)[0]]),
                        y_true=y.astype(np.float32), y_pred=np.asarray(ysave, np.float32), targets=np.array(tnames))
    # Truncate ONLY at the very start of the run. The seed loop restarts epochs at 1, so keying the "w" on
    # epoch alone made seed 2 erase seed 1 and seed 3 erase seed 2 -- every "3-seed" CSV held ONE seed, with
    # no column to reveal it. Results looked complete; two thirds of the compute was silently discarded.
    mode = "w" if (epoch == 1 and first_seed) else "a"
    with open(out_csv, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


def build_family_state(cfg, fam, tr_ids, va_ids, dev, C, out_dir):
    """One family's independent probe: model + optimizer + labels + loss config. No coupling to other families."""
    P = cfg.probe
    Ytr, mtr, dim = labels_family(tr_ids, "train", fam)
    Yva, mva, _ = labels_family(va_ids, "valid", fam)
    task = FAMILIES[fam]["task"]
    model = ProbeModel(C, dim, FAMILIES[fam]["readout"], P.readout_dim, P.readout_heads, P.dropout, proj=P.proj).to(dev)
    opt = torch.optim.AdamW(model.parameters(), P.lr, weight_decay=P.weight_decay)
    st = {"fam": fam, "model": model, "opt": opt, "task": task, "dim": dim,
          "Ytr": Ytr, "mtr": mtr, "Yva": Yva, "mva": mva, "norm": None, "crit": None,
          "out_csv": str(Path(out_dir) / f"{cfg.encoder}_{fam}.csv")}
    if task == "classify":
        pos = np.nanmean(Ytr[mtr], 0).clip(1e-3, 1 - 1e-3)
        st["crit"] = torch.tensor(((1 - pos) / pos).clip(max=100.0), dtype=torch.float32, device=dev)  # pos_weight for masked_bce
    else:
        mu = np.nanmean(Ytr[mtr], 0); sd = np.nanstd(Ytr[mtr], 0) + 1e-6
        st["norm"] = (mu, sd)
        st["muT"] = torch.tensor(mu, dtype=torch.float32, device=dev)
        st["sdT"] = torch.tensor(sd, dtype=torch.float32, device=dev)
    print(f"[probing] fam={fam} dim={dim} token_dim={C} proj={P.proj} readout={FAMILIES[fam]['readout']} "
          f"train_labeled={int(mtr.sum())} valid_labeled={int(mva.sum())} -> {st['out_csv']}", flush=True)
    return st


def evaluate_all(states, dev, enc, comp, va_ids, P, epoch, seed, first_seed, cache_root=None):
    """ONE valid pass over the shared compressed tokens -> feed every family's head -> score+save each."""
    vl = DataLoader(ProbeSet(enc, "valid", va_ids, comp, cache_root), batch_size=P.bs, shuffle=False, num_workers=4,
                    pin_memory=True, collate_fn=_collate_for(comp.method))
    for st in states:
        st["model"].eval(); st["_pred"] = []
    with torch.no_grad():
        for batch in tqdm(vl, desc=f"eval ep{epoch}", disable=_NOTQDM):
            if len(batch) == 3:
                xb, maskb, _ = batch; mdev = maskb.to(dev)
            else:
                xb, _ = batch; mdev = None
            xdev = xb.to(dev).float()
            for st in states:
                st["_pred"].append(st["model"](xdev, mdev).cpu().numpy())
    for st in states:
        p_full = np.concatenate(st["_pred"]); mva = st["mva"]
        score_and_save(st["fam"], p_full[mva], st["Yva"][mva], st["task"], st["norm"], epoch, st["out_csv"], va_ids, mva,
                       seed, first_seed)


def resolve_cache_root(exp_id):
    """Compression-cache dir (compressed tokens are deterministic in (encoder, compression), so cache once + reuse).
    PROBE_CACHE_ROOT set -> SHARED persistent cache {ROOT}/{exp_id}: survives across the compress-shard jobs AND the
      later train job, never wiped -> lets many cheap CPU jobs pre-populate the cache that the GPU trainer consumes.
    unset -> node-local /tmp/probecache/{exp_id}, wiped at start + on exit (the single-job default; unchanged).
    PROBE_NO_CACHE=1 -> None (compress every epoch, no cache)."""
    import os, shutil, atexit
    if os.environ.get("PROBE_NO_CACHE", "0") == "1":
        return None
    shared = os.environ.get("PROBE_CACHE_ROOT", "").strip()
    if shared:
        root = f"{shared}/{exp_id}"
        os.makedirs(root, exist_ok=True)                          # persistent: NO wipe, NO atexit-delete (shards depend on it)
        print(f"[probing] compression cache (SHARED persistent): {root}", flush=True)
        return root
    root = f"/tmp/probecache/{exp_id}"
    shutil.rmtree(root, ignore_errors=True)                       # node-local: fresh each run, no stale-cache risk
    atexit.register(lambda: shutil.rmtree(root, ignore_errors=True))
    print(f"[probing] compression cache (node-local /tmp, wiped on exit): {root}", flush=True)
    return root


def compress_shard(cfg, num_shards, shard_id, which_split):
    """Populate the SHARED cache for ids[shard_id::num_shards], then exit -- NO training. CPU-bound compressors
    (btb3d connectivity-Ward, ~9s/vol) fan out over many cheap CPU jobs this way; the GPU trainer later reads an
    all-hit cache. Idempotent: an already-cached vid is loaded (not recomputed), so a re-run only fills gaps."""
    cache_root = resolve_cache_root(cfg.exp_id)
    assert cache_root and os.environ.get("PROBE_CACHE_ROOT", "").strip(), \
        "compress-only needs PROBE_CACHE_ROOT (a shared path); node-local /tmp cannot be shared across jobs"
    enc, comp = cfg.encoder, cfg.compression
    splits = ["train", "valid"] if which_split == "both" else [which_split]
    for sp in splits:
        limit = cfg.data.limit if sp == "train" else cfg.data.valid_limit
        ids = filtered_ids(enc, sp, limit)
        mine = ids[shard_id::num_shards]                          # strided: every shard sees the same id list, disjoint slices
        ds = ProbeSet(enc, sp, mine, comp, cache_root)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers, collate_fn=lambda b: 0)
        print(f"[compress] {cfg.exp_id} {sp} shard {shard_id}/{num_shards}: {len(mine)}/{len(ids)} vids "
              f"-> {cache_root}/{sp}", flush=True)
        n = 0
        for _ in dl:                                              # __getitem__ side effect = atomic per-vid cache write
            n += 1
            if n % 100 == 0: print(f"[compress] {sp} {n}/{len(mine)}", flush=True)
        print(f"[compress] {sp} shard {shard_id}: DONE {n} vids", flush=True)


def train_all_families(cfg, fams, tr_ids, va_ids, dev, out_dir):
    """Shared data pipeline: load+compress each grid ONCE per epoch, train N independent family models on it."""
    enc, comp, P = cfg.encoder, cfg.compression, cfg.probe
    # token dim from one compressed sample (identical across families for a fixed encoder+compression)
    g0 = GridLoader(ENCODERS[enc], "train").load(tr_ids[0])
    sc0 = None                                                    # dim probe: dispatch score by source, same as ProbeSet.__getitem__
    if co.REGISTRY[comp.method]["needs_score"]:                   # (attention encoders have no foreground map -> must not hardcode foreground_score)
        _ss = co.REGISTRY[comp.method].get("score_source", "foreground")
        sc0 = co.attn_score(tr_ids[0], "train", g0, enc) if _ss == "attention" else co.foreground_score(tr_ids[0], "train", g0)
    C = co.apply(comp.method, g0, budget=comp.budget, score=sc0, **comp.params).shape[-1]
    print(f"[probing] SHARED pipeline: {len(fams)} families {fams} comp={comp.method}{comp.params} token_dim={C}", flush=True)

    cache_root = resolve_cache_root(cfg.exp_id)                    # shared-persistent (PROBE_CACHE_ROOT) or node-local /tmp

    states = None
    for seed in P.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        states = [build_family_state(cfg, fam, tr_ids, va_ids, dev, C, out_dir) for fam in fams]
        collate = _collate_for(comp.method)                       # None (fixed-len) or pad+mask collate (variable_len)
        loader = DataLoader(ProbeSet(enc, "train", tr_ids, comp, cache_root), batch_size=P.bs, shuffle=True, drop_last=True,
                            num_workers=cfg.data.num_workers, pin_memory=True, persistent_workers=False,
                            prefetch_factor=cfg.data.prefetch, collate_fn=collate)
        for ep in range(P.epochs):
            for st in states:
                st["model"].train()
            t0 = time.time(); _printed = False
            for batch in tqdm(loader, desc=f"train ep{ep+1}/{P.epochs}", disable=_NOTQDM):
                if len(batch) == 3:                               # variable_len path: (tokens, mask, idx)
                    xb, maskb, idx = batch; mdev = maskb.to(dev)
                else:                                             # fixed-len path: (tokens, idx) -- mask=None, bit-identical
                    xb, idx = batch; mdev = None
                idx = idx.numpy(); xdev = xb.to(dev).float()      # shared compressed tokens for this batch
                if mdev is not None and not _printed:             # loud one-time check per epoch that padding looks sane
                    real = mdev.sum(1)
                    print(f"[varlen] ep{ep+1} batch: padded W={xdev.shape[1]}, real/vol min/mean/max="
                          f"{int(real.min())}/{real.float().mean():.0f}/{int(real.max())}, batch={xdev.shape[0]}", flush=True)
                    assert int(real.min()) > 0, "a volume has 0 real tokens"
                    _printed = True
                for st in states:                                 # each family: independent forward/backward/step
                    sel = st["mtr"][idx]
                    if sel.sum() < 4:
                        continue
                    selt = torch.from_numpy(sel)
                    x = xdev[selt]; m = mdev[selt] if mdev is not None else None
                    yb = torch.tensor(st["Ytr"][idx][sel], device=dev)
                    st["opt"].zero_grad(); out = st["model"](x, m)
                    if st["task"] == "regress":
                        yb = (yb - st["muT"]) / st["sdT"]; loss = masked_mse(out, yb)
                    else:
                        loss = masked_bce(out, yb, st["crit"])
                    if torch.isfinite(loss):
                        loss.backward(); nn.utils.clip_grad_norm_(st["model"].parameters(), 1.0); st["opt"].step()
            print(f"[probing] shared ep{ep+1}/{P.epochs}: {time.time()-t0:.0f}s ({len(states)} families)", flush=True)
            evaluate_all(states, dev, enc, comp, va_ids, P, ep + 1, seed, seed == P.seeds[0], cache_root)
    for st in states:
        plots.plot_family(out_dir, enc, st["fam"], st["task"])    # auto-diagnostics (best epoch)
        print(f"[probing] DONE {st['fam']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compress-only", action="store_true", help="populate the SHARED cache for one shard, then exit (no training)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--split", default="both", choices=["train", "valid", "both"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.compress_only:                                              # CPU fan-out phase: fill cache, no GPU/model/out_dir needed
        compress_shard(cfg, a.num_shards, a.shard_id, a.split)
        print("[compress] shard DONE", flush=True)
        return
    dev = torch.device(a.device)
    out_dir = ROOT / "results" / "experiments" / cfg.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copy(a.config, out_dir / "config.yaml")     # self-describing: config next to results
    tr_ids = filtered_ids(cfg.encoder, "train", cfg.data.limit)
    va_ids = filtered_ids(cfg.encoder, "valid", 0)                    # valid ALWAYS full
    print(f"[probing] exp={cfg.exp_id} enc={cfg.encoder} comp={cfg.compression.method} fams={cfg.probe.families} "
          f"train={len(tr_ids)} valid={len(va_ids)} -> {out_dir}", flush=True)
    if os.environ.get("PROBE_SANITY_VIZ", "1") != "0":               # per-run alignment canary (emb+mask, production loaders)
        try:                                                         # NON-FATAL by design: a diagnostic viz must never abort a run
            from sanity_viz import render_sanity
            w = render_sanity(cfg.encoder, "valid", va_ids, str(out_dir / "sanity"), n=3)
            print(f"[sanity_viz] wrote {len(w)} alignment canary images -> {out_dir}/sanity", flush=True)
        except Exception as e:
            print(f"[sanity_viz] skipped (non-fatal): {type(e).__name__}: {e}", flush=True)
    train_all_families(cfg, cfg.probe.families, tr_ids, va_ids, dev, str(out_dir))
    print("[probing] DONE all families", flush=True)


if __name__ == "__main__":
    main()
