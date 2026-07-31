#!/usr/bin/env python3
"""Held-out test runner: train on train' (train minus a pre-generated holdout), evaluate on BOTH the official
valid split AND the held-out volumes. Confirms the ORCA-vs-baseline ranking is a property of the representation,
not an artifact of the validation split / probe overfitting to val.

    python gen_heldout_split.py                          # once: writes heldout_ids/{enc}_{trainho,heldout}.txt
    python run_heldout.py experiments/<exp>.yaml         # trains on trainho, evals valid + heldout

REUSES run.py wholesale (ProbeSet, build_family_state, score_and_save, the train step, compression cache) -- the
main pipeline is untouched. The held-out volumes are TRAIN volumes, so they load with split="train"; nothing is
duplicated. Per family it writes <enc>_<fam>.csv (valid) and <enc>_<fam>_heldout.csv (held-out); both carry the
same epoch,seed,... schema, so scan_results / the raw-CSV verifier read them identically.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import run as R                                                    # every primitive comes from the main pipeline
from schema import load_config
from config import ENCODERS

HO_DIR = Path(__file__).resolve().parent / "heldout_ids"


def read_ids(enc, name):
    p = HO_DIR / f"{enc}_{name}.txt"
    assert p.exists(), f"missing split file {p} -- run gen_heldout_split.py first"
    return [l.strip() for l in open(p) if l.strip()]


def evaluate_heldout(states, dev, enc, comp, ho_ids, P, epoch, seed, first_seed, cache_root):
    """Mirror run.evaluate_all, but eval the HELD-OUT ids (loaded from the train split) into <fam>_heldout.csv."""
    dl = DataLoader(R.ProbeSet(enc, "train", ho_ids, comp, cache_root), batch_size=P.bs, shuffle=False,
                    num_workers=4, pin_memory=True, collate_fn=R._collate_for(comp.method))
    for st in states:
        st["model"].eval(); st["_hpred"] = []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"heldout ep{epoch}", disable=R._NOTQDM):
            xb, mdev = (batch[0], batch[1].to(dev)) if len(batch) == 3 else (batch[0], None)
            xdev = xb.to(dev).float()
            for st in states:
                st["_hpred"].append(st["model"](xdev, mdev).cpu().numpy())
    for st in states:
        p_full = np.concatenate(st["_hpred"]); mho = st["mho"]
        R.score_and_save(st["fam"], p_full[mho], st["Yho"][mho], st["task"], st["norm"], epoch,
                         st["out_csv_ho"], ho_ids, mho, seed, first_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device(a.device)
    enc, comp, P = cfg.encoder, cfg.compression, cfg.probe
    fams = cfg.probe.families

    tr_ids = read_ids(enc, "trainho")                              # train' = train minus holdout
    ho_ids = read_ids(enc, "heldout")                              # held-out test (train volumes)
    va_ids = R.filtered_ids(enc, "valid", 0)                       # official valid, unchanged
    out_dir = R.ROOT / "results" / "experiments" / cfg.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copy(a.config, out_dir / "config.yaml")
    print(f"[heldout] exp={cfg.exp_id} enc={enc} comp={comp.method} fams={fams} "
          f"trainho={len(tr_ids)} heldout={len(ho_ids)} valid={len(va_ids)} -> {out_dir}", flush=True)

    # token dim (dispatch score by source, identical to run.train_all_families)
    g0 = R.GridLoader(ENCODERS[enc], "train").load(tr_ids[0])
    sc0 = None
    if R.co.REGISTRY[comp.method]["needs_score"]:
        ss = R.co.REGISTRY[comp.method].get("score_source", "foreground")
        sc0 = R.co.attn_score(tr_ids[0], "train", g0, enc) if ss == "attention" else R.co.foreground_score(tr_ids[0], "train", g0)
    C = R.co.apply(comp.method, g0, budget=comp.budget, score=sc0, **comp.params).shape[-1]
    print(f"[heldout] token_dim={C} comp={comp.method}{comp.params}", flush=True)

    cache_root = R.resolve_cache_root(cfg.exp_id)
    for seed in P.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        states = [R.build_family_state(cfg, fam, tr_ids, va_ids, dev, C, out_dir) for fam in fams]
        for st in states:                                          # attach held-out labels + its own csv (train-split labels)
            Yho, mho, _ = R.labels_family(ho_ids, "train", st["fam"])
            st["Yho"], st["mho"] = Yho, mho
            st["out_csv_ho"] = st["out_csv"].replace(".csv", "_heldout.csv")
            print(f"[heldout] fam={st['fam']} heldout_labeled={int(mho.sum())} -> {Path(st['out_csv_ho']).name}", flush=True)
        collate = R._collate_for(comp.method)
        loader = DataLoader(R.ProbeSet(enc, "train", tr_ids, comp, cache_root), batch_size=P.bs, shuffle=True,
                            drop_last=True, num_workers=cfg.data.num_workers, pin_memory=True,
                            persistent_workers=False, prefetch_factor=cfg.data.prefetch, collate_fn=collate)
        first = seed == P.seeds[0]
        for ep in range(P.epochs):
            for st in states:
                st["model"].train()
            t0 = time.time()
            for batch in tqdm(loader, desc=f"train ep{ep+1}/{P.epochs}", disable=R._NOTQDM):
                if len(batch) == 3:
                    xb, maskb, idx = batch; mdev = maskb.to(dev)
                else:
                    xb, idx = batch; mdev = None
                idx = idx.numpy(); xdev = xb.to(dev).float()
                for st in states:
                    sel = st["mtr"][idx]
                    if sel.sum() < 4:
                        continue
                    selt = torch.from_numpy(sel)
                    x = xdev[selt]; m = mdev[selt] if mdev is not None else None
                    yb = torch.tensor(st["Ytr"][idx][sel], device=dev)
                    st["opt"].zero_grad(); out = st["model"](x, m)
                    if st["task"] == "regress":
                        yb = (yb - st["muT"]) / st["sdT"]; loss = R.masked_mse(out, yb)
                    else:
                        loss = R.masked_bce(out, yb, st["crit"])
                    if torch.isfinite(loss):
                        loss.backward(); nn.utils.clip_grad_norm_(st["model"].parameters(), 1.0); st["opt"].step()
            print(f"[heldout] seed{seed} ep{ep+1}/{P.epochs}: {time.time()-t0:.0f}s", flush=True)
            R.evaluate_all(states, dev, enc, comp, va_ids, P, ep + 1, seed, first, cache_root)      # -> <fam>.csv (valid)
            evaluate_heldout(states, dev, enc, comp, ho_ids, P, ep + 1, seed, first, cache_root)    # -> <fam>_heldout.csv
    print("[heldout] DONE all families", flush=True)


if __name__ == "__main__":
    main()
