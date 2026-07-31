#!/usr/bin/env python3
"""Index every finished probing run into one JSON so tables are filled from data, not from memory.

    python paper_compression/figures_code/scan_results.py   ->  paper_compression/results_index.json

For each results/experiments/<exp_id>/ it records the config that produced it (encoder, method, params,
budget) and, per family, the score at the best epoch chosen ON THE SEED MEAN -- never per-seed-then-averaged,
which is a max-over-noise and is what inflated an earlier version of this project's headline numbers.

Seed provenance is explicit because it has burned us: run.py used to truncate its CSV at every seed's first
epoch, so files that declared three seeds held one. A run counts as multi-seed only if the CSV carries a
`seed` column with >1 value, or a seeds_recovered.csv backfilled from the slurm logs exists.
"""
import json, re, sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results/experiments"
OUT = Path(__file__).resolve().parent.parent / "results_index.json"


def family_scores(d: Path, enc: str):
    """-> {family: {"mean":…, "sd":…, "n_seed":…, "epoch":…, "src":…}} at the best epoch on the seed mean."""
    out = {}
    rec = d / "seeds_recovered.csv"
    recdf = pd.read_csv(rec) if rec.exists() else None
    for f in sorted(d.glob("*.csv")):
        if f.name == "seeds_recovered.csv":
            continue
        fam = f.stem
        for pre in (f"{enc}_", "colipri_", "ct_clip_", "btb3d_", f"merlin_{enc}_"):
            if fam.startswith(pre):
                fam = fam[len(pre):]
                break
        fam = fam.replace("_merlin", "")
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or "metric" not in df:
            continue
        m = df[df.metric.isin(["r2", "auroc"])]
        if m.empty:
            continue
        # disease is macro-AUROC; regression families are the mean over their targets
        m = m[m.target == "macro"] if (fam == "disease" and (m.target == "macro").any()) else m[m.target != "micro"]
        src = "csv"
        if "seed" in m.columns and m.seed.nunique() > 1:
            per = m.groupby(["seed", "epoch"]).score.mean().reset_index()
        elif recdf is not None and fam in set(recdf.family.str.replace("_merlin", "", regex=False)):
            r = recdf[recdf.family.str.replace("_merlin", "", regex=False) == fam]
            per = r.rename(columns={"score": "score"})[["seed", "epoch", "score"]]
            src = "seeds_recovered"
        else:
            per = m.groupby("epoch").score.mean().reset_index().assign(seed=0)
        piv = per.pivot_table(index="epoch", columns="seed", values="score").dropna()
        if piv.empty:
            continue
        be = piv.mean(1).idxmax()
        v = piv.loc[be].values
        out[fam] = {"mean": round(float(v.mean()), 4),
                    "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                    "n_seed": int(len(v)), "epoch": int(be), "src": src}
    return out


rows = {}
for d in sorted(EXP.iterdir()):
    if not d.is_dir():
        continue
    cfg_f = d / "config.yaml"
    cfg = {}
    if cfg_f.exists():
        try:
            cfg = yaml.safe_load(cfg_f.read_text()) or {}
        except Exception:
            cfg = {}
    if not cfg:
        g = ROOT / "probing/experiments" / f"{d.name}.yaml"
        if g.exists():
            try:
                cfg = yaml.safe_load(g.read_text()) or {}
            except Exception:
                cfg = {}
    enc = cfg.get("encoder", "?")
    comp = cfg.get("compression", {}) or {}
    params = comp.get("params", {}) or {}
    fams = family_scores(d, enc.replace("merlin_", ""))
    if not fams:
        continue
    rows[d.name] = {
        "encoder": enc,
        "method": comp.get("method", "?"),
        "budget": comp.get("budget"),
        "r": params.get("r"),
        "lam": params.get("lam"),
        "centroid": bool(params.get("centroid", False)),
        "centroid_encoding": params.get("centroid_encoding"),
        "centroid_freqs": params.get("centroid_freqs"),
        "centroid_scale": params.get("centroid_scale"),
        "connectivity": params.get("connectivity"),
        "linkage": params.get("linkage"),
        "declared_seeds": (cfg.get("probe", {}) or {}).get("seeds"),
        "families": fams,
    }

OUT.write_text(json.dumps(rows, indent=1))
ns = sum(1 for v in rows.values() if any(f["n_seed"] > 1 for f in v["families"].values()))
print(f"indexed {len(rows)} experiments -> {OUT}")
print(f"  multi-seed: {ns}   single-seed: {len(rows)-ns}")
by = {}
for v in rows.values():
    by[v["encoder"]] = by.get(v["encoder"], 0) + 1
print("  by encoder:", dict(sorted(by.items(), key=lambda x: -x[1])))
