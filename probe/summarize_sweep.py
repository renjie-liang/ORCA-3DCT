#!/usr/bin/env python3
"""Scan results/experiments/exp_*/ and print a compression-sweep trend table: one row per experiment
(method, tokens, encoder), columns = the 5 families (disease macro-AUROC; size/density/location/radiomics
mean R2 at best epoch). Incomplete runs show what's done so far. Run anytime:  python summarize_sweep.py"""
import glob, os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
EXPD = ROOT / "results" / "experiments"
FAMS = ["disease", "size", "density", "location", "radiomics"]


def best(csv):
    c = pd.read_csv(csv); c["score"] = pd.to_numeric(c["score"], errors="coerce")
    fam = None
    agg = "macro" if (c["target"].astype(str) == "macro").any() else "mean"
    a = c[c["target"].astype(str) == agg].dropna(subset=["score"])
    if a.empty:
        return None, None
    be = int(a.loc[a["score"].idxmax(), "epoch"])
    return float(a["score"].max()), be


def main():
    rows = []
    for d in sorted(glob.glob(str(EXPD / "exp_*"))):
        cfgp = Path(d) / "config.yaml"
        if not cfgp.exists():
            continue
        cfg = yaml.safe_load(open(cfgp))
        method = cfg["compression"]["method"]
        enc = cfg["encoder"]
        r = cfg["compression"].get("params", {}).get("r")
        budget = cfg["compression"].get("budget")
        tok = f"r{r}" if r else f"b{budget}"
        row = {"exp": os.path.basename(d).split("_all5")[0], "method": method, "enc": enc, "tok": tok}
        maxep = 0
        for fam in FAMS:
            csv = Path(d) / f"{enc}_{fam}.csv"
            if csv.exists():
                s, be = best(csv)
                row[fam] = f"{s:.3f}" if s is not None else "-"
                maxep = max(maxep, be or 0)
            else:
                row[fam] = "-"
        row["ep"] = maxep                                    # progress: best epoch seen (10 = done)
        rows.append(row)
    if not rows:
        print("no experiments yet"); return
    df = pd.DataFrame(rows)[["exp", "method", "enc", "tok", "ep"] + FAMS]
    df = df.sort_values(["method", "enc", "tok"])
    print("disease=macro AUROC | size/density/location/radiomics=mean R2 (best epoch). ep=best-epoch reached (10=done).\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
