#!/usr/bin/env python3
"""Build the F1-F3 (budget) and F4 (lambda) figure-data JSONs from results_index.json -- one source of truth.

    python paper_compression/figures_code/scan_results.py       # refresh results_index.json first
    python paper_compression/figures_code/build_figdata.py      # -> results_llm/budget_curve_data.json + lambda_data.json

Every cell is [best-epoch-on-seed-mean mean, sd] straight from the index (see scan_results.py for the convention).
The figure scripts (fig_budget_curve.py, fig_lambda.py) only PLOT these files -- no numbers live in plot code.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDX = json.load(open(ROOT / "paper_compression" / "results_index.json"))
OUTB = ROOT / "results_llm" / "budget_curve_data.json"
OUTL = ROOT / "results_llm" / "lambda_data.json"

# enc-token in exp_id -> (corpus key, families in display order, merlin-suffix on family keys)
CORPUS = {
    "colipri": ("ctrate_colipri", ["size", "density", "location", "radiomics"], ""),
    "suprem":  ("merlin_suprem",  ["size", "density", "location"],              "_merlin"),
    "segvol":  ("merlin_segvol",  ["size", "density", "location"],              "_merlin"),
}
TITLE = {"ctrate_colipri": "CT-RATE / colipri", "merlin_suprem": "Merlin / SuPreM",
         "merlin_segvol": "Merlin / SegVol"}
CORPORA = {CORPUS[e][0]: CORPUS[e] for e in CORPUS}     # unique corpus key -> (key, families, suffix)


def cell(fams, fam):
    """[mean, sd] for a family in an index row, sd->0.0 if single-seed. None if the family is absent."""
    if fam not in fams:
        return None
    r = fams[fam]
    return [r["mean"], r["sd"] if r["sd"] is not None else 0.0]


# ---------- F1-F3: budget curves (exp_bc3_{method}_{enc}_b{budget}) ----------
RB = re.compile(r"^exp_bc3_(orca|avgpack|tome|dins)_(colipri|suprem|segvol)_b(\d+)$")
budget = {ck: {"families": [f + suf for f in fl], "data": {}}
          for ck, fl, suf in CORPORA.values()}
for name, row in IDX.items():
    m = RB.match(name)
    if not m:
        continue
    method, enc, b = m.group(1), m.group(2), m.group(3)
    ck, fl, suf = CORPUS[enc]
    d = budget[ck]["data"].setdefault(method, {})
    cells = {f + suf: cell(row["families"], f) for f in fl}
    d[b] = {k: v for k, v in cells.items() if v is not None}

# ---------- F4: lambda sweep (exp_lam3_{enc}_lam{val}_b{budget}) ----------
RL = re.compile(r"^exp_lam3_(colipri|suprem|segvol)_lam(0p5|10000|\d+)_b\d+$")
LAM = {"0": "0", "0p5": "0.5", "2": "2", "8": "8", "32": "32", "10000": "10000"}
lam = {ck: {"title": TITLE[ck], "families": [f + suf for f in fl], "data": {}}
       for ck, fl, suf in CORPORA.values()}
for name, row in IDX.items():
    m = RL.match(name)
    if not m:
        continue
    enc, lraw = m.group(1), m.group(2)
    ck, fl, suf = CORPUS[enc]
    lam[ck]["data"][LAM[lraw]] = {f + suf: cell(row["families"], f)
                                  for f in fl if cell(row["families"], f) is not None}

OUTB.write_text(json.dumps(budget, indent=1))
OUTL.write_text(json.dumps(lam, indent=1))
for tag, obj in (("budget", budget), ("lambda", lam)):
    print(f"=== {tag} ===")
    for ck, v in obj.items():
        got = sorted(v["data"].keys())
        print(f"  {ck:16s} fams={v['families']} keys={got}")
print(f"\nwrote {OUTB.relative_to(ROOT)} + {OUTL.relative_to(ROOT)}")
