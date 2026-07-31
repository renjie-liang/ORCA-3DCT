#!/usr/bin/env python3
"""Summarise the CT-RATE VQA sweep: 3 arms x 4 budgets x 4 families, from each cell's best.json.

Arms are avgpack / orcabase / orcasin. The legacy `orca` cells are agglo_ORGAN lam=0.5 with no position
block -- a different method -- so they are listed separately and never mixed into the comparison.
"""
import json, glob, re, sys
from pathlib import Path

R = Path("./results_llm")
FAMS = ["density", "radiomics", "size", "location"]
ARMS = ["avgpack", "orcabase", "orcasin"]
best = {}
for f in glob.glob(str(R / "s1sweep_b*" / "*" / "best.json")):
    m = re.search(r"vqa_single_colipri_(\w+?)_b(\d+)__(\w+)__s(\d)", f)
    if not m:
        continue
    arm, b, fam, stage = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    try:
        best[(arm, b, fam, stage)] = json.load(open(f)).get("best_acc")
    except Exception:
        pass

for stage in ("1", "2"):
    rows = [(a, b) for a in ARMS for b in (8, 27, 64, 216)
            if any((a, b, fm, stage) in best for fm in FAMS)]
    if not rows:
        continue
    print(f"\n===== stage {stage} =====")
    print(f"{'arm/budget':18s} " + " ".join(f"{x[:9]:>10s}" for x in FAMS))
    for a, b in rows:
        v = [best.get((a, b, fm, stage)) for fm in FAMS]
        print(f"{a + ' b' + str(b):18s} " + " ".join(f"{x:10.4f}" if x is not None else f"{'--':>10s}" for x in v))
leg = {k: v for k, v in best.items() if k[0] == "orca"}
if leg:
    print(f"\n(legacy `orca` = agglo_organ lam=0.5, NOT part of the comparison: {len(leg)} cells)")
done = len({k[:3] for k in best if k[3] == '1'})
print(f"\ns1 cells with best.json: {done}/48")
