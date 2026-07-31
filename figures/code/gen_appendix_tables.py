#!/usr/bin/env python3
"""Generate the per-attribute appendix tables from the LABEL FILES.

    python paper_compression/figures_code/gen_appendix_tables.py
    -> paper_compression/appendix_tables.tex     (\\input{} from TABLES_AND_FIGURES.tex)

Attribute names are READ FROM THE LABEL CSVs, never transcribed. Hand-typed names drift silently from what
was actually probed, and the appendix is precisely where nobody re-checks. When a target is added or renamed,
re-run this; the tables follow.

Layout: ATTRIBUTES ARE ROWS, methods are columns. The transpose was tried and reverted -- 15-30 rotated
column heads needed splitting and were hard to scan.

Cells are emitted as \\NUM placeholders. Fill them by extending this script to read
results/experiments/<exp_id>/<encoder>_<family>.csv once the runs exist, so the numbers are generated too and
cannot drift from the names.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "probing"))
import pandas as pd
from config import FAMILIES

ap = argparse.ArgumentParser()
ap.add_argument("--twocolumn", action="store_true",
                help="emit table* spanning floats (for the AAAI paper). Default: plain table, which is what "
                     "the one-column planning file needs -- table* there is silently dropped.")
TWOCOL = ap.parse_args().twocolumn

OUT = Path(__file__).resolve().parent.parent / "appendix_tables.tex"
ID_COLS = {"vid", "volumename", "id", "split", "study id"}      # 'study id' hides in the Merlin disease file

METHODS = ["Average pooling", "Slice pooling", "DivPrune", "MedPruner-DINS", "ToMe",
           "R2 token pooling", "ORCA"]
# Short column heads: the full names are uneven in height as \shortstack and waste width. The caption of
# each table carries the full names once, so the abbreviation costs nothing.
HEAD = {"Average pooling": "Avg.\\ pool", "Slice pooling": "Slice pool", "DivPrune": "DivPrune",
        "MedPruner-DINS": "MedPruner", "ToMe": "ToMe", "R2 token pooling": "MedRegion",
        "ORCA": "\\textbf{ORCA}"}

def targets(fam):
    df = pd.read_csv(FAMILIES[fam]["valid"], nrows=1)
    return [c for c in df.columns if c.lower() not in ID_COLS]

def esc(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

# --- filling from result CSVs -----------------------------------------------------------------------------
# per-family per-target scores at the best epoch (on the seed mean), keyed by the label-column order which the
# probing writer preserves (target index j == the j-th label column). Cells with no run stay \NUM.
import glob, os

def _read(exp, enc, fam):
    """{target_label: (mean, sd, n_seed)} at best epoch on the seed mean, or {} if the run is absent. The label
    is the CSV `target` string verbatim: an integer "0".."4" for regression families, a finding NAME for
    disease. table() resolves an attribute to its label by name first, then by positional index."""
    for suf in ("_merlin", ""):
        f = f"{PROOT}/results/experiments/{exp}/{enc}_{fam}{suf}.csv"
        if os.path.exists(f):
            break
    else:
        return {}
    m = pd.read_csv(f)
    m = m[m.metric.isin(["r2", "auroc"])]
    m = m[~m.target.isin(["mean", "macro", "micro"])]
    out = {}
    for t, g in m.groupby("target"):
        if "seed" in g.columns and g.seed.nunique() > 1:
            piv = g.groupby(["seed", "epoch"]).score.mean().reset_index().pivot(
                index="epoch", columns="seed", values="score").dropna()
            be = piv.mean(1).idxmax(); v = piv.loc[be].values
            out[str(t)] = (v.mean(), v.std(ddof=1), len(v))
        else:
            be = g.groupby("epoch").score.mean().idxmax()
            out[str(t)] = (g[g.epoch == be].score.mean(), None, 1)
    return out

PROOT = str(Path(__file__).resolve().parents[2])

def _fmt(cell):
    if cell is None:
        return r"\NUM"
    mean, sd, n = cell
    if n >= 3 and sd is not None:
        return rf"\FIN{{{mean:.3f}}}{{{sd:.3f}}}"
    return rf"\PROV{{{mean:.3f}}}{{{n}}}"

def table(label, caption, groups, wide=False, family_col=True, enc=None, exp_by_method=None, attr_offset=0):
    r"""groups: [(family_name_used_as_fam_key, [attribute, ...]), ...]

    ATTRIBUTES ARE ROWS, methods are columns. Cells are filled from result CSVs when `enc` and
    `exp_by_method` are given: exp_by_method maps a METHODS entry -> exp_id (or None). Attribute index j maps
    to the j-th label column, which is the order probing writes per-target scores in. Missing runs stay \NUM.

    `attr_offset` shifts the CSV target index when a family's attributes are printed across several tables
    (e.g. A6b shows findings 15..29, but they are rows 0..14 of its slice -- offset restores the global index).

    `wide` picks the float environment (table* only in two-column mode; silently dropped in one-column).
    """
    env = "table*" if (wide and TWOCOL) else "table"
    lead = "ll" if family_col else "l"
    L = [rf"\begin{{{env}}}[t]\centering", r"\setlength{\tabcolsep}{4pt}\footnotesize",
         r"\begin{tabular}{" + lead + "c" * len(METHODS) + "}", r"\toprule"]
    head = (["family", "attribute"] if family_col else ["attribute"]) + [HEAD[m] for m in METHODS]
    L.append(" & ".join(head) + r" \\")
    L.append(r"\midrule")
    # pre-read each method's scores for each family present in `groups`
    cache = {}
    if enc and exp_by_method:
        for g, _ in groups:
            for meth, exp in exp_by_method.items():
                cache[(g, meth)] = _read(exp, enc, g) if exp else {}
    for gi, (g, atts) in enumerate(groups):
        if gi and family_col:
            L.append(r"\addlinespace[2pt]")
        for ai, a in enumerate(atts):
            cells = []
            for meth in METHODS:
                sc = cache.get((g, meth), {})
                key = a if a in sc else str(ai + attr_offset)   # disease: match by finding name; regression: by index
                cells.append(_fmt(sc.get(key)))
            row = ([rf"\multirow{{{len(atts)}}}{{*}}{{{g}}}" if ai == 0 else ""] if family_col else []) \
                  + [rf"\texttt{{\footnotesize {esc(a)}}}"] + cells
            L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{\textbf{{{label}}} {caption}}}", rf"\end{{{env}}}", ""]
    return "\n".join(L)


# method -> exp_id per table. None = no run (stays \NUM). CT-RATE colipri and Merlin use different runs/budgets.
CT_B216 = {"Average pooling": "exp_bl_avgpack_colipri_b216", "Slice pooling": "exp_bl_slice_colipri",
           "DivPrune": "exp_bl_divprune_colipri_b216",
           "MedPruner-DINS": "exp_bl_dins_colipri_b216", "ToMe": "exp_bl_tome_colipri_b216",
           "R2 token pooling": "exp_bl_r2faithful_colipri", "ORCA": "exp_co_organ_lam0p5_b216"}
CT_B27 = {**CT_B216, "Average pooling": "exp_bl_avgpack_colipri_b27",
          "ToMe": "exp_bl_tome_colipri_b27", "DivPrune": "exp_bl_divprune_colipri_b27",
          "MedPruner-DINS": "exp_bl_dins_colipri_b27", "ORCA": "exp_co_organ_lam0p5_b27",
          "R2 token pooling": None}                                # R2 is fixed ~205 tokens, only comparable at b216
SUPREM = {"Average pooling": "exp_msd_suprem_avg_b216", "Slice pooling": None, "DivPrune": None,
          "MedPruner-DINS": None, "ToMe": None, "R2 token pooling": None, "ORCA": "exp_msd_suprem_lam2_b216"}
SEGVOL = {"Average pooling": "exp_msd_segvol_avg_b256", "Slice pooling": None, "DivPrune": None,
          "MedPruner-DINS": None, "ToMe": None, "R2 token pooling": None, "ORCA": "exp_msd_segvol_lam2_b256"}
# Merlin disease is a SEPARATE per-family run (the regression exps omit it), so A6 needs its own mapping to the
# *_disease exp_ids. ORCA = agglo_organ lam=2 + f4s2 sinusoidal position (the complete method, 3-seed).
SUPREM_DIS = {"Average pooling": "exp_msd_suprem_avg_b216_disease", "Slice pooling": None, "DivPrune": None,
              "MedPruner-DINS": None, "ToMe": None, "R2 token pooling": None,
              "ORCA": "exp_msd_suprem_lam2_b216_disease"}


ct_reg = [("size", targets("size")), ("density", targets("density")),
          ("location", targets("location")), ("radiomics", targets("radiomics"))]
me_reg = [("size", targets("size_merlin")), ("density", targets("density_merlin")),
          ("location", targets("location_merlin"))]
ct_dis = targets("disease")
me_dis = targets("disease_merlin")

parts = ["% AUTO-GENERATED by figures_code/gen_appendix_tables.py -- do not hand-edit.",
         f"% float environment: {'table*' if TWOCOL else 'table'} "
         f"(regenerate with --twocolumn for the two-column AAAI paper).",
         "% Attribute names come from the label CSVs; re-run after any target change.", ""]

parts.append(table("A1.", r"CT-RATE / colipri, per-attribute probing $R^2$ at $B{=}27$. "
                   r"Cells are mean over three seeds with the sd on the last digit, e.g.\ $0.870(2)$.",
                   ct_reg, wide=True, enc="colipri", exp_by_method=CT_B27))
parts.append(table("A2.", r"CT-RATE / colipri at $B{=}216$; same attributes as A1.", ct_reg, wide=True,
                   enc="colipri", exp_by_method=CT_B216))
parts.append(table("A3.", r"Merlin / SuPreM at $B{=}216$. Merlin has no texture family: the texture targets "
                   r"are lung-based and Merlin is abdominal CT. Size carries six targets because three "
                   r"organs are measured in both absolute and vertebra-ratio form.", me_reg, wide=True,
                   enc="merlin_suprem", exp_by_method=SUPREM))
parts.append(table("A4.", r"Merlin / SegVol at $B{=}256$; same attributes as A3. The budget differs from A3 "
                   r"because the encoders' token grids differ ($8{\times}16{\times}16$ vs.\ $12^3$).",
                   me_reg, wide=True, enc="merlin_segvol", exp_by_method=SEGVOL))
parts.append(table("A5.", r"CT-RATE per-finding macro-AUROC at $B{=}216$. One budget only: disease is flat "
                   r"across budgets, so a second panel would repeat 18 columns of numbers.",
                   [("disease", ct_dis)], wide=True, family_col=False, enc="colipri", exp_by_method=CT_B216))
half = (len(me_dis) + 1) // 2
parts.append(table("A6a.", rf"Merlin per-finding macro-AUROC at $B{{=}}216$ (SuPreM), findings 1--{half} of "
                   rf"{len(me_dis)}. Split across two tables because {len(me_dis)} rotated columns do not fit "
                   r"one page width. Labels are missing-not-at-random --- a finding is recorded mostly when "
                   r"present --- so these are conditional on being labelled, not population prevalence.",
                   [("disease", me_dis[:half])], wide=True, family_col=False, enc="merlin_suprem", exp_by_method=SUPREM_DIS))
parts.append(table("A6b.", rf"Merlin per-finding macro-AUROC, findings {half+1}--{len(me_dis)}.",
                   [("disease", me_dis[half:])], wide=True, family_col=False, enc="merlin_suprem", exp_by_method=SUPREM_DIS,
                   attr_offset=half))

OUT.write_text("\n".join(parts))
print(f"wrote {OUT}")


def _splice_into_main():
    r"""Keep the inline appendix in TABLES_AND_FIGURES.tex in sync automatically. The main file carries a pasted
    copy of these tables between two sentinels; without this, regenerating appendix_tables.tex silently left the
    paper file stale (which happened -- days of fills never reached the paper). Replace between the sentinels
    with the freshly generated table blocks (everything from the first \begin{table}), so 'generate' == 'synced'."""
    main = OUT.parent / "TABLES_AND_FIGURES.tex"
    BEG, END = "% <<<APPENDIX_AUTO_BEGIN>>>", "% <<<APPENDIX_AUTO_END>>>"
    lines = main.read_text().split("\n")
    b = next((i for i, l in enumerate(lines) if l.startswith(BEG)), None)
    e = next((i for i, l in enumerate(lines) if l.startswith(END)), None)
    if b is None or e is None:
        print(f"  [splice] SKIPPED: sentinels not found in {main.name} ({BEG} / {END})")
        return
    block = parts[next(i for i, p in enumerate(parts) if p.startswith(r"\begin{table}")):]
    while block and block[-1].strip() == "":
        block.pop()
    lines = lines[:b + 1] + block + lines[e:]           # keep BEGIN line, new blocks, keep END line onward
    text = "\n".join(lines)
    main.write_text(text)
    print(f"  [splice] synced {len(block)} table blocks into {main.name} between sentinels")


_splice_into_main()
print(f"  CT-RATE regression {sum(len(a) for _, a in ct_reg)} attrs | disease {len(ct_dis)}")
print(f"  Merlin  regression {sum(len(a) for _, a in me_reg)} attrs | disease {len(me_dis)} (split {half}+{len(me_dis)-half})")
