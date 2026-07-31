#!/usr/bin/env python3
"""VQA accuracy vs token budget (appendix), parallel to the probing budget curves (fig_budget_colipri).

Backs the claim that VQA accuracy rises with the budget and keeps the same per-family method ordering as
probing. Best-validation-epoch accuracy at four budgets (8, 27, 64, 216) for the three compressors in
Table~\\ref{tab:textgen}: ORCA (full, 792-d), MedPruner-DINS, Grid average. One panel per family.

  -> AAAI_2027/Figures/fig_vqa_budget.pdf

Reads results_llm/s1sweep_b{8,27,64,216}/vqa_single_colipri_{method}_b{B}__{family}/
     *__s{1,2}/evaluations/step_*/vqa_metrics.json  ->  per_axis.{family}.acc  (single seed, best epoch).
MedPruner-DINS was run only at B=27 and B=216, so its curve has two points.
"""
import glob
import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "AAAI_2027" / "Figures"
SWEEP = ROOT / "results_llm"

METHODS = {
    "orcafull": ("ORCA (full)",    "#2a78d6", "-",  "o", 1.4),
    "dins":     ("MedPruner-DINS", "#008300", "-.", "^", 1.0),
    "avgpack":  ("Grid average",   "#6a6a6a", "--", "D", 1.0),
}
BUDGETS = [8, 27, 64, 216]
FAMS = ["size", "density", "location", "radiomics"]
FAM_NICE = {"radiomics": "texture"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.linewidth": 0.45, "axes.edgecolor": "#8a8a86",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def best_acc(method, B, fam):
    base = SWEEP / f"s1sweep_b{B}" / f"vqa_single_colipri_{method}_b{B}__{fam}"
    best = None
    for step in glob.glob(str(base / "*__s*" / "evaluations" / "step_*")):
        f = os.path.join(step, "vqa_metrics.json")
        if os.path.exists(f):
            a = json.load(open(f)).get("per_axis", {}).get(fam, {}).get("acc")
            if a is not None and (best is None or a > best):
                best = a
    return best


OUT.mkdir(parents=True, exist_ok=True)
slot = {b: i for i, b in enumerate(BUDGETS)}
fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.1))
for ax, fam in zip(axes.flatten(), FAMS):
    ax.grid(True, color="#ebeae6", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for m, (disp, col, ls, mk, lw) in METHODS.items():
        pts = [(slot[b], best_acc(m, b, fam)) for b in BUDGETS]
        pts = [(x, y) for x, y in pts if y is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls, color=col, linewidth=lw, marker=mk, markersize=2.8,
                markeredgecolor="white", markeredgewidth=0.35, label=disp, zorder=5, clip_on=False)
    ax.set_title(FAM_NICE.get(fam, fam), fontsize=7.5, color="#0b0b0b", pad=3)
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=6.2)
    ax.set_xlim(-0.35, len(BUDGETS) - 0.65)
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("tokens $B$", fontsize=7, color="#52514e", labelpad=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0, 0].set_ylabel("accuracy", fontsize=7.5, color="#52514e")
axes[1, 0].set_ylabel("accuracy", fontsize=7.5, color="#52514e")

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=6.2,
           bbox_to_anchor=(0.5, -0.01), handlelength=1.9, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.2, w_pad=0.5, h_pad=0.8)
fig.savefig(OUT / "fig_vqa_budget.pdf", dpi=300, bbox_inches="tight", pad_inches=0.015)
plt.close(fig)
print("wrote", OUT / "fig_vqa_budget.pdf")
