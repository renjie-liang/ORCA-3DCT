#!/usr/bin/env python3
"""VQA two-stage per-epoch accuracy curves (appendix diagnostic).

Companion to fig_reportgen_2stage.py. The report-generation curves showed the LLM is still climbing at the
end of s2 (under-trained) on the near-saturated clinical metric. This figure checks the SAME thing for the
discriminative downstream task, VQA: is per-family accuracy converged by the end of training, or still rising?

VQA is trained per family (size / density / location / texture) as a separate two-stage run: s1 projector
warmup (8 epochs) then s2 LoRA + projector (8 epochs). We plot per-family accuracy per epoch across both
stages for the three compressors that appear in Table~\\ref{tab:textgen}: ORCA (full, 792-d), MedPruner-DINS,
Grid average. Rows are the two token budgets (27, 216); columns are the four families.

  -> AAAI_2027/Figures/fig_vqa_2stage.pdf

Reads results_llm/s1sweep_b{27,216}/vqa_single_colipri_{method}_b{B}__{family}/
     *__s{1,2}/evaluations/step_*/vqa_metrics.json  ->  per_axis.{family}.acc  (single seed).
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

# method -> (display, color, linestyle, marker, linewidth)
METHODS = {
    "orcafull": ("ORCA (full)",     "#2a78d6", "-",  "o", 1.5),
    "dins":     ("MedPruner-DINS",  "#008300", "-.", "^", 1.1),
    "avgpack":  ("Grid average",    "#6a6a6a", "--", "D", 1.1),
}
BUDGETS = [27, 216]
FAMS = ["size", "density", "location", "radiomics"]     # data key
FAM_NICE = {"radiomics": "texture"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.linewidth": 0.45, "axes.edgecolor": "#8a8a86",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def acc_series(method, B, fam):
    """per_axis.{fam}.acc across s1 (then s2) epochs; None-padded where an eval is missing."""
    base = SWEEP / f"s1sweep_b{B}" / f"vqa_single_colipri_{method}_b{B}__{fam}"
    out = []
    for stage in ("s1", "s2"):
        vals = []
        for step in sorted(glob.glob(str(base / f"*__{stage}" / "evaluations" / "step_*"))):
            f = os.path.join(step, "vqa_metrics.json")
            v = None
            if os.path.exists(f):
                pa = json.load(open(f)).get("per_axis", {})
                v = pa.get(fam, {}).get("acc")
            vals.append(v)
        out.append(vals)
    return out  # [s1_list, s2_list]


OUT.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(len(BUDGETS), len(FAMS), figsize=(9.2, 4.4), sharex=True)
for r, B in enumerate(BUDGETS):
    for c, fam in enumerate(FAMS):
        ax = axes[r, c]
        ax.axvspan(0.5, 8.5, color="#eef3fb", zorder=0)
        ax.axvspan(8.5, 16.5, color="#fdf3ea", zorder=0)
        ax.axvline(8.5, color="#b5b5b0", lw=0.8, ls=":", zorder=1)
        for m, (disp, col, ls, mk, lw) in METHODS.items():
            s1, s2 = acc_series(m, B, fam)
            xs = list(range(1, 1 + len(s1))) + list(range(9, 9 + len(s2)))
            ys = s1 + s2
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts:
                continue
            xx, yy = zip(*pts)
            ax.plot(xx, yy, ls, color=col, lw=lw, marker=mk, markersize=2.4,
                    markeredgecolor="white", markeredgewidth=0.3, label=disp, zorder=5, clip_on=False)
        if r == 0:
            ax.set_title(FAM_NICE.get(fam, fam), fontsize=8, color="#0b0b0b", pad=3)
        if c == 0:
            ax.set_ylabel(f"$B{{=}}{B}$\naccuracy", fontsize=7.5, color="#52514e")
        ax.set_xticks(range(1, 17))
        ax.set_xticklabels([str(i) for i in range(1, 9)] + [str(i) for i in range(1, 9)], fontsize=5.2)
        ax.tick_params(labelsize=6.0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

for ax in axes[-1]:
    ax.set_xlabel("epoch (s1: 1-8 | s2: 1-8)", fontsize=6.6, color="#52514e", labelpad=1.5)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=6.6,
           bbox_to_anchor=(0.5, -0.02), handlelength=2.0, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.05, 1, 1), pad=0.3, w_pad=0.7, h_pad=0.8)
fig.savefig(OUT / "fig_vqa_2stage.pdf", dpi=300, bbox_inches="tight", pad_inches=0.015)
plt.close(fig)
print("wrote", OUT / "fig_vqa_2stage.pdf")
