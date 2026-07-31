#!/usr/bin/env python3
"""F4 -- effect of the organ-prior weight lambda, one panel per corpus/encoder, REAL 3-seed data.

    python paper_compression/figures_code/scan_results.py      # refresh index
    python paper_compression/figures_code/build_figdata.py     # -> results_llm/lambda_data.json
    python paper_compression/figures_code/fig_lambda.py        # -> figures/fig_lambda.{pdf,png}

Reads results_llm/lambda_data.json ({corpus: {title, families, data:{lambda: {family:[mean,sd]}}}}); numbers live
in that file, not here. Each point is the best-epoch-on-seed-mean R^2 over 3 seeds; the shaded band is +/- 1 sd.

Design decisions:
  * x is a CATEGORICAL axis -- the sampled lambda settings, evenly spaced -- not a true log axis. lambda=0 is an
    ordinary leftmost point (0 cannot sit on a log axis); the jump 32 -> 1e4 costs one slot, not 2.5 empty decades;
    the reader compares the settings we actually ran. Distances do NOT encode ratios -- said so in the caption.
  * lambda=1e4 is the ORGAN-DOMINATED LIMIT: the organ block swamps Ward's distance -> "merge within organ" (hard
    organ pooling). It answers "why not just pool by the mask?" as a continuous deformation of our own method, on
    the same axis and code, not a separate baseline. Its slot is lightly shaded to mark it as a limit.
  * one line per FAMILY, not a mean -- the prior may help one family and hurt another, which an average hides.
  * every configuration carries the sinusoidal centroid block, so the curve isolates the organ PRIOR (the old
    exp_orgl* colipri sweep has no position block and is excluded: there the prior partly stood in for position).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "AAAI_2027" / "Figures"
DATA = json.load(open(ROOT / "results_llm" / "lambda_data.json"))

ORDER = ["ctrate_colipri", "merlin_suprem", "merlin_segvol"]      # left-to-right panel order
BUD = {"ctrate_colipri": 216, "merlin_suprem": 216, "merlin_segvol": 256}
LAM = ["0", "0.5", "2", "8", "32", "10000"]                       # categorical, evenly spaced (0 included)
LAM_LABELS = ["0", "0.5", "2", "8", "32", "$10^4$"]
FAM = {"size": "#2a78d6", "density": "#008300", "location": "#eda100", "texture": "#4a3aa7"}
DASH = {"size": "-", "density": "--", "location": "-.", "texture": ":"}
MK = {"size": "o", "density": "s", "location": "^", "texture": "D"}


def disp(fkey):                                                  # family key -> style/legend name
    f = fkey.replace("_merlin", "")
    return "texture" if f == "radiomics" else f


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.linewidth": 0.45, "axes.edgecolor": "#8a8a86",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.size": 2, "ytick.major.size": 2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, len(ORDER), figsize=(7.16, 1.95))
x = np.arange(len(LAM))
for ax, ck in zip(axes, ORDER):
    d = DATA[ck]["data"]
    ax.grid(True, color="#ebeae6", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.axvspan(x[-1] - .5, x[-1] + .5, color="#8a8a86", alpha=.09, zorder=1)   # organ-dominated limit slot
    for fkey in DATA[ck]["families"]:
        name = disp(fkey); col = FAM[name]
        ys = np.array([d[l][fkey][0] for l in LAM])
        sd = np.array([d[l][fkey][1] for l in LAM])
        ax.fill_between(x, ys - sd, ys + sd, color=col, alpha=.12, linewidth=0, zorder=4)
        ax.plot(x, ys, DASH[name], color=col, linewidth=1.3, marker=MK[name], markersize=2.6,
                markeredgecolor="white", markeredgewidth=.35, label=name, zorder=6, clip_on=False)
    ax.set_title(f"{DATA[ck]['title']}, $B{{=}}{BUD[ck]}$", fontsize=7.5, color="#0b0b0b", pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(LAM_LABELS, fontsize=6.5)
    ax.set_xlim(-.35, len(x) - .65)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel(r"organ-prior weight $\lambda$", fontsize=7, color="#52514e", labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel(r"Probe $R^2$", fontsize=7.5, color="#52514e", labelpad=2)

h, l = [], []
for ax in axes:                                   # union of families across panels, first occurrence wins
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.8,
           bbox_to_anchor=(.5, -.06), handlelength=2.2, columnspacing=1.3)

fig.tight_layout(rect=(0, .06, 1, .99), pad=0.2, w_pad=.5, h_pad=.8)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf",):
    fig.savefig(OUT / f"fig_lambda.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.015)
print("wrote fig_lambda.pdf / .png (real 3-seed, bands = +/-1 sd)")
