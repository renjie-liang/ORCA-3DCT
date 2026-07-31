#!/usr/bin/env python3
"""Budget-curve figures.

  main:   COLIPRI only, 2x2 (size/density/location/texture) + legend below, single-column
          -> AAAI_2027/Figures/fig_budget_colipri.{pdf,png}
  append: SuPreM (top row) + SegVol (bottom row), 2x3 (size/density/location) + legend below
          -> AAAI_2027/Figures/fig_budget_merlin.{pdf,png}

Reads results_llm/budget_curve_data.json (3-seed; each point is [mean, sd], bands = +/-1 sd). Grid average is
ratio-based and is absent at the non-cubic budget 125. COLIPRI's x-grid is forced to include 512; those points
are empty until the b512 backfill lands, then re-run this script to fill them.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "AAAI_2027" / "Figures"
DATA = json.load(open(ROOT / "results_llm" / "budget_curve_data.json"))

C = {"orca": "#2a78d6", "avg": "#4a3aa7", "dins": "#008300", "tome": "#eda100"}
STYLE = {
    "orca":    ("ORCA (ours)",    C["orca"], "-",  "o", 1.35, 6),
    "avgpack": ("Grid average",   C["avg"],  "--", "s", 0.95, 4),
    "dins":    ("MedPruner-DINS", C["dins"], "-.", "^", 0.95, 3),
    "tome":    ("ToMe",           C["tome"], ":",  "D", 0.95, 2),
}
FAM_NICE = {"radiomics": "texture", "size_merlin": "size", "density_merlin": "density",
            "location_merlin": "location"}
FORCE_B = {"ctrate_colipri": [8, 27, 64, 125, 216, 512]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.linewidth": 0.45, "axes.edgecolor": "#8a8a86",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
OUT.mkdir(parents=True, exist_ok=True)


def panel(ax, corpus, fam):
    data = DATA[corpus]["data"]
    forced = FORCE_B.get(corpus)
    measured = {int(b) for m in data.values() for b in m}
    all_b = sorted(set(forced) | measured) if forced else sorted(measured)
    slot = {b: i for i, b in enumerate(all_b)}
    ax.grid(True, color="#ebeae6", linewidth=0.4, zorder=0); ax.set_axisbelow(True)
    for mkey in STYLE:
        if mkey not in data:
            continue
        disp, col, dash, mk, lw, z = STYLE[mkey]
        pts = sorted((int(b), fr[fam]) for b, fr in data[mkey].items() if fam in fr)
        if not pts:
            continue
        xs = [slot[b] for b, _ in pts]; ys = [v[0] for _, v in pts]; sd = [v[1] for _, v in pts]
        ax.fill_between(xs, [m - s for m, s in zip(ys, sd)], [m + s for m, s in zip(ys, sd)],
                        color=col, alpha=.12, linewidth=0, zorder=z)
        ax.plot(xs, ys, dash, color=col, linewidth=lw, marker=mk, markersize=2.6,
                markeredgecolor="white", markeredgewidth=.35, zorder=z + 10, label=disp, clip_on=False)
    ax.set_title(FAM_NICE.get(fam, fam), fontsize=7.5, color="#0b0b0b", pad=3)
    ax.set_xticks(range(len(all_b))); ax.set_xticklabels([str(b) for b in all_b], fontsize=6.0)
    ax.set_xlim(-0.35, len(all_b) - 0.65); ax.tick_params(labelsize=6.5)
    ax.set_xlabel("tokens $B$", fontsize=7, color="#52514e", labelpad=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def legend_below(fig, src_ax, y=-.02):
    h, l = src_ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=6.2,
               bbox_to_anchor=(.5, y), handlelength=1.9, columnspacing=1.2)


# ---- MAIN: COLIPRI only, 2x2 + legend below (single-column) ----
col_fams = DATA["ctrate_colipri"]["families"]
fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.0))
axf = axes.flatten()
for j, fam in enumerate(col_fams):
    panel(axf[j], "ctrate_colipri", fam)
axf[0].set_ylabel(r"$R^2$", fontsize=7.5, color="#52514e"); axf[2].set_ylabel(r"$R^2$", fontsize=7.5, color="#52514e")
legend_below(fig, axf[0], y=-.01)
fig.tight_layout(rect=(0, .06, 1, 1), pad=0.2, w_pad=.5, h_pad=.8)
for ext in ("pdf",):
    fig.savefig(OUT / f"fig_budget_colipri.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.015)
plt.close(fig)
print("wrote fig_budget_colipri (COLIPRI 2x2)")

# ---- APPENDIX: SuPreM (row 0) + SegVol (row 1), 2x3 + legend below ----
mer_fams = DATA["merlin_suprem"]["families"]              # size_merlin density_merlin location_merlin
fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.7))
for j, fam in enumerate(mer_fams):
    panel(axes[0, j], "merlin_suprem", fam)
    panel(axes[1, j], "merlin_segvol", fam)
axes[0, 0].set_ylabel(r"SuPreM   $R^2$", fontsize=7.5, color="#52514e", labelpad=2)
axes[1, 0].set_ylabel(r"SegVol   $R^2$", fontsize=7.5, color="#52514e", labelpad=2)
legend_below(fig, axes[0, 0], y=-.1)
fig.tight_layout(rect=(0, .05, 1, 1), pad=0.2, w_pad=.5, h_pad=.8)
for ext in ("pdf",):
    fig.savefig(OUT / f"fig_budget_merlin.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.015)
plt.close(fig)
print("wrote fig_budget_merlin (SuPreM + SegVol 2x3)")
