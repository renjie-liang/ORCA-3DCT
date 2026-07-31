#!/usr/bin/env python3
"""Report-generation two-stage per-epoch curves (appendix diagnostic).

For COLIPRI b216, the report-gen model is trained in two stages: s1 projector-only warmup (8 epochs),
then s2 LoRA + projector (4 epochs). We plot every validation metric per epoch across BOTH stages for the
four compressors (ORCA full 792-dim with position, ORCA 768-dim no-position ablation, MedPruner-DINS,
Grid average).

The story the figure carries:
  - After s1 (projector only) all four are tied (~0.43 clinical F1); ORCA-792 even starts highest and has
    the LOWEST validation loss -> the projector represents the 792-dim tokens fine, position does not hurt s1.
  - The gap opens the instant LoRA turns on (s2). Clinical metrics (clinical F1, CRG) jump +0.04 for three
    arms but only +0.007/+0.001 for ORCA-792. Lexical metrics (BLEU-1/4, ROUGE-L) and validation loss show
    NO such gap -> the deficit is clinical-content-specific, not fluency or language-modeling.
  => the LM objective (next-token loss) is not the clinical objective; the 24 sinusoidal position dims shift
     the LM-optimal generation slightly away from the clinical-optimal one, a cost a linear probe never pays.

  -> AAAI_2027/Figures/fig_reportgen_2stage.pdf

Reads results/from_collabrator/runs/reportgen_{arm}_b216/reportgen_{arm}_b216__{s1,s2}/evaluations/step_*/
metrics_fast.json (single seed; best-validation-epoch cells are what tab:textgen reports).
"""
import json
import glob
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "AAAI_2027" / "Figures"
RUNS = ROOT / "results" / "from_collabrator" / "runs"

# arm -> (display, color, linestyle, marker, linewidth)
# colors matched to fig_budget_combined.py: orca=#2a78d6, Grid average=#4a3aa7, DINS=#008300
ARMS = {
    "orcafull": ("ORCA (full, +position)", "#2a78d6", "-",  "o", 1.5),
    "orca":     ("ORCA (no position)",     "#7fb3e6", "-",  "v", 1.2),
    "dins":     ("MedPruner-DINS",         "#008300", "-.", "^", 1.0),
    "avgpack":  ("Grid average",           "#4a3aa7", "--", "s", 1.0),
}
# (key, panel title, lower-is-better)
METRICS = [
    ("summary.clinical_f1", "clinical F1", False),
    ("summary.crg",         "CRG",         False),
    ("valid_loss",          "validation loss", True),
    ("summary.bleu_1",      "BLEU-1",      False),
    ("summary.bleu_4",      "BLEU-4",      False),
    ("summary.rouge_l",     "ROUGE-L",     False),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.linewidth": 0.45, "axes.edgecolor": "#8a8a86",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _flat(o, p=""):
    d = {}
    if isinstance(o, dict):
        for k, v in o.items():
            d.update(_flat(v, p + k + "."))
        return d
    d[p[:-1]] = o
    return d


def series(arm, key, bud):
    """Per-epoch metric across s1 then s2 for one arm at budget bud."""
    out = []
    for stage in ("s1", "s2"):
        base = RUNS / f"reportgen_{arm}_{bud}" / f"reportgen_{arm}_{bud}__{stage}" / "evaluations"
        vals = []
        for step in sorted(glob.glob(str(base / "step_*"))):
            f = os.path.join(step, "metrics_fast.json")
            vals.append(_flat(json.load(open(f))).get(key) if os.path.exists(f) else None)
        out.append(vals)
    return out  # [s1_list, s2_list]


def n_steps(arm, stage, bud):
    base = RUNS / f"reportgen_{arm}_{bud}" / f"reportgen_{arm}_{bud}__{stage}" / "evaluations"
    return len(glob.glob(str(base / "step_*")))


def make_fig(bud):
    S1N = max(n_steps(a, "s1", bud) for a in ARMS)   # projector-only epochs
    S2N = max(n_steps(a, "s2", bud) for a in ARMS)   # LoRA+projector epochs (variable)
    BOUND = S1N + 0.5
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.0))
    for ax, (key, title, lower) in zip(axes.flat, METRICS):
        ax.axvspan(0.5, BOUND, color="#eef3fb", zorder=0)                 # s1 tint
        ax.axvspan(BOUND, S1N + S2N + 0.5, color="#fdf3ea", zorder=0)     # s2 tint
        ax.axvline(BOUND, color="#b5b5b0", lw=0.8, ls=":", zorder=1)
        for arm, (disp, col, ls, mk, lw) in ARMS.items():
            s1, s2 = series(arm, key, bud)
            xs = list(range(1, 1 + len(s1))) + list(range(S1N + 1, S1N + 1 + len(s2)))
            ax.plot(xs, s1 + s2, ls, color=col, lw=lw, marker=mk, markersize=2.4,
                    markeredgecolor="white", markeredgewidth=0.35, label=disp, zorder=5, clip_on=False)
        ax.set_title(title, fontsize=7.5, color="#0b0b0b", pad=3)
        ticks = list(range(1, S1N + 1)) + list(range(S1N + 1, S1N + 1 + S2N))
        labels = [str(i) for i in range(1, S1N + 1)] + [str(i) for i in range(1, S2N + 1)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=4.8)
        ax.set_xlim(0.4, S1N + S2N + 0.6)
        ax.tick_params(labelsize=6.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(f"epoch  (s1: 1-{S1N}  |  s2: 1-{S2N})", fontsize=6.6, color="#52514e", labelpad=1.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.5, -0.015), handlelength=2.0, columnspacing=1.4)
    fig.suptitle(f"Report generation, $B={bud[1:]}$", fontsize=8.5, y=1.005)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98), pad=0.3, w_pad=0.7, h_pad=0.9)
    out = OUT / f"fig_reportgen_2stage_{bud}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print("wrote", out)


OUT.mkdir(parents=True, exist_ok=True)
for bud in ("b27", "b216"):
    make_fig(bud)
