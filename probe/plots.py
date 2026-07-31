#!/usr/bin/env python3
"""Auto-diagnostics from the saved preds. run.py calls plot_family() after each family (BEST epoch); or run
standalone to (re)plot any experiment:  python plots.py results/experiments/{exp_id}

  classify (disease) -> per-label AUROC bar (sorted) + ROC curves (3 weak + 3 strong)  [no calibration: pos_weight distorts it]
  regress            -> per-target error histogram + pred-vs-true scatter (RMSE/MAE/R2)

Output -> {exp_dir}/plots/{enc}_{fam}_{auroc|error}.png. Best epoch = max of the aggregate CSV row (macro / mean).
"""
import csv, glob, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import FAMILIES, ENCODERS   # noqa: E402


def _best_epoch(csv_path, agg):                                  # agg="macro"(classify) / "mean"(regress)
    best_e, best_s = 1, -1e9
    for r in csv.DictReader(open(csv_path)):
        if str(r.get("target")) == agg and r["score"] not in ("", "nan"):
            s = float(r["score"])
            if s > best_s:
                best_s, best_e = s, int(r["epoch"])
    return best_e


def plot_classify(npz, out_png, title):
    d = np.load(npz, allow_pickle=True)
    # float, NOT astype(int): NaN marks an unlabeled (volume, label) cell (Merlin marks 77% missing) and
    # casting it to int yields a huge sentinel that silently corrupts every AUROC/curve below.
    y = d["y_true"].astype(np.float32); prob = 1 / (1 + np.exp(-d["y_pred"])); names = [str(t) for t in d["targets"]]
    K = len(names)

    def _au(j):                                                  # score label j over its finite cells only
        m = np.isfinite(y[:, j])
        if m.sum() < 2 or y[m, j].min() == y[m, j].max():
            return np.nan
        return roc_auc_score(y[m, j], prob[m, j])
    au = np.array([_au(j) for j in range(K)])
    order = np.argsort(au); order = order[np.isfinite(au[order])]
    rep = list(order[:3]) + list(order[-3:])                     # 3 weak + 3 strong
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    cols = ["#2ca02c" if v >= 0.85 else "#ff7f0e" if v >= 0.75 else "#d62728" for v in au[order]]
    ax[0].barh(range(len(order)), au[order], color=cols)
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([names[i] for i in order], fontsize=8)
    mf = np.isfinite(y); yf, pf = y[mf].ravel(), prob[mf].ravel()
    macro = float(np.nanmean(au))
    micro = float(roc_auc_score(yf, pf)) if yf.size and yf.min() != yf.max() else float("nan")
    ax[0].axvline(macro, color="b", ls="--", label=f"macro={macro:.3f}")
    ax[0].axvline(micro, color="purple", ls=":", label=f"micro={micro:.3f}")
    ax[0].set_xlim(0.5, 1.0); ax[0].set_title("per-label AUROC (sorted)"); ax[0].legend(fontsize=8)
    for j in rep:
        mj = np.isfinite(y[:, j])
        fpr, tpr, _ = roc_curve(y[mj, j], prob[mj, j]); ax[1].plot(fpr, tpr, lw=1.3, label=f"{names[j][:20]} ({au[j]:.2f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[1].set_title("ROC (3 weak + 3 strong)"); ax[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=12, weight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=110); plt.close(fig)


def plot_regress(npz, out_png, title):
    d = np.load(npz, allow_pickle=True)
    yt, yp, names = d["y_true"], d["y_pred"], [str(t) for t in d["targets"]]; K = len(names)
    fig, ax = plt.subplots(2, K, figsize=(4.6 * K, 9), squeeze=False)
    for j in range(K):
        t, p = yt[:, j], yp[:, j]; m = np.isfinite(t) & np.isfinite(p); t, p = t[m], p[m]; err = p - t
        r2 = 1 - ((p - t) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9)
        ax[0, j].hist(err, bins=50, color="#cc4444", alpha=0.85); ax[0, j].axvline(0, color="k", lw=1)
        ax[0, j].set_title(f"{names[j]}: error\nstd={t.std():.3f} RMSE={np.sqrt((err**2).mean()):.3f} "
                           f"MAE={np.abs(err).mean():.3f} R2={r2:.3f}", fontsize=9)
        ax[0, j].set_xlabel("pred - true")
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax[1, j].scatter(t, p, s=6, alpha=0.3, color="#4477aa"); ax[1, j].plot(lim, lim, "r--", lw=1.2)
        ax[1, j].set_xlabel("true"); ax[1, j].set_ylabel("pred"); ax[1, j].set_title(f"{names[j]}: pred vs true", fontsize=9)
    fig.suptitle(title, fontsize=12, weight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110); plt.close(fig)


def plot_family(exp_dir, enc, fam, task):
    exp_dir = Path(exp_dir); csv_path = exp_dir / f"{enc}_{fam}.csv"
    if not csv_path.exists():
        return
    ep = _best_epoch(csv_path, "macro" if task == "classify" else "mean")
    npz = exp_dir / "preds" / f"{enc}_{fam}_ep{ep}.npz"
    if not npz.exists():
        return
    pdir = exp_dir / "plots"; pdir.mkdir(exist_ok=True)
    title = f"{exp_dir.name} :: {fam} ({enc}, best epoch {ep})"
    if task == "classify":
        plot_classify(str(npz), str(pdir / f"{enc}_{fam}_auroc.png"), title)
    else:
        plot_regress(str(npz), str(pdir / f"{enc}_{fam}_error.png"), title)
    print(f"[plots] {fam}: best epoch {ep} -> {pdir}", flush=True)


def main():   # standalone: (re)plot every family found in an experiment dir
    exp = Path(sys.argv[1])
    for csvf in sorted(glob.glob(str(exp / "*.csv"))):
        stem = Path(csvf).stem                                   # {enc}_{fam}
        enc = next((e for e in ENCODERS if stem.startswith(e + "_")), None)
        if not enc:
            continue
        fam = stem[len(enc) + 1:]
        if fam in FAMILIES:
            plot_family(str(exp), enc, fam, FAMILIES[fam]["task"])


if __name__ == "__main__":
    main()
