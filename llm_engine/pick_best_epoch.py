#!/usr/bin/env python3
"""Pick the best epoch of a report-gen stage by clinical F1, and write best.json.

  python llm_engine/pick_best_epoch.py --run_dir results_llm/reportgen_ORCA_b27/reportgen_ORCA_b27__s1

Why this is not score_sweep.py: that one scores MULTIPLE-CHOICE answers (exact letter match) and reads each
epoch's `vqa_metrics.json`. Report-gen has no letters — the engine runs eval_fast.py instead and writes
`metrics_fast.json` (RadBERT clinical precision/recall/F1 over CT-RATE's 18 findings, plus BLEU/ROUGE-L/
METEOR/CIDEr and CRG). So report-gen produces no best.json at all unless something like this makes one, and
s2 needs it to know which s1 checkpoint to warm-start from.

Selection metric: **clinical F1**. The text metrics reward copying phrasing; only the RadBERT labels speak to
whether the compressed tokens preserved the findings, which is the question the study asks.
"""
import argparse, glob, json, os, re, sys


def f1_of(payload):
    """Pull clinical F1 out of metrics_fast.json.

    eval_fast.py writes the same number in two places (eval_fast.py:286 `out["summary"] = make_flat_summary(out)`):
      payload["clinical"]["f1"]        the nested value
      payload["summary"]["clinical_f1"]  the flattened alias
    Check both, and return None rather than guessing — the caller turns that into a hard error. A selector
    that silently picks epoch 0 because it could not find its metric is worse than one that stops.
    """
    cl = payload.get("clinical")
    if isinstance(cl, dict) and isinstance(cl.get("f1"), (int, float)):
        return float(cl["f1"])
    sm = payload.get("summary")
    if isinstance(sm, dict) and isinstance(sm.get("clinical_f1"), (int, float)):
        return float(sm["clinical_f1"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.run_dir}/evaluations/step_*/metrics_fast.json"),
                   key=lambda p: int(re.sub(r"\D", "", os.path.basename(os.path.dirname(p)))))
    if not files:
        sys.exit(f"FATAL: no evaluations/step_*/metrics_fast.json under {a.run_dir}\n"
                 f"  If the runs finished, eval_fast did not — check run.log for the eval_fast subprocess.")

    curve = {}
    for p in files:
        step = os.path.basename(os.path.dirname(p))
        f1 = f1_of(json.load(open(p)))
        if f1 is None:
            sys.exit(f"FATAL: {p} has no clinical F1 field — eval_fast's output shape changed, tell the authors")
        curve[step] = f1

    best = max(curve, key=curve.get)
    out = a.out or f"{a.run_dir}/best.json"
    json.dump({"best_step": best, "best_acc": curve[best], "metric": "clinical_f1", "curve": curve},
              open(out, "w"), indent=2)
    print(f"{'epoch':>14s} {'clinical F1':>12s}")
    for k, v in curve.items():
        print(f"{k:>14s} {v:12.4f}" + ("   <-- BEST" if k == best else ""))
    print(f"\nbest={best} f1={curve[best]:.4f} -> {out}")


if __name__ == "__main__":
    main()
