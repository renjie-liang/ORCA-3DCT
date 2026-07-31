#!/usr/bin/env python3
"""Score every per-epoch eval in an s1-sweep run dir and pick the best epoch.
For each evaluations/step_*/raw_btb3d_output.jsonl, run score_vqa (writes vqa_metrics.json alongside), collect
the overall acc curve, write best.json {best_step, best_acc, curve}. CPU.

  python score_sweep.py --run_dir <.../__s1> --gold <valid_vqa_single_<fam>.json>"""
import argparse, glob, json, os, subprocess, sys

SCORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_vqa.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--gold", required=True)
    a = ap.parse_args()
    raws = sorted(glob.glob(f"{a.run_dir}/evaluations/step_*/raw_btb3d_output.jsonl"))
    if not raws:
        print(f"NO eval outputs under {a.run_dir}/evaluations/", flush=True); sys.exit(1)
    curve = {}
    for raw in raws:
        step = raw.split("/")[-2]
        mj = os.path.join(os.path.dirname(raw), "vqa_metrics.json")
        if not os.path.exists(mj):
            subprocess.run([sys.executable, SCORE, "--pred", raw, "--gold", a.gold], check=True)
        d = json.load(open(mj))
        curve[step] = d["overall"]["acc"] if isinstance(d.get("overall"), dict) else d.get("overall")
        print(f"  {step}: acc={curve[step]:.4f}", flush=True)
    best = max(curve, key=curve.get)
    json.dump({"best_step": best, "best_acc": curve[best], "curve": curve},
              open(f"{a.run_dir}/best.json", "w"), indent=2)
    print(f"BEST {a.run_dir.split('/')[-2]}: {best} acc={curve[best]:.4f}\nDONE", flush=True)


if __name__ == "__main__":
    main()
