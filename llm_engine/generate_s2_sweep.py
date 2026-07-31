#!/usr/bin/env python3
"""STAGE-2 sweep: warm-start each cell from its BEST s1 epoch (read from s1's best.json), then train
LoRA+projector for 8 epochs with per-epoch FULL-valid eval, and pick the best s2 epoch. Self-contained engine.

  python generate_s2_sweep.py                 # budget 27
  python generate_s2_sweep.py --budget 64     # budget 64
Writes cells_s2sweep_b<BUDGET>/<cell>__<fam>/{s2.sh,score.sh,submit_chain.sh}.
Requires the s1 sweep for that budget to have finished (best.json present per cell). Time limits are GENEROUS
on purpose: a too-tight limit killed an earlier s2 at 80% (TIMEOUT) — never under-budget wall time again.

Kept in step with generate_s1_sweep.py (2026-07-16): per-family gold from data/vqa/per_family/, "radiomics"
not "texture", disease excluded (report-gen replaces it), and --budget parameterised instead of hard-coded 27.
"""
import argparse, json, math, os, stat
from pathlib import Path

CTOKEN = "."
CT = "/orange/anon/anon/3DCT/Compress_CT_Token"
ENG = f"{CTOKEN}/llm_engine"
EFF_BATCH = 16; EPOCHS = 8
FAMILIES = ["location", "size", "density", "radiomics"]

_ap = argparse.ArgumentParser(); _ap.add_argument("--budget", type=int, default=27); _ap.add_argument("--method", default=None)
_A = _ap.parse_args(); BUDGET = _A.budget; ONLY = _A.method
OUT = Path(f"{ENG}/cells_s2sweep_b{BUDGET}")
RUNROOT = f"{CTOKEN}/results_llm/s1sweep_b{BUDGET}"      # s2 lives alongside s1 in the same cell dir
def _man(name):   # a manifest must exist BEFORE a sweep is generated, not fail deep inside a job
    p = f"{CT}/configs/npy_manifests/{name}.json"
    assert os.path.exists(p), f"missing precomputed-token manifest: {p}"
    return p


# The three arms of the compression comparison. The legacy "orca" manifest (colipri_orca_b27/b64) is
# agglo_ORGAN lam=0.5 with NO position block -- neither ORCA-base nor ORCA+sin -- so it is deliberately not
# used here; `colipri_orca_nosin_noorgan_d768_*` and `colipri_orcasin_f4s2_*` are the clean arms.
METHODS = {  # method -> (manifest, token_selection, token_budget)
    "avgpack":  (f"{CT}/configs/npy_manifests/colipri.json", "uniform_pool", BUDGET),
    "dins":     (_man(f"colipri_dins_b{BUDGET}"), "none", 0),   # MedPruner-DINS baseline (precompressed manifest)
    "orca_nosin_noorgan": (_man(f"colipri_orca_nosin_noorgan_d768_b{BUDGET}"), "none", 0),
    "orcasin":  (_man(f"colipri_orcasin_f4s2_b{BUDGET}"), "none", 0),
    "orcafull": (_man(f"colipri_orcafull_lam0p5_f4s2_b{BUDGET}"), "none", 0),   # THE full method (organ prior + sin position, 792-d)
}


def fam_json(sp, fam):
    """Per-family gold from data_prep/vqa_labelgen/split_per_family.py — NOT the 2026-06-27 set under
    Compress_CT_Token/results/vqa/, which was superseded on 2026-07-13 and survived only as an sbatch
    default (see results_llm/_DEPRECATED_2026-07-16/DEPRECATED.md)."""
    p = f"{CTOKEN}/data/vqa/per_family/{sp}_{fam}.json"
    assert os.path.exists(p), f"missing per-family gold: {p}\n  run: python data_prep/vqa_labelgen/split_per_family.py"
    return p


def colipri_ids(sp): return set(l.strip() for l in open(f"{CT}/configs/npy_manifests/colipri_{sp}_ids.txt") if l.strip())


def spe(fam):
    recs = json.loads(Path(fam_json("train", fam)).read_text()); ids = colipri_ids("train")
    return math.ceil(sum(1 for r in recs if r["image"][:-7] in ids) / EFF_BATCH)


def w(p, t): p.write_text(t); p.chmod(p.stat().st_mode | stat.S_IEXEC)


# The budget MUST be in the job name. Without it b27 and b64 are indistinguishable in squeue AND
# in the log filenames (--output=%x_%j), and on 2026-07-17 that cost a false alarm: two jobs named
# s1_avgpack_density looked like duplicate runs racing on one directory, when they were simply the
# b27 and b64 sweeps. The only way to tell them apart was grepping the out-dir out of each log.
def gen(method, fam):
    mani, sel, bud = METHODS[method]; cell = f"colipri_{method}_b{BUDGET}"
    run = f"{RUNROOT}/vqa_single_{cell}__{fam}"
    s1dir = f"{run}/vqa_single_{cell}__{fam}__s1"
    best = json.loads(Path(f"{s1dir}/best.json").read_text())     # fail fast if s1 not finished
    init = f"{s1dir}/checkpoints/{best['best_step']}"
    e = spe(fam); steps = EPOCHS * e
    d = OUT / f"{cell}__{fam}"; d.mkdir(parents=True, exist_ok=True)
    tv, vv = fam_json("train", fam), fam_json("valid", fam)
    w(d / "s2.sh", f"""#!/usr/bin/env bash
# {cell}/{fam} STAGE2 LoRA+proj, {EPOCHS} epochs (spe={e}, STEPS={steps}), per-epoch full-valid eval
# warm-start from BEST s1 epoch {best['best_step']} (s1 best_acc={best['best_acc']:.4f})
set -euo pipefail
DEP="${{1:-}}"
MANIFEST="{mani}" TOKEN_SELECTION={sel} TOKEN_BUDGET={bud} \\
  TRAIN_VQA_JSON="{tv}" VALID_VQA_JSON="{vv}" \\
  RUN_ROOT={run} LABEL=vqa_single_{cell}__{fam}__s2 SAVE_EVERY={e} EVAL_EVERY={e} VALID_LIMIT=0 \\
  PROJECTOR_ONLY=0 LR=2e-5 INIT_WEIGHTS_FROM_CHECKPOINT={init} STEPS={steps} \\
  sbatch --parsable --time=14:00:00 --job-name=s2_{method}_b{BUDGET}_{fam} \\
    ${{DEP:+--dependency=afterok:$DEP}} "{ENG}/vqa_train_sc.sbatch"
""")
    w(d / "score.sh", f"""#!/usr/bin/env bash
# {cell}/{fam} score every s2 epoch -> best.json
set -euo pipefail
DEP="${{1:-}}"
sbatch --parsable --time=00:40:00 --partition=hpg-default --account=anon --ntasks=1 --cpus-per-task=2 --mem=16gb \\
  --job-name=s2sc_{method}_b{BUDGET}_{fam} --output={CTOKEN}/results_llm/slurm_logs/%x_%j.out --error={CTOKEN}/results_llm/slurm_logs/%x_%j.err \\
  ${{DEP:+--dependency=afterok:$DEP}} \\
  --wrap="export MAMBA_EXE=/home/anon/micromamba MAMBA_ROOT_PREFIX=/blue/anon/anon/micromamba; \\
    eval \\"\\$(\\$MAMBA_EXE shell hook --shell bash --root-prefix \\$MAMBA_ROOT_PREFIX)\\"; micromamba activate b200; \\
    python {ENG}/score_sweep.py --run_dir {run}/vqa_single_{cell}__{fam}__s2 --gold {vv}"
""")
    w(d / "submit_chain.sh", f"""#!/usr/bin/env bash
set -euo pipefail
D={OUT}/{cell}__{fam}
J1=$(bash "$D/s2.sh"); echo "s2=$J1"
J2=$(bash "$D/score.sh" "$J1"); echo "score=$J2"
echo "CHAIN {cell}__{fam}: $J1 -> $J2"
""")
    return f"{cell}__{fam}", e, steps, best["best_step"], best["best_acc"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    methods = [ONLY] if ONLY else list(METHODS)
    rows = [gen(m, f) for m in methods for f in FAMILIES]
    subs = "\n".join(f'bash {OUT}/{n}/submit_chain.sh' for n, _, _, _, _ in rows)
    w(OUT / "submit_all.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + subs + "\n")
    print("S2-sweep cells (budget 27, 8 epochs, warm-start from best s1):")
    for n, e, s, bs, ba in rows:
        print(f"  {n:28s} spe={e} s2_steps={s}  init={bs} (s1 acc={ba:.4f})")
    print(f"\nMaster: {OUT}/submit_all.sh")


if __name__ == "__main__":
    main()
