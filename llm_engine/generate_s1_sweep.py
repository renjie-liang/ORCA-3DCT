#!/usr/bin/env python3
"""S1-convergence sweep: colipri x {avgpack, orca} x {location,size,density,texture}, STAGE-1 ONLY.
Train the projector for UP TO 8 epochs with per-epoch FULL-valid eval (--eval-every=spe --valid-limit 0), so we
can pick the best epoch before committing to stage-2. Self-contained engine (llm_engine + results_llm), no DTBD3D.

  python generate_s1_sweep.py                 # budget 27 (the original sweep)
  python generate_s1_sweep.py --budget 64     # budget 64
Writes cells_s1sweep_b<BUDGET>/<cell>__<fam>/{s1.sh,score.sh,submit_chain.sh}; then submit_all.sh.
After: score.sh scores every epoch checkpoint -> best.json per cell.

avgpack pools on the fly (uniform_pool); ORCA tokens are PRECOMPUTED per budget (token_selection=none), so a
new budget needs its colipri_orca_b<BUDGET>.json manifest to exist -- asserted below rather than failing
halfway through a sweep."""
import argparse, json, math, os, stat
from pathlib import Path

CTOKEN = "."
CT = "/orange/anon/anon/3DCT/Compress_CT_Token"
ENG = f"{CTOKEN}/llm_engine"
EFF_BATCH = 16; EPOCHS = 8
# "radiomics", not the old "texture": the current registry (data_prep/vqa_labelgen/attributes.py) calls this
# axis radiomics, and it has 3 targets (the old set had 2 -- vert_firstorder_Kurtosis was simply absent).
# disease is deliberately NOT here: it is replaced by report-generation scored with clinical F1 over the same
# 18 findings (see REPORT_GEN_HANDOFF.md). It is also 18x the cost -- 217k steps vs ~12k.
FAMILIES = ["location", "size", "density", "radiomics"]

_ap = argparse.ArgumentParser(); _ap.add_argument("--budget", type=int, default=27)
_ap.add_argument("--only", default="", help="comma-separated arms to generate; empty = all. Use this when "
                 "adding an arm so submit_all.sh does not re-submit finished ones.")
_ap.add_argument("--seed", type=int, default=2026, help="training seed. 2026 keeps the existing __s1 dirs; "
                 "any other seed writes independent __s1_seed<N> runs (for the 3-seed VQA report-grade batch).")
_args = _ap.parse_args(); BUDGET = _args.budget; SEED = _args.seed
ONLY = [x for x in _args.only.split(",") if x]
STAG = "s1" if SEED == 2026 else f"s1_seed{SEED}"          # label/stage suffix -> per-seed dirs, no collision
SFX = "" if SEED == 2026 else f"_s{SEED}"                  # job-name / submit-script suffix
OUT = Path(f"{ENG}/cells_s1sweep_b{BUDGET}")
RUNROOT = f"{CTOKEN}/results_llm/s1sweep_b{BUDGET}"
def _man(name):   # a manifest must exist BEFORE a sweep is generated, not fail deep inside a job
    p = f"{CT}/configs/npy_manifests/{name}.json"
    assert os.path.exists(p), f"missing precomputed-token manifest: {p}"
    return p


# The three arms of the compression comparison. The legacy "orca" manifest (colipri_orca_nosin_d768_b27/b64 (renamed 2026-07-22)) is
# agglo_ORGAN lam=0.5 with NO position block -- neither ORCA-base nor ORCA+sin -- so it is deliberately not
# used here; `colipri_orca_nosin_noorgan_d768_*` and `colipri_orcasin_f4s2_*` are the clean arms.
METHODS = {  # method -> (manifest, token_selection, token_budget)
    "avgpack":  (f"{CT}/configs/npy_manifests/colipri.json", "uniform_pool", BUDGET),
    "orca_nosin_noorgan": (_man(f"colipri_orca_nosin_noorgan_d768_b{BUDGET}"), "none", 0),
    "orcasin":  (_man(f"colipri_orcasin_f4s2_b{BUDGET}"), "none", 0),
    # COMPLETE ORCA: organ prior lam=0.5 AND f4s2 position, 792-d. The full method, added 2026-07-19. The
    # three arms above are all ablations of it -- none has both halves. Manifest built by the VQA-side
    # writer in the same npy_manifests dir.
    "orcafull": (_man(f"colipri_orcafull_lam0p5_f4s2_b{BUDGET}"), "none", 0),
    # MedPruner-DINS: top-B by encoder attention + residual merge, 768-d, no position. Precomputed cache
    # (make_reportgen_pkg.py --arm dins), same manifest used by the report-gen handoff. T2's DINS VQA row.
    "dins":     (_man(f"colipri_dins_b{BUDGET}"), "none", 0),
}


def fam_json(sp, fam):
    """Per-family gold from data_prep/vqa_labelgen/split_per_family.py.

    NOT Compress_CT_Token/results/vqa/*_vqa_single_*.json: that set was built 2026-06-27 and superseded on
    2026-07-13, but the path lived on as an sbatch default, so three weeks of runs trained against it
    unnoticed (it still asks about `diaphragm_asym`, dropped 2026-07-12 for having no discrimination, and
    `trachea_x`, which no longer exists). Archived under results_llm/_DEPRECATED_2026-07-16/.
    Asserted below rather than left to fail deep inside a job.
    """
    p = f"{CTOKEN}/data/vqa/per_family/{sp}_{fam}.json"
    assert os.path.exists(p), f"missing per-family gold: {p}\n  run: python data_prep/vqa_labelgen/split_per_family.py"
    return p


def colipri_ids(sp): return set(l.strip() for l in open(f"{CT}/configs/npy_manifests/colipri_{sp}_ids.txt") if l.strip())


def spe(fam):
    recs = json.loads(Path(fam_json("train", fam)).read_text()); ids = colipri_ids("train")
    return math.ceil(sum(1 for r in recs if r["image"][:-7] in ids) / EFF_BATCH)


def w(p, t): p.write_text(t); p.chmod(p.stat().st_mode | stat.S_IEXEC)


# Wall-clock request. Was 7 h, which fitted the 2026-06-27 labels (size: 3958 spe -> 4.4 h) but not the
# current ones (size: 7528 spe -> 60,224 steps -> ~7.4 h at the measured ~2.3 step/s), so s1_avgpack_size was
# submitted ~24 min short and will TIMEOUT. SLURM will not extend a RUNNING job, so this number cannot be
# fixed after submit.
# 10 h, flat: the slowest family needs ~8.4 h at the pessimistic rate, and asking for much more just means
# never getting scheduled. A TIMEOUT is cheap anyway -- save_every=spe, so resume loses at most one epoch.
HOURS = 10


# The budget MUST be in the job name. Without it b27 and b64 are indistinguishable in squeue AND
# in the log filenames (--output=%x_%j), and on 2026-07-17 that cost a false alarm: two jobs named
# s1_avgpack_density looked like duplicate runs racing on one directory, when they were simply the
# b27 and b64 sweeps. The only way to tell them apart was grepping the out-dir out of each log.
def gen(method, fam):
    mani, sel, bud = METHODS[method]; cell = f"colipri_{method}_b{BUDGET}"
    e = spe(fam); steps = EPOCHS * e; run = f"{RUNROOT}/vqa_single_{cell}__{fam}"
    d = OUT / f"{cell}__{fam}"; d.mkdir(parents=True, exist_ok=True)
    tv, vv = fam_json("train", fam), fam_json("valid", fam)
    w(d / f"{STAG}.sh", f"""#!/usr/bin/env bash
# {cell}/{fam} STAGE1 projector seed={SEED}, {EPOCHS} epochs (spe={e}, STEPS={steps}), per-epoch full-valid eval
set -euo pipefail
CT={CTOKEN}; DEP="${{1:-}}"
MANIFEST="{mani}" TOKEN_SELECTION={sel} TOKEN_BUDGET={bud} SEED={SEED} \\
  TRAIN_VQA_JSON="{tv}" VALID_VQA_JSON="{vv}" \\
  RUN_ROOT={run} LABEL=vqa_single_{cell}__{fam}__{STAG} SAVE_EVERY={e} EVAL_EVERY={e} VALID_LIMIT=0 \\
  PROJECTOR_ONLY=1 LR=5e-4 STEPS={steps} \\
  sbatch --parsable --time={HOURS}:00:00 --job-name=s1_{method}_b{BUDGET}_{fam}{SFX} \\
    ${{DEP:+--dependency=afterok:$DEP}} "{ENG}/vqa_train_sc.sbatch"
""")
    w(d / f"score{SFX}.sh", f"""#!/usr/bin/env bash
# {cell}/{fam} seed={SEED} score every per-epoch eval -> best.json (run AFTER s1 done; CPU)
set -euo pipefail
DEP="${{1:-}}"
sbatch --parsable --time=00:40:00 --partition=hpg-default --account=anon --ntasks=1 --cpus-per-task=2 --mem=16gb \\
  --job-name=sc_{method}_b{BUDGET}_{fam}{SFX} --output={CTOKEN}/results_llm/slurm_logs/%x_%j.out --error={CTOKEN}/results_llm/slurm_logs/%x_%j.err \\
  ${{DEP:+--dependency=afterok:$DEP}} \\
  --wrap="export MAMBA_EXE=/home/anon/micromamba MAMBA_ROOT_PREFIX=/blue/anon/anon/micromamba; \\
    eval \\"\\$(\\$MAMBA_EXE shell hook --shell bash --root-prefix \\$MAMBA_ROOT_PREFIX)\\"; micromamba activate b200; \\
    python {ENG}/score_sweep.py --run_dir {run}/vqa_single_{cell}__{fam}__{STAG} --gold {vv}"
""")
    w(d / f"submit_chain{SFX}.sh", f"""#!/usr/bin/env bash
set -euo pipefail
D={OUT}/{cell}__{fam}
J1=$(bash "$D/{STAG}.sh"); echo "s1=$J1"
J2=$(bash "$D/score{SFX}.sh" "$J1"); echo "score=$J2"
echo "CHAIN {cell}__{fam} seed={SEED}: $J1 -> $J2"
""")
    return f"{cell}__{fam}", e, steps


def main():
    OUT.mkdir(parents=True, exist_ok=True); Path(f"{CTOKEN}/results_llm/slurm_logs").mkdir(parents=True, exist_ok=True)
    arms = ONLY or list(METHODS)
    for m in arms:
        assert m in METHODS, f"unknown arm {m}"
    lines = [gen(m, f) for m in arms for f in FAMILIES]
    # a scoped submit script when --only is used, so it can never touch the other arms
    subname = (f"submit_{'_'.join(arms)}{SFX}.sh" if ONLY else f"submit_all{SFX}.sh")
    subs = "\n".join(f'bash {OUT}/{n}/submit_chain{SFX}.sh' for n, _, _ in lines)
    w(OUT / subname, "#!/usr/bin/env bash\nset -euo pipefail\n" + subs + "\n")
    print(f"[submit script] {OUT}/{subname}")
    print(f"S1-sweep cells (budget {BUDGET}, {EPOCHS} epochs, per-epoch eval):")
    for n, e, s in lines: print(f"  {n:28s} spe={e} s1_steps={s}")
    print(f"\nMaster: {OUT}/submit_all.sh")


if __name__ == "__main__":
    main()
