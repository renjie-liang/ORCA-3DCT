#!/usr/bin/env python3
"""Merlin VQA sweep, STAGE-1: <encoder> x {avgpack, orca} x {size,density,location,disease}, per-family.

    python generate_merlin_sweep.py --encoder segvol --budget 32
    python generate_merlin_sweep.py --encoder segvol --budget 256

Writes cells_merlin_<encoder>_b<BUDGET>/<cell>__<fam>/{s1.sh,score.sh,submit_chain.sh} + submit_all.sh.

ISOLATED from generate_s1_sweep.py on purpose. That file is hard-wired to colipri / CT-RATE (its manifests,
its per-family gold, its id lists, its family set with radiomics). Merlin differs on every one of those, and
editing the CT-RATE generator while its sweep is running would be reckless. Same TRAINING ENGINE
(vqa_train_sc.sbatch) -- that is env-driven and encoder-agnostic, so only the wiring changes here.

WHAT IS DIFFERENT FROM CT-RATE, and where it comes from:
  * encoder            SegVol (8x16x16) or SuPreM (12^3), not colipri.
  * budgets            SegVol 32/256, SuPreM 27/216 -- the token grids differ, so the ladders differ.
  * families           size, density, location, disease. NO radiomics (lung-based, absent in abdominal Merlin).
                       disease is the 30-finding presence VQA -- ~2.3x the records, so it gets 20h not 10h.
                       (CT-RATE replaces disease VQA with report generation; Merlin KEEPS disease VQA because
                       its 30 findings were human-labelled, with no RadBERT-style labeler to close the loop.)
  * per-family gold    data/vqa_merlin_v1/per_family/{sp}_<fam>_merlin.json  (note the _merlin suffix).
  * id list            merlin_{train,valid}_ids.txt, not colipri_*.
  * BOTH arms cached   avgpack is PRECOMPUTED here (uniform_pool cache), unlike CT-RATE where it pools on the
                       fly -- because the non-cubic SegVol grid is cleaner to pool offline. ORCA is lambda=2
                       (the 3-seed-settled Merlin value), 792-d for segvol / 216-d for suprem (feat+24).

Both arms use token_selection=none (precomputed), so both need their manifest to exist -- asserted up front.
"""
import argparse, json, math, os, stat
from pathlib import Path

CTOKEN = "."
CT = "/orange/anon/anon/3DCT/Compress_CT_Token"
ENG = f"{CTOKEN}/llm_engine"
MANDIR = f"{CT}/configs/npy_manifests"
EFF_BATCH = 16
EPOCHS = 8
HOURS = 10          # disease is ~2.3x the steps (30 findings x more records); bumped per-family in gen()
FAMILIES = ["size", "density", "location", "disease"]

_ap = argparse.ArgumentParser()
_ap.add_argument("--encoder", choices=["segvol", "suprem"], required=True)
_ap.add_argument("--budget", type=int, required=True)
_ap.add_argument("--only", default="", help="comma-separated families to generate (e.g. location_binary)")
_a = _ap.parse_args()
ENC, BUDGET = _a.encoder, _a.budget

OUT = Path(f"{ENG}/cells_merlin_{ENC}_b{BUDGET}")
RUNROOT = f"{CTOKEN}/results_llm/merlin_{ENC}_b{BUDGET}"


def _man(name):
    p = f"{MANDIR}/{name}.json"
    assert os.path.exists(p), (f"missing Merlin manifest: {p}\n"
                               f"  generate it: python data_prep/orca_tokens/make_merlin_vqa_cache.py "
                               f"--encoder {ENC} --manifest-only (after the cache shards finish)")
    return p


# Both arms precomputed -> token_selection=none, token_budget=0 (the budget is baked into the cache).
METHODS = {  # method -> (manifest, token_selection, token_budget)
    "avgpack": (_man(f"merlin_{ENC}_avgpool_b{BUDGET}"), "none", 0),
    "orca":    (_man(f"merlin_{ENC}_orcafull_lam2_f4s2_b{BUDGET}"), "none", 0),
}


def fam_json(sp, fam):
    # disease is `<sp>_disease.json`; location_binary is the median-split rebuild (tertile was unlearnable);
    # the regression families carry a `_merlin` suffix
    if fam == "location_binary":
        stem = "location_merlin_binary"
    elif fam == "disease":
        stem = "disease"
    else:
        stem = f"{fam}_merlin"
    p = f"{CTOKEN}/data/vqa_merlin_v1/per_family/{sp}_{stem}.json"
    assert os.path.exists(p), f"missing Merlin per-family gold: {p}"
    return p


def merlin_ids(sp):
    return set(l.strip() for l in open(f"{MANDIR}/merlin_{sp}_ids.txt") if l.strip())


def spe(fam):
    """Steps per epoch = ceil(labelled train records intersected with this encoder's volumes / batch)."""
    recs = json.loads(Path(fam_json("train", fam)).read_text())
    ids = merlin_ids("train")
    n = sum(1 for r in recs if r["image"][:-7] in ids)   # strip .nii.gz
    return math.ceil(n / EFF_BATCH)


def w(p, t):
    p.write_text(t)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def gen(method, fam):
    mani, sel, bud = METHODS[method]
    cell = f"merlin_{ENC}_{method}_b{BUDGET}"
    e = spe(fam)
    steps = EPOCHS * e
    run = f"{RUNROOT}/vqa_single_{cell}__{fam}"
    d = OUT / f"{cell}__{fam}"
    d.mkdir(parents=True, exist_ok=True)
    hours = 20 if fam == "disease" else HOURS      # disease sweep is ~2.3x, needs the time
    tv, vv = fam_json("train", fam), fam_json("valid", fam)
    w(d / "s1.sh", f"""#!/usr/bin/env bash
# {cell}/{fam} STAGE1 projector, {EPOCHS} epochs (spe={e}, STEPS={steps}), per-epoch full-valid eval
set -euo pipefail
DEP="${{1:-}}"
MANIFEST="{mani}" TOKEN_SELECTION={sel} TOKEN_BUDGET={bud} \\
  TRAIN_VQA_JSON="{tv}" VALID_VQA_JSON="{vv}" \\
  RUN_ROOT={run} LABEL=vqa_single_{cell}__{fam}__s1 SAVE_EVERY={e} EVAL_EVERY={e} VALID_LIMIT=0 \\
  PROJECTOR_ONLY=1 LR=5e-4 STEPS={steps} \\
  sbatch --parsable --time={hours}:00:00 --job-name=ms1_{ENC}_{method}_b{BUDGET}_{fam} \\
    ${{DEP:+--dependency=afterok:$DEP}} "{ENG}/vqa_train_sc.sbatch"
""")
    w(d / "score.sh", f"""#!/usr/bin/env bash
# {cell}/{fam} score every per-epoch eval -> best.json (CPU, after s1)
set -euo pipefail
DEP="${{1:-}}"
sbatch --parsable --time=00:40:00 --partition=hpg-default --account=anon --ntasks=1 --cpus-per-task=2 --mem=16gb \\
  --job-name=msc_{ENC}_{method}_b{BUDGET}_{fam} \\
  --output={CTOKEN}/results_llm/slurm_logs/%x_%j.out --error={CTOKEN}/results_llm/slurm_logs/%x_%j.err \\
  ${{DEP:+--dependency=afterok:$DEP}} \\
  --wrap="export MAMBA_EXE=/home/anon/micromamba MAMBA_ROOT_PREFIX=/blue/anon/anon/micromamba; \\
    eval \\"\\$(\\$MAMBA_EXE shell hook --shell bash --root-prefix \\$MAMBA_ROOT_PREFIX)\\"; micromamba activate b200; \\
    python {ENG}/score_sweep.py --run_dir {run}/vqa_single_{cell}__{fam}__s1 --gold {vv}"
""")
    w(d / "submit_chain.sh", f"""#!/usr/bin/env bash
set -euo pipefail
D={OUT}/{cell}__{fam}
J1=$(bash "$D/s1.sh"); echo "s1=$J1"
J2=$(bash "$D/score.sh" "$J1"); echo "score=$J2"
echo "CHAIN {cell}__{fam}: $J1 -> $J2"
""")
    return f"{cell}__{fam}", e, steps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Path(f"{CTOKEN}/results_llm/slurm_logs").mkdir(parents=True, exist_ok=True)
    fams = [x for x in _a.only.split(",") if x] or FAMILIES
    lines = [gen(m, f) for m in METHODS for f in fams]
    subs = "\n".join(f'bash {OUT}/{n}/submit_chain.sh' for n, _, _ in lines)
    subname = "submit_" + "_".join(_a.only.split(",")) + ".sh" if _a.only else "submit_all.sh"
    w(OUT / subname, "#!/usr/bin/env bash\nset -euo pipefail\n" + subs + "\n")
    print(f"[submit] {OUT}/{subname}")
    print(f"Merlin sweep: encoder={ENC} budget={BUDGET}, {len(METHODS)} arms x {len(FAMILIES)} families, "
          f"{EPOCHS} epochs, per-epoch eval:")
    for n, e, s in lines:
        print(f"  {n:40s} spe={e} s1_steps={s}")
    print(f"\nMaster: {OUT}/submit_all.sh")
    print(f"Manifests used:")
    for m, (mani, _, _) in METHODS.items():
        print(f"  {m:8s} {mani}")


if __name__ == "__main__":
    main()
