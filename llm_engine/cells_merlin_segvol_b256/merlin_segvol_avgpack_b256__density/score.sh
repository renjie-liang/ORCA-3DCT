#!/usr/bin/env bash
# merlin_segvol_avgpack_b256/density score every per-epoch eval -> best.json (CPU, after s1)
set -euo pipefail
DEP="${1:-}"
sbatch --parsable --time=00:40:00 --partition=hpg-default --account=anon --ntasks=1 --cpus-per-task=2 --mem=16gb \
  --job-name=msc_segvol_avgpack_b256_density \
  --output=./results_llm/slurm_logs/%x_%j.out --error=./results_llm/slurm_logs/%x_%j.err \
  ${DEP:+--dependency=afterok:$DEP} \
  --wrap="export MAMBA_EXE=/home/anon/micromamba MAMBA_ROOT_PREFIX=/blue/anon/anon/micromamba; \
    eval \"\$(\$MAMBA_EXE shell hook --shell bash --root-prefix \$MAMBA_ROOT_PREFIX)\"; micromamba activate b200; \
    python ./llm_engine/score_sweep.py --run_dir ./results_llm/merlin_segvol_b256/vqa_single_merlin_segvol_avgpack_b256__density/vqa_single_merlin_segvol_avgpack_b256__density__s1 --gold ./data/vqa_merlin_v1/per_family/valid_density_merlin.json"
