#!/usr/bin/env bash
# colipri_avgpack_b216/density score every s2 epoch -> best.json
set -euo pipefail
DEP="${1:-}"
sbatch --parsable --time=00:40:00 --partition=hpg-default --account=anon --ntasks=1 --cpus-per-task=2 --mem=16gb \
  --job-name=s2sc_avgpack_b216_density --output=./results_llm/slurm_logs/%x_%j.out --error=./results_llm/slurm_logs/%x_%j.err \
  ${DEP:+--dependency=afterok:$DEP} \
  --wrap="export MAMBA_EXE=/home/anon/micromamba MAMBA_ROOT_PREFIX=/blue/anon/anon/micromamba; \
    eval \"\$(\$MAMBA_EXE shell hook --shell bash --root-prefix \$MAMBA_ROOT_PREFIX)\"; micromamba activate b200; \
    python ./llm_engine/score_sweep.py --run_dir ./results_llm/s1sweep_b216/vqa_single_colipri_avgpack_b216__density/vqa_single_colipri_avgpack_b216__density__s2 --gold ./data/vqa/per_family/valid_density.json"
