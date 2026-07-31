#!/usr/bin/env bash
# colipri_avgpack_b27/density score every per-epoch eval -> best.json (run AFTER s1 done; CPU)
set -euo pipefail
DEP="${1:-}"
sbatch --parsable --time=00:40:00 --partition=hpg-default --account=anon --ntasks=1 --cpus-per-task=2 --mem=16gb \
  --job-name=sc_avgpack_density --output=./results_llm/slurm_logs/%x_%j.out --error=./results_llm/slurm_logs/%x_%j.err \
  ${DEP:+--dependency=afterok:$DEP} \
  --wrap="export MAMBA_EXE=/home/anon/micromamba MAMBA_ROOT_PREFIX=/blue/anon/anon/micromamba; \
    eval \"\$(\$MAMBA_EXE shell hook --shell bash --root-prefix \$MAMBA_ROOT_PREFIX)\"; micromamba activate b200; \
    python ./llm_engine/score_sweep.py --run_dir ./results_llm/s1sweep_b27/vqa_single_colipri_avgpack_b27__density/vqa_single_colipri_avgpack_b27__density__s1 --gold /orange/anon/anon/3DCT/Compress_CT_Token/results/vqa/valid_vqa_single_density.json"
