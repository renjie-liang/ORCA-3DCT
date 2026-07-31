#!/usr/bin/env bash
# colipri_avgpack_b216/density STAGE1 projector seed=2027, 8 epochs (spe=6021, STEPS=48168), per-epoch full-valid eval
set -euo pipefail
CT=.; DEP="${1:-}"
MANIFEST="./vendor/npy_manifests/colipri.json" TOKEN_SELECTION=uniform_pool TOKEN_BUDGET=216 SEED=2027 \
  TRAIN_VQA_JSON="./data/vqa/per_family/train_density.json" VALID_VQA_JSON="./data/vqa/per_family/valid_density.json" \
  RUN_ROOT=./results_llm/s1sweep_b216/vqa_single_colipri_avgpack_b216__density LABEL=vqa_single_colipri_avgpack_b216__density__s1_seed2027 SAVE_EVERY=6021 EVAL_EVERY=6021 VALID_LIMIT=0 \
  PROJECTOR_ONLY=1 LR=5e-4 STEPS=48168 \
  sbatch --parsable --time=10:00:00 --job-name=s1_avgpack_b216_density_s2027 \
    ${DEP:+--dependency=afterok:$DEP} "./llm_engine/vqa_train_sc.sbatch"
