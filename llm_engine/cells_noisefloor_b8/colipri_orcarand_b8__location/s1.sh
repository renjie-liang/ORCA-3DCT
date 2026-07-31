#!/usr/bin/env bash
# NOISE FLOOR colipri_orcarand_b8/location STAGE1 projector, 8 epochs (spe=4520, STEPS=36160). VQA_NOISE_INPUT replaces visual tokens with fresh N(0,1).
set -euo pipefail
DEP="${1:-}"
VQA_NOISE_INPUT=1 MANIFEST="./vendor/npy_manifests/colipri_orcafull_lam0p5_f4s2_b8.json" TOKEN_SELECTION=none TOKEN_BUDGET=0 \
  TRAIN_VQA_JSON="./data/vqa/per_family/train_location.json" VALID_VQA_JSON="./data/vqa/per_family/valid_location.json" \
  RUN_ROOT=./results_llm/noisefloor_b8/vqa_single_colipri_orcarand_b8__location LABEL=vqa_single_colipri_orcarand_b8__location__s1 SAVE_EVERY=4520 EVAL_EVERY=4520 VALID_LIMIT=0 \
  PROJECTOR_ONLY=1 LR=5e-4 STEPS=36160 \
  sbatch --parsable --time=10:00:00 --job-name=s1_orcarand_b8_location \
    --export=ALL,VQA_NOISE_INPUT=1 \
    ${DEP:+--dependency=afterok:$DEP} "./llm_engine/vqa_train_sc.sbatch"
