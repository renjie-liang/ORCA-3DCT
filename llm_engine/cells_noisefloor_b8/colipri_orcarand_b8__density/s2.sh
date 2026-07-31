#!/usr/bin/env bash
# NOISE FLOOR colipri_orcarand_b8/density STAGE2 LoRA+proj, warm-start from FINAL s1 ckpt step_036160 (option A: final not best-epoch).
set -euo pipefail
DEP="${1:-}"
VQA_NOISE_INPUT=1 MANIFEST="./vendor/npy_manifests/colipri_orcafull_lam0p5_f4s2_b8.json" TOKEN_SELECTION=none TOKEN_BUDGET=0 \
  TRAIN_VQA_JSON="./data/vqa/per_family/train_density.json" VALID_VQA_JSON="./data/vqa/per_family/valid_density.json" \
  RUN_ROOT=./results_llm/noisefloor_b8/vqa_single_colipri_orcarand_b8__density LABEL=vqa_single_colipri_orcarand_b8__density__s2 SAVE_EVERY=4520 EVAL_EVERY=4520 VALID_LIMIT=0 \
  PROJECTOR_ONLY=0 LR=2e-5 INIT_WEIGHTS_FROM_CHECKPOINT=./results_llm/noisefloor_b8/vqa_single_colipri_orcarand_b8__density/vqa_single_colipri_orcarand_b8__density__s1/checkpoints/step_036160 STEPS=36160 \
  sbatch --parsable --time=14:00:00 --job-name=s2_orcarand_b8_density \
    --export=ALL,VQA_NOISE_INPUT=1 \
    ${DEP:+--dependency=afterok:$DEP} "./llm_engine/vqa_train_sc.sbatch"
