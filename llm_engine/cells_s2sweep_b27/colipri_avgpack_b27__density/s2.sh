#!/usr/bin/env bash
# colipri_avgpack_b27/density STAGE2 LoRA+proj, 8 epochs (spe=6021, STEPS=48168), per-epoch full-valid eval
# warm-start from BEST s1 epoch step_042147 (s1 best_acc=0.7670)
set -euo pipefail
DEP="${1:-}"
MANIFEST="./vendor/npy_manifests/colipri.json" TOKEN_SELECTION=uniform_pool TOKEN_BUDGET=27 \
  TRAIN_VQA_JSON="./data/vqa/per_family/train_density.json" VALID_VQA_JSON="./data/vqa/per_family/valid_density.json" \
  RUN_ROOT=./results_llm/s1sweep_b27/vqa_single_colipri_avgpack_b27__density LABEL=vqa_single_colipri_avgpack_b27__density__s2 SAVE_EVERY=6021 EVAL_EVERY=6021 VALID_LIMIT=0 \
  PROJECTOR_ONLY=0 LR=2e-5 INIT_WEIGHTS_FROM_CHECKPOINT=./results_llm/s1sweep_b27/vqa_single_colipri_avgpack_b27__density/vqa_single_colipri_avgpack_b27__density__s1/checkpoints/step_042147 STEPS=48168 \
  sbatch --parsable --time=14:00:00 --job-name=s2_avgpack_b27_density \
    ${DEP:+--dependency=afterok:$DEP} "./llm_engine/vqa_train_sc.sbatch"
