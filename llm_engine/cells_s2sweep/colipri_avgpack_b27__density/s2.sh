#!/usr/bin/env bash
# colipri_avgpack_b27/density STAGE2 LoRA+proj, 8 epochs (spe=4524, STEPS=36192), per-epoch full-valid eval
# warm-start from BEST s1 epoch step_031668 (s1 best_acc=0.7843)
set -euo pipefail
DEP="${1:-}"
MANIFEST="./vendor/npy_manifests/colipri.json" TOKEN_SELECTION=uniform_pool TOKEN_BUDGET=27 \
  TRAIN_VQA_JSON="/orange/anon/anon/3DCT/Compress_CT_Token/results/vqa/train_vqa_single_density.json" VALID_VQA_JSON="/orange/anon/anon/3DCT/Compress_CT_Token/results/vqa/valid_vqa_single_density.json" \
  RUN_ROOT=./results_llm/s1sweep_b27/vqa_single_colipri_avgpack_b27__density LABEL=vqa_single_colipri_avgpack_b27__density__s2 SAVE_EVERY=4524 EVAL_EVERY=4524 VALID_LIMIT=0 \
  PROJECTOR_ONLY=0 LR=2e-5 INIT_WEIGHTS_FROM_CHECKPOINT=./results_llm/s1sweep_b27/vqa_single_colipri_avgpack_b27__density/vqa_single_colipri_avgpack_b27__density__s1/checkpoints/step_031668 STEPS=36192 \
  sbatch --parsable --time=14:00:00 --job-name=s2_avgpack_density \
    ${DEP:+--dependency=afterok:$DEP} "./llm_engine/vqa_train_sc.sbatch"
