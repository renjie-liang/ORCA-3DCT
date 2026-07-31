#!/usr/bin/env bash
# colipri_avgpack_b27/density STAGE1 projector, 8 epochs (spe=4524, STEPS=36192), per-epoch full-valid eval
set -euo pipefail
CT=.; DEP="${1:-}"
MANIFEST="./vendor/npy_manifests/colipri.json" TOKEN_SELECTION=uniform_pool TOKEN_BUDGET=27 \
  TRAIN_VQA_JSON="/orange/anon/anon/3DCT/Compress_CT_Token/results/vqa/train_vqa_single_density.json" VALID_VQA_JSON="/orange/anon/anon/3DCT/Compress_CT_Token/results/vqa/valid_vqa_single_density.json" \
  RUN_ROOT=./results_llm/s1sweep_b27/vqa_single_colipri_avgpack_b27__density LABEL=vqa_single_colipri_avgpack_b27__density__s1 SAVE_EVERY=4524 EVAL_EVERY=4524 VALID_LIMIT=0 \
  PROJECTOR_ONLY=1 LR=5e-4 STEPS=36192 \
  sbatch --parsable --time=07:00:00 --job-name=s1_avgpack_density \
    ${DEP:+--dependency=afterok:$DEP} "./llm_engine/vqa_train_sc.sbatch"
