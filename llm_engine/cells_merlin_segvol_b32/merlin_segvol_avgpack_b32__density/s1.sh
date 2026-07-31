#!/usr/bin/env bash
# merlin_segvol_avgpack_b32/density STAGE1 projector, 8 epochs (spe=2838, STEPS=22704), per-epoch full-valid eval
set -euo pipefail
DEP="${1:-}"
MANIFEST="./vendor/npy_manifests/merlin_segvol_avgpool_b32.json" TOKEN_SELECTION=none TOKEN_BUDGET=0 \
  TRAIN_VQA_JSON="./data/vqa_merlin_v1/per_family/train_density_merlin.json" VALID_VQA_JSON="./data/vqa_merlin_v1/per_family/valid_density_merlin.json" \
  RUN_ROOT=./results_llm/merlin_segvol_b32/vqa_single_merlin_segvol_avgpack_b32__density LABEL=vqa_single_merlin_segvol_avgpack_b32__density__s1 SAVE_EVERY=2838 EVAL_EVERY=2838 VALID_LIMIT=0 \
  PROJECTOR_ONLY=1 LR=5e-4 STEPS=22704 \
  sbatch --parsable --time=10:00:00 --job-name=ms1_segvol_avgpack_b32_density \
    ${DEP:+--dependency=afterok:$DEP} "./llm_engine/vqa_train_sc.sbatch"
