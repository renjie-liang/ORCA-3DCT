#!/usr/bin/env bash
# colipri_orcafull_b64/density STAGE2 LoRA+proj, 8 epochs (spe=6021, STEPS=48168), per-epoch full-valid eval
# warm-start from BEST s1 epoch step_036126 (s1 best_acc=0.8467)
set -euo pipefail
DEP="${1:-}"
MANIFEST="./vendor/npy_manifests/colipri_orcafull_lam0p5_f4s2_b64.json" TOKEN_SELECTION=none TOKEN_BUDGET=0 \
  TRAIN_VQA_JSON="./data/vqa/per_family/train_density.json" VALID_VQA_JSON="./data/vqa/per_family/valid_density.json" \
  RUN_ROOT=./results_llm/s1sweep_b64/vqa_single_colipri_orcafull_b64__density LABEL=vqa_single_colipri_orcafull_b64__density__s2 SAVE_EVERY=6021 EVAL_EVERY=6021 VALID_LIMIT=0 \
  PROJECTOR_ONLY=0 LR=2e-5 INIT_WEIGHTS_FROM_CHECKPOINT=./results_llm/s1sweep_b64/vqa_single_colipri_orcafull_b64__density/vqa_single_colipri_orcafull_b64__density__s1/checkpoints/step_036126 STEPS=48168 \
  sbatch --parsable --time=14:00:00 --job-name=s2_orcafull_b64_density \
    ${DEP:+--dependency=afterok:$DEP} "./llm_engine/vqa_train_sc.sbatch"
