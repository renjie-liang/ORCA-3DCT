#!/usr/bin/env bash
#SBATCH --job-name=vqa-train
#SBATCH --partition=hpg-b200
#SBATCH --account=anon
#SBATCH --qos=anon
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=./results_llm/slurm_logs/%x_%j.out
#SBATCH --error=./results_llm/slurm_logs/%x_%j.err
#
# SELF-CONTAINED VQA training engine (2026-07-15). Same as scripts_probing/vqa/vqa_train.sbatch but with ALL
# DTBD3D references vendored into CompressToken/llm_engine: engine py + dtbd3d pkg + dsconfig + base checkpoint.
# Only the shared Llama-3.1-8B base + the (self-owned) Compress_CT_Token VQA jsons/manifests stay referenced.
# Caller env (per cell/stage): MANIFEST TOKEN_SELECTION TOKEN_BUDGET RUN_ROOT LABEL LR STEPS SAVE_EVERY
#   PROJECTOR_ONLY INIT_WEIGHTS_FROM_CHECKPOINT TRAIN_VQA_JSON VALID_VQA_JSON
set -uo pipefail
: "${MANIFEST:?}"; : "${TOKEN_SELECTION:?}"; : "${TOKEN_BUDGET:?}"
: "${RUN_ROOT:?}"; : "${LABEL:?}"; : "${LR:?}"; : "${STEPS:?}"; : "${SAVE_EVERY:?}"
# REQUIRED, deliberately no default (2026-07-16). These used to default to Compress_CT_Token/results/vqa/*,
# a set superseded on 2026-07-13 -- and because a default cannot fail, three weeks of runs trained on stale
# labels while looking perfectly healthy. Fail loudly instead: a wrong/missing path must abort the job.
: "${TRAIN_VQA_JSON:?set TRAIN_VQA_JSON (data/vqa/per_family/train_<fam>.json)}"
: "${VALID_VQA_JSON:?set VALID_VQA_JSON (data/vqa/per_family/valid_<fam>.json)}"
for _j in "$TRAIN_VQA_JSON" "$VALID_VQA_JSON"; do
  [[ -f "$_j" ]] || { echo "FATAL: vqa json not found: $_j" >&2; exit 1; }
done
PROJECTOR_ONLY="${PROJECTOR_ONLY:-0}"
INIT_WEIGHTS_FROM_CHECKPOINT="${INIT_WEIGHTS_FROM_CHECKPOINT:-}"

CTOKEN=.
CT=/orange/anon/anon/3DCT/Compress_CT_Token       # self-owned VQA jsons/manifests (NOT DTBD3D)
ENG=$CTOKEN/llm_engine                                     # vendored engine (dtbd3d pkg + copies + base ckpt)
cd "$CTOKEN"

export MAMBA_EXE='/home/anon/micromamba'; export MAMBA_ROOT_PREFIX='/blue/anon/anon/micromamba'
set +eu; eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"; micromamba activate b200; set -u
export PYTHONPATH="$ENG/core_code:$ENG/core_code/btb3d_baseline/report-generation:${PYTHONPATH:-}"  # vendored dtbd3d + llava
export PYTHONUNBUFFERED=1
mkdir -p "$CTOKEN/results_llm/slurm_logs" "$CTOKEN/tmp/cache/triton" "$CTOKEN/tmp/cache/mpl"
export TRITON_CACHE_DIR="$CTOKEN/tmp/cache/triton" MPLCONFIGDIR="$CTOKEN/tmp/cache/mpl"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 40000)))}"
ZERO_CONFIG="$ENG/zero1_author_reportgen.dsconfig"
MODEL_PATH="$ENG/base/checkpoint-38000"

RUN_DIR="$RUN_ROOT/$LABEL"; LOG="$RUN_DIR/run.log"; mkdir -p "$RUN_DIR"

RESUME_ARGS=(); INIT_WEIGHTS_ARGS=()
if [[ -e "$RUN_DIR/checkpoints/training_state_latest.pt" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$RUN_DIR/checkpoints/training_state_latest.pt")
elif [[ -n "$INIT_WEIGHTS_FROM_CHECKPOINT" ]]; then
  [[ -e "$INIT_WEIGHTS_FROM_CHECKPOINT" ]] || { echo "Missing init weights: $INIT_WEIGHTS_FROM_CHECKPOINT" >&2; exit 1; }
  INIT_WEIGHTS_ARGS=(--init-weights-from-checkpoint "$INIT_WEIGHTS_FROM_CHECKPOINT")
fi
PROJECTOR_ARGS=(); [[ "$PROJECTOR_ONLY" == 1 ]] && PROJECTOR_ARGS=(--projector-only)

{
  date; hostname
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}  LABEL=$LABEL  STEPS=$STEPS  LR=$LR  PROJECTOR_ONLY=$PROJECTOR_ONLY"
  echo "ENGINE=$ENG (self-contained)  MANIFEST=$MANIFEST  TOKEN=$TOKEN_SELECTION/$TOKEN_BUDGET  RUN_DIR=$RUN_DIR"
  echo "RESUME=${RESUME_ARGS[*]:-none}  INIT_WEIGHTS=${INIT_WEIGHTS_ARGS[*]:-none}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
} | tee "$LOG"

deepspeed --num_gpus=1 --master_port "$MASTER_PORT" "$ENG/vqa_train.py" \
  --reportgen-artifact-manifest "$MANIFEST" \
  --out-dir "$RUN_DIR" --run-name "$LABEL" --steps "$STEPS" \
  --train-limit 0 --valid-limit "${VALID_LIMIT:-8}" --batch-size 16 --eval-batch-size 1 \
  --gradient-accumulation-steps 1 --num-workers 8 \
  --lr "$LR" --mm-projector-lr "$LR" --weight-decay 0.0 --warmup-ratio 0.03 \
  --save-every "$SAVE_EVERY" --eval-every "${EVAL_EVERY:-99999}" --device cuda:0 --seed "${SEED:-2026}" \
  --max-new-tokens 16 --model-path "$MODEL_PATH" --init-from-scratch \
  "${PROJECTOR_ARGS[@]}" --skip-final-eval "${INIT_WEIGHTS_ARGS[@]}" \
  --token-selection "$TOKEN_SELECTION" --token-budget "$TOKEN_BUDGET" \
  --train-vqa-json "$TRAIN_VQA_JSON" \
  --valid-vqa-json "$VALID_VQA_JSON" \
  --record-type ig --multi-qa-per-volume --skip-metrics \
  --deepspeed-config "$ZERO_CONFIG" "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
echo "RUN_DIR=$RUN_DIR"
