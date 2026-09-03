#!/usr/bin/env bash
# Report generation — the only entry point you need.
#
#   bash llm_engine/run_reportgen.sh --smoke                       # ~15 min wiring check — do this FIRST
#   bash llm_engine/run_reportgen.sh --method ORCA --budget 216    # one full cell: s1 then s2 (~36 h)
#
#   method: ORCA | GridAvg        budget: 8 | 27 | 64 | 216
#
# Two stages:
#   s1  projector only, 8 epochs, LoRA frozen        -> pick its best epoch by clinical F1
#   s2  LoRA + projector, 8 epochs, warm-started from that best s1 epoch
# Both evaluate on the FULL validation set after every epoch. See TRAINING.md for the recipe and costs.
#
# Every path the engine could otherwise default is passed EXPLICITLY, and the package is verified before a
# GPU-hour is spent: a default that silently resolves to the wrong place is how this project once lost three
# weeks of runs to a superseded label set. Pass everything, check first, fail loudly.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$HERE/llm_engine"; DATA="$HERE/data"; W="$HERE/checkpoints"

METHOD=""; BUDGET=""; SMOKE=0; ONLY=""; MAXSTEPS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --stage)  ONLY="$2"; shift 2 ;;          # s1 | s2 — optional; default runs both in order
    --max-steps) MAXSTEPS="$2"; shift 2 ;;   # stop at this CUMULATIVE step count and exit (chain links)
    --smoke)  SMOKE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ $SMOKE -eq 1 ]] && { METHOD="${METHOD:-ORCA}"; BUDGET="${BUDGET:-27}"; }
[[ -n "$METHOD" && -n "$BUDGET" ]] || { echo "need --method {ORCA|GridAvg} --budget {8|27|64|216}" >&2; exit 2; }
[[ "$METHOD" =~ ^(ORCA|GridAvg)$ ]] || { echo "bad --method: $METHOD (ORCA | GridAvg)" >&2; exit 2; }
[[ "$BUDGET" =~ ^(8|27|64|216)$ ]] || { echo "bad --budget: $BUDGET (8 | 27 | 64 | 216)" >&2; exit 2; }

MANIFEST="$DATA/embeddings/manifest_${METHOD}_b${BUDGET}.json"
CELL="$HERE/results_llm/reportgen_${METHOD}_b${BUDGET}"

# ---- verify before spending a GPU-hour ------------------------------------------------------------------
fail=0
for p in "$CODE/vqa_train.py" "$CODE/zero1_author_reportgen.dsconfig" \
         "$CODE/base/llava_config/config.json" "$CODE/base/llava_config/adapter_config.json" \
         "$CODE/llava/model/multimodal_projector/projector.py" \
         "$MANIFEST" \
         "$DATA/vqa/train_reportgen.json" "$DATA/vqa/valid_reportgen.json" \
         "$DATA/reports/validation_reports.csv" "$DATA/labels/valid_predicted_labels.csv" \
         "$W/Llama-3.1-8B-Instruct" "$W/RadBertClassifier.pth"; do
  [[ -e "$p" ]] || { echo "MISSING: $p" >&2; fail=1; }
done
if [[ ! -f "$DATA/vqa/train_reportgen.json" || ! -f "$DATA/reports/validation_reports.csv" ]]; then
  echo "  -> the reports, labels and question set are CT-RATE's own files and are not mirrored with the" >&2
  echo "     token bundles. Accept CT-RATE's terms on the Hub, then:  python download.py --annotations" >&2
fi
[[ $fail -eq 0 ]] || { echo "package is incomplete — do not proceed" >&2; exit 1; }

# The manifest points at the stacked token arrays; check them and report what we are about to train on.
python3 - "$MANIFEST" <<'PY' || exit 1
import json, sys, pathlib, numpy as np
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
for split in ("train", "valid"):
    se = m["splits"][split]
    arr, ids = pathlib.Path(se["array"]), pathlib.Path(se["ids"])
    for p in (arr, ids):
        if not p.exists():
            sys.exit(f"MISSING: {p}")
    a = np.load(arr, mmap_mode="r")
    n_ids = len([l for l in ids.read_text().splitlines() if l.strip()])
    if a.shape[0] != n_ids:
        sys.exit(f"FATAL: {arr} has {a.shape[0]} rows but {ids} has {n_ids} ids")
    print(f"[check] {split}: {a.shape[0]} volumes, {a.shape[1]} tokens x {a.shape[2]}-d ({a.dtype})")
PY

export PYTHONPATH="$CODE/core_code:$CODE:${PYTHONPATH:-}"   # core_code + the llava package
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# 24,128 volumes, one report each, effective batch 16 -> 1508 steps/epoch.
SPE=1508; EPOCHS=8

# --------------------------------------------------------------------------------------------------------
run_stage () {
  local st="$1"; shift
  local run="$CELL/reportgen_${METHOD}_b${BUDGET}__${st}"
  local steps=$((SPE * EPOCHS)) spe="$SPE" vlim=0 lr="5e-4"
  # --max-steps is a CUMULATIVE total; the runner auto-resumes from checkpoints/training_state_latest.pt,
  # so link k continues where k-1 stopped instead of restarting.
  [[ -n "$MAXSTEPS" ]] && steps="$MAXSTEPS"
  local extra=()
  mkdir -p "$run"

  if [[ "$st" == "s1" ]]; then
    extra+=(--projector-only)                       # LoRA frozen; train the projector only
  else
    lr="2e-5"                                       # LoRA + projector: much smaller LR
    local s1dir="$CELL/reportgen_${METHOD}_b${BUDGET}__s1"
    # Report generation writes metrics_fast.json, not the vqa_metrics.json that score_sweep.py reads, so it
    # produces no best.json on its own. pick_best_epoch.py makes one, selecting by clinical F1.
    python3 "$CODE/pick_best_epoch.py" --run_dir "$s1dir" || {
      echo "FATAL: could not pick s1's best epoch — run s1 to completion first" >&2; return 1; }
    local best; best="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_step'])" \
                        "$s1dir/best.json")"
    [[ -n "$best" ]] || { echo "FATAL: s1 best.json unreadable" >&2; return 1; }
    local init="$s1dir/checkpoints/$best"
    [[ -d "$init" ]] || { echo "FATAL: missing s1 checkpoint $init" >&2; return 1; }
    echo "[s2] warm-starting from s1 $best"
    extra+=(--init-weights-from-checkpoint "$init")
  fi

  if [[ $SMOKE -eq 1 ]]; then
    steps=30; spe=999999; vlim=8; extra+=(--skip-final-eval)
    run="$CELL/_smoke__${st}"; mkdir -p "$run"
    echo "[smoke] 30 steps, 8 valid volumes — proving the wiring only"
  fi

  echo "[run] $st  method=$METHOD budget=$BUDGET steps=$steps lr=$lr -> $run"
  deepspeed --num_gpus=1 --master_port "$((29000 + RANDOM % 1000))" "$CODE/vqa_train.py" \
    --reportgen-artifact-manifest "$MANIFEST" \
    --token-selection none --token-budget 0 \
    --out-dir "$run" --run-name "reportgen_${METHOD}_b${BUDGET}__${st}" --steps "$steps" \
    --train-limit 0 --valid-limit "$vlim" --batch-size 16 --eval-batch-size 1 \
    `# --eval-batch-size stays 1. Evaluation is ~10x the training cost, so batching it is the obvious` \
    `# speed-up, and it is deliberately not taken: generation runs with padding_side="left", and batched` \
    `# left-padded generation does not fail loudly -- it quietly produces slightly worse reports, which` \
    `# lands straight in the clinical F1 we are trying to measure. Do not change it.` \
    --gradient-accumulation-steps 1 --num-workers 8 \
    --lr "$lr" --mm-projector-lr "$lr" --weight-decay 0.0 --warmup-ratio 0.03 \
    --save-every "$spe" --eval-every "$spe" --device cuda:0 --seed 2026 \
    --max-new-tokens 512 \
    --model-path "$CODE/base/llava_config" --model-base "$W/Llama-3.1-8B-Instruct" --init-from-scratch \
    --record-type report_generation \
    --train-vqa-json "$DATA/vqa/train_reportgen.json" \
    --valid-vqa-json "$DATA/vqa/valid_reportgen.json" \
    --valid-reports-csv "$DATA/reports/validation_reports.csv" \
    --valid-labels-csv "$DATA/labels/valid_predicted_labels.csv" \
    --radbert-checkpoint "$W/RadBertClassifier.pth" \
    --llava-repo "$CODE" \
    --deepspeed-config "$CODE/zero1_author_reportgen.dsconfig" \
    "${extra[@]}" "$@" 2>&1 | tee -a "$run/run.log"
}

# --skip-metrics is deliberately NOT passed. With it the engine only keeps raw predictions for an external
# scorer. Without it, it converts predictions against reports/validation_reports.csv and runs eval_fast.py
# -> RadBERT clinical F1 + BLEU/ROUGE-L/METEOR/CIDEr + CRG into evaluations/step_*/metrics_fast.json.
# That file is the result.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
case "${ONLY:-both}" in
  s1) run_stage s1 ;;
  s2) run_stage s2 ;;
  *)  run_stage s1 && run_stage s2 ;;
esac
echo "[done] $METHOD b$BUDGET -> $CELL"
