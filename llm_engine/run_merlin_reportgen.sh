#!/usr/bin/env bash
# Merlin report generation, ONE cell. Test harness first (--smoke), then a real run.
#
#   bash llm_engine/run_merlin_reportgen.sh --method orca --budget 256 --smoke   # 30 steps, 8 vols, wiring only
#   bash llm_engine/run_merlin_reportgen.sh --method orca --budget 256           # real s1
#
# ISOLATED from the CT-RATE run_reportgen.sh (that one is wired to the handoff package's paths + RadBERT).
# Key differences for Merlin:
#   * SegVol tokens (our precomputed VQA caches -- report gen shares them).
#   * report json from data/vqa_merlin_v1/reportgen/ (findings-only, cleaned).
#   * --skip-metrics: Merlin has NO RadBERT-style labeler, so the engine only writes raw generations
#     (raw_btb3d_output.jsonl); RadGraph-F1 is scored SEPARATELY afterwards by score_merlin_reportgen.py.
#     This is why no --radbert-checkpoint / --valid-labels-csv are passed.
#
# Same engine as VQA (llm_engine/vqa_train.py), driven by flags -- no engine change.
set -uo pipefail

CTOKEN=.
ENG="$CTOKEN/llm_engine"
CODE="$ENG/core_code"
MANDIR=./vendor/npy_manifests
RG="$CTOKEN/data/vqa_merlin_v1/reportgen"
W=./checkpoints
MODELPATH="$ENG/base/checkpoint-38000"   # local LLaVA checkpoint (same as CT-RATE VQA)
ENCODER=segvol

METHOD=""; BUDGET=""; SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --smoke)  SMOKE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$METHOD" && -n "$BUDGET" ]] || { echo "need --method {avgpack|orca} --budget {32|256}" >&2; exit 2; }
[[ "$METHOD" =~ ^(avgpack|orca)$ ]] || { echo "bad --method: $METHOD" >&2; exit 2; }

if [[ "$METHOD" == orca ]]; then
  MANIFEST="$MANDIR/merlin_${ENCODER}_orcafull_lam2_f4s2_b${BUDGET}.json"
else
  MANIFEST="$MANDIR/merlin_${ENCODER}_avgpool_b${BUDGET}.json"
fi
CELL="$CTOKEN/results_llm/merlin_reportgen_${ENCODER}_${METHOD}_b${BUDGET}"

echo "=== preflight ==="
fail=0
for p in "$ENG/vqa_train.py" "$CODE" "$MANIFEST" \
         "$RG/train_reportgen.json" "$RG/valid_reportgen.json" "$RG/valid_reference_reports.csv" \
         "$W/Llama-3.1-8B-Instruct" "$MODELPATH/config.json"; do
  if [[ -e "$p" ]]; then echo "  ok   $p"; else echo "  MISSING $p" >&2; fail=1; fi
done
[[ $fail -eq 0 ]] || { echo "preflight failed" >&2; exit 1; }

steps=8000; spe=8000; vlim=0; extra=(); run="$CELL/s1"
if [[ $SMOKE -eq 1 ]]; then
  steps=30; spe=999999; vlim=8; extra+=(--skip-final-eval); run="$CELL/_smoke"
  echo "[smoke] 30 steps, 8 valid volumes -- proving the wiring only"
fi
mkdir -p "$run"

echo "=== run: method=$METHOD budget=$BUDGET encoder=$ENCODER steps=$steps -> $run ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

deepspeed --num_gpus=1 --master_port "$((29000 + RANDOM % 1000))" "$ENG/vqa_train.py" \
  --reportgen-artifact-manifest "$MANIFEST" \
  --token-selection none --token-budget 0 \
  --out-dir "$run" --run-name "merlin_reportgen_${METHOD}_b${BUDGET}" --steps "$steps" \
  --train-limit 0 --valid-limit "$vlim" --batch-size 16 --eval-batch-size 1 \
  --gradient-accumulation-steps 1 --num-workers 8 \
  --lr 5e-4 --mm-projector-lr 5e-4 --weight-decay 0.0 --warmup-ratio 0.03 \
  --save-every "$spe" --eval-every "$spe" --device cuda:0 --seed 2026 \
  --max-new-tokens 512 \
  --model-path "$MODELPATH" --model-base "$W/Llama-3.1-8B-Instruct" --init-from-scratch \
  --record-type report_generation \
  --train-vqa-json "$RG/train_reportgen.json" \
  --valid-vqa-json "$RG/valid_reportgen.json" \
  --projector-only \
  --skip-metrics \
  --deepspeed-config "$ENG/zero1_author_reportgen.dsconfig" \
  "${extra[@]}" 2>&1 | tee -a "$run/run.log"

echo "=== done. raw generations -> $run/.../raw_btb3d_output.jsonl ==="
echo "next: python llm_engine/score_merlin_reportgen.py --run $run --ref $RG/valid_reference_reports.csv"
