#!/bin/bash
#SBATCH --job-name=green_eval
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50gb
#SBATCH --time=08:00:00
#SBATCH --account=anon
#SBATCH --output=out_slurm/green_%x_%j.out
#SBATCH --error=out_slurm/green_%x_%j.err
#
# Usage:
#   sbatch eval/green/run_green.sh <predictions.jsonl> <out_dir>
#
# Scores one report-gen eval step with the StanfordAIMI GREEN metric on a
# single L4 GPU (GREEN-radllama2-7b, fp16 ~13.5GB, fits 24GB L4).
#
# Weights are loaded from a local cache (no download):
#   ./checkpoints/GREEN-RadLlama2-7b
# The mpnet summary/clustering is disabled in score_green.py, so no other
# model download is needed.

set -euo pipefail

PRED="$1"
OUT="$2"

export MAMBA_EXE='/home/anon/micromamba'
export MAMBA_ROOT_PREFIX='/blue/anon/anon/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate b200
export TQDM_DISABLE=1
export HF_HUB_OFFLINE=1          # weights pre-fetched on login node
export TOKENIZERS_PARALLELISM=false

cd .

echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
nvidia-smi
echo "PRED=$PRED"
echo "OUT=$OUT"

python -m eval.green.score_green --pred "$PRED" --out "$OUT"
EXIT_CODE=$?
echo "Done. exit=$EXIT_CODE"
exit $EXIT_CODE
