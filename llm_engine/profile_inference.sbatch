#!/bin/bash
#SBATCH --job-name=t6b_infer_profile
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:40:00
#SBATCH --account=anon
#SBATCH --qos=anon
#SBATCH --output=out_slurm/%x_%j.out
#SBATCH --error=out_slurm/%x_%j.err
set -euo pipefail
cd .
export MAMBA_EXE='/home/anon/micromamba'
export MAMBA_ROOT_PREFIX='/blue/anon/anon/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate b200
export TQDM_DISABLE=1
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python llm_engine/profile_inference.py --gen "${GEN:-128}" --reps "${REPS:-5}" \
    --passes "${PASSES:-4}" --warmup-passes "${WARMUP:-1}" --tag "${TAG:-}" \
    --budgets "${BUD:-8,27,64,216,13824}" \
    --out "${OUT:-./results_llm/t6b_inference_cost.json}"
