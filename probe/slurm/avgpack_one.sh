#!/usr/bin/env bash
#SBATCH --partition=hpg-turin
#SBATCH --account=anon
#SBATCH --qos=anon
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
#SBATCH --output=./out_slurm/%x_%j.out
#SBATCH --error=./out_slurm/%x_%j.err
set -euo pipefail
: "${CFG:?set CFG=<experiment yaml basename, e.g. exp_001_avgpack_r2_all5>}"
cd ./probing
export MAMBA_EXE='/home/anon/micromamba'; export MAMBA_ROOT_PREFIX='/blue/anon/anon/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate b200
export TQDM_DISABLE=1
nvidia-smi --query-gpu=name --format=csv,noheader; echo "Job $SLURM_JOB_ID ($CFG) on $SLURMD_NODENAME"
python run.py experiments/${CFG}.yaml
echo "[avgpack] DONE $CFG"
