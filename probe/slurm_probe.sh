#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --account=anon
#SBATCH --qos=anon
#SBATCH --output=out_slurm/%x_%j.out
#SBATCH --error=out_slurm/%x_%j.err
#
# Generic probing runner:  sbatch --job-name=<tag> probing/slurm_probe.sh <exp_id>
# Reads probing/experiments/<exp_id>.yaml. Kept as a FILE on purpose -- the earlier sweeps were submitted
# from throwaway heredocs, so when a job needed re-running the exact command no longer existed anywhere.
set -euo pipefail
EXP="${1:?usage: sbatch probing/slurm_probe.sh <exp_id>}"
cd .

export MAMBA_EXE='/home/anon/micromamba'
export MAMBA_ROOT_PREFIX='/blue/anon/anon/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate b200
export TQDM_DISABLE=1

CFG="probing/experiments/${EXP}.yaml"
[ -f "$CFG" ] || { echo "no such config: $CFG"; exit 2; }
echo "Job ${SLURM_JOB_ID} (${EXP}) on ${SLURMD_NODENAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd probing && python run.py "../${CFG}"
EXIT_CODE=$?
echo "[exit] ${EXIT_CODE}"
exit $EXIT_CODE
