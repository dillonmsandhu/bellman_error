#!/bin/bash
#SBATCH --job-name=emp_var_inv
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1

set -e

# Working directory resolution: preserve submit directory under SLURM
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Create slurm logs directory
mkdir -p slurm

# Python interpreter resolution (cluster pyenv vs local fallback)
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    PYTHON="python3"
fi

# Determine task index (support manual argument for local debug execution)
TASK_ID=${SLURM_ARRAY_TASK_ID:-$1}
if [ -z "$TASK_ID" ]; then
    TASK_ID=0
fi

# Environment mapping for array: 0 -> FourRooms, 1 -> EightRooms
ENVS=(
    "FourRooms-misc"
    "eightrooms-misc"
)

ENV_NAME="${ENVS[$TASK_ID]}"

echo "=========================================================="
echo "Starting SLURM Empirical Variance Investigation Job"
echo "Job ID:      ${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
echo "Host:        $(hostname)"
echo "Environment: ${ENV_NAME}"
echo "Repo Root:   ${REPO_ROOT}"
echo "Python:      ${PYTHON}"
echo "Start Time:  $(date +"%Y-%m-%d %H:%M:%S")"
echo "=========================================================="

CMD="$PYTHON scripts/variance_investigation/run_empirical_variance_experiment.py \
    --env-name $ENV_NAME \
    --num-updates 150 \
    --num-envs 64 \
    --num-steps 64 \
    --lr 3e-4 \
    --seed 42"

echo "Executing: $CMD"
eval "$CMD"

echo "=========================================================="
echo "Job Complete at $(date +"%Y-%m-%d %H:%M:%S")"
echo "=========================================================="
