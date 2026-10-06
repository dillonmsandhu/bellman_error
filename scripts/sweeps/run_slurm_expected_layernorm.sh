#!/bin/bash
#SBATCH --job-name=expected_ln_sweep
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-6

set -e

# Ensure working directory is repo root
if [ -n "$REPO_ROOT" ] && [ -f "$REPO_ROOT/core/config.py" ]; then
    cd "$REPO_ROOT"
elif [ -f "core/config.py" ]; then
    : # already in repo root
else
    cd "$(dirname "$0")/../.."
fi
REPO_ROOT="$(pwd)"
mkdir -p slurm
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Python interpreter resolution
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

ENVS=(
    "fourrooms-dense"
    "FourRooms-misc"
    "eightrooms-dense"
    "eightrooms-misc"
    "whirlpool-misc"
    "SpaceInvadersExactValue"
    "mountaincar-dense"
)

# Allow manual index via $1 or SLURM_ARRAY_TASK_ID
TARGET_ARG=${1:-$SLURM_ARRAY_TASK_ID}
if [ -z "$TARGET_ARG" ]; then
    echo "No task index provided; running all 7 tasks sequentially..."
    for task_idx in $(seq 0 $((${#ENVS[@]} - 1))); do
        "$0" "$task_idx"
    done
    exit 0
fi

if [[ "$TARGET_ARG" =~ ^[0-9]+$ ]]; then
    ENV_NAME="${ENVS[$TARGET_ARG]}"
    ENV_IDX="$TARGET_ARG"
else
    ENV_NAME="$TARGET_ARG"
    ENV_IDX="-1"
fi

echo "======================================================================"
echo "LAUNCHING EXPECTED PPO LAYER NORM & LR SWEEP"
echo "Job ID:           ${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-manual}"
echo "Host:             $(hostname)"
echo "Target Env Index: $ENV_IDX"
echo "Target Env Name:  $ENV_NAME"
echo "Python:           $PYTHON"
echo "Working Dir:      $(pwd)"
echo "======================================================================"

$PYTHON scripts/sweeps/sweep_expected_layernorm.py \
    --env_name "$ENV_NAME" \
    --algos exact_E exact_td exact_mc \
    --lr_grid 0.005 0.001 0.0003 0.0001 \
    --num_updates 300 \
    --num_seeds 4 \
    --seed 42 \
    --window_size 30

echo "======================================================================"
echo "Task finished successfully for $ENV_NAME"
echo "======================================================================"
