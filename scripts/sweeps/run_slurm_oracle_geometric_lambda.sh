#!/bin/bash
#SBATCH --job-name=oracle_geom_e
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-6

set -e

# Ensure working directory is repo root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
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

TARGET_ARG=${1:-$SLURM_ARRAY_TASK_ID}
VALUE_LAMBDA=${VALUE_LAMBDA:-0.9}

if [ -z "$TARGET_ARG" ]; then
    echo "No task index provided; running all tasks sequentially..."
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
echo "LAUNCHING GEOMETRIC SAMPLED E(lambda) ORACLE VALIDATION"
echo "Job ID:           ${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-manual}"
echo "Target Env Index: $ENV_IDX"
echo "Target Env Name:  $ENV_NAME"
echo "Value Lambda:     $VALUE_LAMBDA"
echo "Python:           $PYTHON"
echo "======================================================================"

$PYTHON scripts/sweeps/sweep_oracle_targets_e_lambda_geometric.py \
    --env_name "$ENV_NAME" \
    --value_lambda "$VALUE_LAMBDA" \
    --num_steps 128 \
    --num_envs 128 \
    --num_updates 35 \
    --num_seeds 4 \
    --seed 42

echo "======================================================================"
echo "Task finished successfully for $ENV_NAME (lambda=$VALUE_LAMBDA)"
echo "======================================================================"
