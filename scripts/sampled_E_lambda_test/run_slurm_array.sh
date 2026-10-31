#!/bin/bash
#SBATCH --job-name=sampled_E_lambda_test
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-8

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
    "fourrooms-mines"
    "fourrooms-mines-dense"
    "eightrooms-dense"
    "eightrooms-misc"
    "whirlpool-misc"
    "SpaceInvadersExactValue"
    "mountaincar-dense"
)

TARGET_ARG=${1:-$SLURM_ARRAY_TASK_ID}
VALUE_LAMBDA=${VALUE_LAMBDA:-0.8}
NUM_UPDATES=${NUM_UPDATES:-50}
N_SEEDS=${N_SEEDS:-3}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-128}
NUM_STEPS=${NUM_STEPS:-128}
OUT_DIR=${OUT_DIR:-"results/sampled_E_lambda_test"}

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

echo "================================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Array Task:    ${SLURM_ARRAY_TASK_ID:-$TARGET_ARG}"
echo "Environment:   $ENV_NAME (Index: $ENV_IDX)"
echo "Value Lambda:  $VALUE_LAMBDA"
echo "Updates:       $NUM_UPDATES"
echo "Seeds:         $N_SEEDS"
echo "Trajectories:  $NUM_TRAJECTORIES trajs x $NUM_STEPS steps"
echo "Output Root:   $OUT_DIR"
echo "================================================================"

$PYTHON scripts/sampled_E_lambda_test/run_single_env.py \
    --env_name "$ENV_NAME" \
    --value_lambda "$VALUE_LAMBDA" \
    --num_updates "$NUM_UPDATES" \
    --n_seeds "$N_SEEDS" \
    --num_trajectories "$NUM_TRAJECTORIES" \
    --num_steps "$NUM_STEPS" \
    --out_dir "$OUT_DIR" \
    --save_metrics

echo "[DONE] Finished array task $TARGET_ARG for $ENV_NAME (lambda=$VALUE_LAMBDA)"
