#!/bin/bash
#SBATCH --job-name=sampled_algo_sweep
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-9

set -e
mkdir -p slurm

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Python interpreter resolution (cluster vs local)
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

ENVS=(
    "FourRooms-misc"
    "fourrooms-dense"
    "fourrooms-mines"
    "fourrooms-mines-dense"
    "eightrooms-misc"
    "eightrooms-dense"
    "whirlpool-misc"
    "MountainCar-v0"
    "mountaincar-dense"
    "SpaceInvadersExactValue"
)

# Allow manual index via $1 or SLURM_ARRAY_TASK_ID
TARGET_ARG=${1:-$SLURM_ARRAY_TASK_ID}
if [ -z "$TARGET_ARG" ]; then
    echo "No task index provided; running all tasks sequentially..."
    for task_idx in $(seq 0 $((${#ENVS[@]} - 1))); do
        "$0" "$task_idx"
    done
    exit 0
fi

if [[ "$TARGET_ARG" =~ ^[0-9]+$ ]]; then
    ENV_NAME="${ENVS[$TARGET_ARG]}"
else
    ENV_NAME="$TARGET_ARG"
fi

OUT_DIR=${OUT_DIR:-"results/sampled_algo_sweep"}
NUM_STEPS=${NUM_STEPS:-256}
NUM_ENVS=${NUM_ENVS:-64}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}
N_SEEDS=${N_SEEDS:-3}
SEED=${SEED:-42}
BASE_LR=${BASE_LR:-0.0003}

echo "=========================================================="
echo "Job ID:           ${SLURM_JOB_ID:-local}"
echo "Array Task ID:    ${SLURM_ARRAY_TASK_ID:-$TARGET_ARG}"
echo "Host:             $(hostname)"
echo "Environment:      $ENV_NAME"
echo "Rollout Steps:    $NUM_STEPS"
echo "Envs:             $NUM_ENVS"
echo "Timesteps:        $TOTAL_TIMESTEPS"
echo "Seeds:            $N_SEEDS (Base: $SEED)"
echo "Base Critic LR:   $BASE_LR"
echo "Output Directory: $OUT_DIR"
echo "=========================================================="

$PYTHON scripts/sampled_algo_sweep/run_single_env.py \
    --env_name "$ENV_NAME" \
    --num_steps "$NUM_STEPS" \
    --num_envs "$NUM_ENVS" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --n_seeds "$N_SEEDS" \
    --seed "$SEED" \
    --base_lr "$BASE_LR" \
    --lr_multipliers 0.5 1.0 2.0 \
    --out_dir "$OUT_DIR"

echo "Task completed successfully for $ENV_NAME"
