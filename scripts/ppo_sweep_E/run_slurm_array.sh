#!/bin/bash
#SBATCH --job-name=ppo_sweep_E
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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

NUM_UPDATES=${NUM_UPDATES:-100}
N_SEEDS=${N_SEEDS:-3}
NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-128}
OUT_DIR=${OUT_DIR:-"results/ppo_sweep_E"}

echo "================================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Array Task:    ${SLURM_ARRAY_TASK_ID:-$TARGET_ARG}"
echo "Environment:   $ENV_NAME"
echo "Host:          $(hostname)"
echo "Python:        $PYTHON"
echo "Num Updates:   $NUM_UPDATES"
echo "Num Seeds:     $N_SEEDS"
echo "Rollout:       ${NUM_ENVS} envs x ${NUM_STEPS} steps"
echo "Output Dir:    $OUT_DIR"
echo "================================================================"

$PYTHON scripts/ppo_sweep_E/run_single_env.py \
    --env_name "$ENV_NAME" \
    --num_updates "$NUM_UPDATES" \
    --n_seeds "$N_SEEDS" \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --out_dir "$OUT_DIR"

echo "Environment $ENV_NAME run completed successfully."
