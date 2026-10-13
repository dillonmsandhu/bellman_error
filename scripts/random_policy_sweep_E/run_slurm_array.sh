#!/bin/bash
#SBATCH --job-name=rand_pol_E
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --array=0-8
#SBATCH --time=02:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# 1. Resolve repository root
if [ -n "$REPO_ROOT" ] && [ -d "$REPO_ROOT" ]; then
    cd "$REPO_ROOT"
elif [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"

# 2. Resolve Python interpreter
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
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
NUM_UPDATES=${NUM_UPDATES:-50}
N_SEEDS=${N_SEEDS:-3}
NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-1024}
OUT_DIR=${OUT_DIR:-"results/random_policy_sweep_E"}

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
echo "Host:          $(hostname)"
echo "Python:        $PYTHON"
echo "Num Updates:   $NUM_UPDATES"
echo "Num Seeds:     $N_SEEDS"
echo "Rollout Config: $NUM_ENVS envs x $NUM_STEPS steps"
echo "Output Dir:    $OUT_DIR"
echo "================================================================"

$PYTHON scripts/random_policy_sweep_E/run_single_env.py \
    --env_name "$ENV_NAME" \
    --num_updates "$NUM_UPDATES" \
    --n_seeds "$N_SEEDS" \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --out_dir "$OUT_DIR"

echo "Completed Random Policy Sampled E task for $ENV_NAME at $(date)"
