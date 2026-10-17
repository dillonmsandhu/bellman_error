#!/bin/bash
#SBATCH --job-name=exact_ppo_sweep
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --array=0-8
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e
mkdir -p slurm

# 1. Resolve working directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -n "$REPO_ROOT" ] && [ -d "$REPO_ROOT" ]; then
    cd "$REPO_ROOT"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 2. Resolve Python interpreter
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

ENVS=(
    "mountaincar"
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
NUM_UPDATES=${NUM_UPDATES:-100}
N_SEEDS=${N_SEEDS:-10}
SEED=${SEED:-42}
K_VAL=${K_VAL:-32}
ACTOR_LR=${ACTOR_LR:-0.0003}
CRITIC_LRS=${CRITIC_LRS:-"0.001 0.005 0.01"}
GAE_LAMBDA=${GAE_LAMBDA:-0.2}
CLIP_EPS=${CLIP_EPS:-0.1}
VAL_LAMBDA=${VAL_LAMBDA:-0.8}
OUT_DIR=${OUT_DIR:-"results/exact_ppo_sweep"}

# Sequential fallback if executed directly without args
if [ -z "$TARGET_ARG" ]; then
    echo "No task index provided; running all 9 environments sequentially..."
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
echo "Critic k:      $K_VAL"
echo "Actor LR:      $ACTOR_LR"
echo "Critic LRs:    $CRITIC_LRS"
echo "GAE Lambda:    $GAE_LAMBDA"
echo "Clip Epsilon:  $CLIP_EPS"
echo "Value Lambda:  $VAL_LAMBDA"
echo "Output Dir:    $OUT_DIR"
echo "Start Time:    $(date)"
echo "================================================================"

$PYTHON scripts/exact_ppo_sweep/run_single_env.py \
    --env_name "$ENV_NAME" \
    --num_updates "$NUM_UPDATES" \
    --n_seeds "$N_SEEDS" \
    --seed "$SEED" \
    --k "$K_VAL" \
    --actor_lr "$ACTOR_LR" \
    --critic_lrs $CRITIC_LRS \
    --gae_lambda "$GAE_LAMBDA" \
    --clip_eps "$CLIP_EPS" \
    --val_lambda "$VAL_LAMBDA" \
    --out_dir "$OUT_DIR"

echo "Completed Exact PPO task for $ENV_NAME at $(date)"
