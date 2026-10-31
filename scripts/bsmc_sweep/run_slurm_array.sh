#!/bin/bash
#SBATCH --job-name=bsmc_sweep
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:30:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-8

set -e
mkdir -p slurm

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Python interpreter resolution chain
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

# Overridable hyperparameters
N_SEEDS=${N_SEEDS:-10}
NUM_UPDATES=${NUM_UPDATES:-100}
K_VAL=${K_VAL:-32}
ACTOR_LR=${ACTOR_LR:-0.0003}
CRITIC_LRS=${CRITIC_LRS:-"0.0005 0.001 0.003"}
LAMBDAS=${LAMBDAS:-"0.4 0.8 0.95 0.99"}
ALGOS=${ALGOS:-"td_lambda e_lambda e_lambda_fwd bsmc_aux bsmc_fwd"}
NUM_ENVS=${NUM_ENVS:-32}
NUM_STEPS=${NUM_STEPS:-256}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-256}
NUM_EPOCHS=${NUM_EPOCHS:-4}
GAE_LAMBDA=${GAE_LAMBDA:-0.8}
CLIP_EPS=${CLIP_EPS:-0.2}
RUNWAY_LENGTH=${RUNWAY_LENGTH:-adaptive}
OUT_DIR=${OUT_DIR:-"results/bsmc_sweep"}

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

echo "=========================================================="
echo "Job ID:           ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-$TARGET_ARG}"
echo "Environment:      $ENV_NAME"
echo "Num Updates:      $NUM_UPDATES"
echo "Num Seeds:        $N_SEEDS"
echo "Critic Feature k: $K_VAL"
echo "Actor LR:         $ACTOR_LR (fixed)"
echo "Critic LRs:       $CRITIC_LRS"
echo "Lambdas:          $LAMBDAS"
echo "Algorithms:       $ALGOS"
echo "Actor GAE Lambda: $GAE_LAMBDA"
echo "Clip Epsilon:     $CLIP_EPS"
echo "Rollout Config:   ${NUM_ENVS} envs x ${NUM_STEPS} steps | MB: ${MINIBATCH_SIZE} | Epochs: ${NUM_EPOCHS}"
echo "Python:           $PYTHON"
echo "Output Directory: $OUT_DIR"
echo "=========================================================="

$PYTHON scripts/bsmc_sweep/run_single_env.py \
    --env_name "$ENV_NAME" \
    --num_updates "$NUM_UPDATES" \
    --n_seeds "$N_SEEDS" \
    --k "$K_VAL" \
    --actor_lr "$ACTOR_LR" \
    --critic_lrs $CRITIC_LRS \
    --lambdas $LAMBDAS \
    --algos $ALGOS \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --gae_lambda "$GAE_LAMBDA" \
    --clip_eps "$CLIP_EPS" \
    --runway_length "$RUNWAY_LENGTH" \
    --out_dir "$OUT_DIR"

echo "Task completed successfully for $ENV_NAME"
