#!/bin/bash
#SBATCH --job-name=smp_e_surg_rand
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
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

# Python resolution
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

# Environment index
TASK_ID="${1:-$SLURM_ARRAY_TASK_ID}"
if [ -z "$TASK_ID" ]; then
    echo "[Error] No TASK_ID or SLURM_ARRAY_TASK_ID specified."
    exit 1
fi

ALL_ENVS=(
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

ENV_NAME="${ALL_ENVS[$TASK_ID]}"
if [ -z "$ENV_NAME" ]; then
    echo "[Error] TASK_ID $TASK_ID out of range (0-8)."
    exit 1
fi

NUM_UPDATES="${NUM_UPDATES:-100}"
N_SEEDS="${N_SEEDS:-10}"
SEED="${SEED:-42}"
K_VALUES="${K_VALUES:-32}"
CRITIC_LRS="${CRITIC_LRS:-0.0005 0.001 0.003}"
NUM_ENVS="${NUM_ENVS:-16}"
NUM_STEPS="${NUM_STEPS:-64}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
VAL_LAMBDA="${VAL_LAMBDA:-1.0}"
OUT_DIR="${OUT_DIR:-results/sampled_E_surgical_random_sweep}"

echo "=================================================================="
echo "SLURM Array Job: $SLURM_ARRAY_JOB_ID, Task: $TASK_ID"
echo "Host: $(hostname) | Date: $(date)"
echo "Environment: $ENV_NAME (Index $TASK_ID)"
echo "Python: $PYTHON"
echo "Config: Updates=$NUM_UPDATES, Seeds=$N_SEEDS, k=($K_VALUES), LRs=($CRITIC_LRS)"
echo "=================================================================="

$PYTHON scripts/sampled_E_surgical_random_sweep/run_single_env.py \
    --env_name "$ENV_NAME" \
    --num_updates "$NUM_UPDATES" \
    --n_seeds "$N_SEEDS" \
    --seed "$SEED" \
    --k_values $K_VALUES \
    --critic_lrs $CRITIC_LRS \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --val_lambda "$VAL_LAMBDA" \
    --out_dir "$OUT_DIR"

echo "Task $TASK_ID ($ENV_NAME) completed successfully."
