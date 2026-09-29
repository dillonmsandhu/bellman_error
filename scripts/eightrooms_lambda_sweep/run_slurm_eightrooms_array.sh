#!/bin/bash
#SBATCH --job-name=eightrooms_lambda_array
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

# ==============================================================================
# Self-Contained SLURM Job Array: EightRooms Benchmark
# Full 2x2 Factorial Comparison: [Discrete vs Continuous] x [Sparse vs Dense]
#
# Environments (1 sub-job per environment, array tasks 0-3):
#   - Task 0: EightRooms-misc (discrete 8 rooms, sparse)
#   - Task 1: continuous-eightrooms (continuous 8 rooms, sparse)
#   - Task 2: EightRooms-Dense (discrete 8 rooms, dense PBRS)
#   - Task 3: continuous-eightrooms-dense (continuous 8 rooms, dense PBRS)
#
# In each sub-job, runs 6 distinct runs with 8 seeds using exact settings:
#   NUM_TIMESTEPS=1000, TOTAL_TIMESTEPS=1000, NUM_EPOCHS=1, NUM_MINIBATCHES=1, MINIBATCH_SIZE=1, N_SEEDS=8
#
# Algorithms / Runs:
#   1. Ground Truth PPO
#   2. Exact E-Lambda PPO (λ = 0.0)
#   3. Exact E-Lambda PPO (λ = 0.8)
#   4. Exact E-Lambda PPO (λ = 0.95)
#   5. Exact E-Lambda PPO (λ = 0.99)
#   6. Exact E-Lambda PPO (λ = 1.0)
#
# Usage:
#   sbatch scripts/eightrooms_lambda_sweep/run_slurm_eightrooms_array.sh
#   Or run locally:
#   ./scripts/eightrooms_lambda_sweep/run_slurm_eightrooms_array.sh 0   # runs env 0
# ==============================================================================

set -e

mkdir -p slurm

# Working directory resolution: SLURM defaults to directory where sbatch was called
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Resolve Python executable
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

# Supported Environments (2x2 Factorial Grid)
ENVS=(
    "EightRooms-misc"
    "continuous-eightrooms"
    "EightRooms-Dense"
    "continuous-eightrooms-dense"
)

# Determine environment from argument $1 or SLURM_ARRAY_TASK_ID
TARGET_ARG=${1:-$SLURM_ARRAY_TASK_ID}
if [ -z "$TARGET_ARG" ]; then
    echo "No task index provided; running all 4 environments sequentially..."
    for task_idx in 0 1 2 3; do
        "$0" "$task_idx"
    done
    exit 0
fi

if [[ "$TARGET_ARG" =~ ^[0-3]$ ]]; then
    ENV_NAME="${ENVS[$TARGET_ARG]}"
else
    ENV_NAME="$TARGET_ARG"
fi

# Shared Sweep ID across array tasks
if [ -z "$SWEEP_ID" ]; then
    if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
        SWEEP_ID="eightrooms_${SLURM_ARRAY_JOB_ID}"
    else
        SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
    fi
fi

# Exact Settings
N_SEEDS=${N_SEEDS:-8}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-1000}
NUM_EPOCHS=1
NUM_MINIBATCHES=1
MINIBATCH_SIZE=1

LAMBDA_VALUES=(0.0 0.8 0.95 0.99 1.0)
SWEEP_SUITE_DIR="results/eightrooms_sweep/${SWEEP_ID}"
SWEEP_BASE_SUFFIX="eightrooms_sweep/${SWEEP_ID}"
ENV_DIR="${SWEEP_SUITE_DIR}/${ENV_NAME}"
mkdir -p "$ENV_DIR"

echo "======================================================================"
echo "EIGHTROOMS BENCHMARK SUB-JOB: $ENV_NAME"
echo "Host:             $(hostname)"
echo "Start Time:       $(date +"%Y-%m-%d %H:%M:%S")"
echo "Environment:      $ENV_NAME"
echo "Sub-job Index:    ${SLURM_ARRAY_TASK_ID:-$TARGET_ARG}"
echo "Sweep ID:         $SWEEP_ID"
echo "Seeds:            $N_SEEDS"
echo "Exact Settings:   NUM_TIMESTEPS=$NUM_TIMESTEPS, NUM_EPOCHS=$NUM_EPOCHS, NUM_MINIBATCHES=$NUM_MINIBATCHES"
echo "Lambdas to run:   ${LAMBDA_VALUES[*]}"
echo "Python:           $PYTHON"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Run Ground Truth PPO (Baseline)
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[1/6] Running Ground Truth PPO on $ENV_NAME ($N_SEEDS seeds)..."
echo "======================================================================"
GT_CONFIG="{\"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_TIMESTEPS\": $NUM_TIMESTEPS, \"NUM_EPOCHS\": $NUM_EPOCHS, \"NUM_MINIBATCHES\": $NUM_MINIBATCHES, \"MINIBATCH_SIZE\": $MINIBATCH_SIZE}"

$PYTHON -m ppo.ground_truth \
    --env-ids "$ENV_NAME" \
    --n-seeds "$N_SEEDS" \
    --config "$GT_CONFIG" \
    --save-metrics \
    --run-suffix "${SWEEP_BASE_SUFFIX}/ground_truth"

echo "[OK] Completed Ground Truth PPO for $ENV_NAME"

# ------------------------------------------------------------------------------
# 2. Run Exact E-Lambda PPO for each lambda in {0.0, 0.8, 0.95, 0.99, 1.0}
# ------------------------------------------------------------------------------
RUN_NUM=2
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "[$RUN_NUM/6] Running Exact E-Lambda PPO (λ = $LAMBDA) on $ENV_NAME ($N_SEEDS seeds)..."
    echo "======================================================================"
    EL_CONFIG="{\"TOTAL_TIMESTEPS\": $TOTAL_TIMESTEPS, \"NUM_TIMESTEPS\": $NUM_TIMESTEPS, \"NUM_EPOCHS\": $NUM_EPOCHS, \"NUM_MINIBATCHES\": $NUM_MINIBATCHES, \"MINIBATCH_SIZE\": $MINIBATCH_SIZE, \"VALUE_LAMBDA\": $LAMBDA}"

    $PYTHON -m ppo.exact_E_lambda \
        --env-ids "$ENV_NAME" \
        --n-seeds "$N_SEEDS" \
        --config "$EL_CONFIG" \
        --save-metrics \
        --run-suffix "${SWEEP_BASE_SUFFIX}/lambda_${LAMBDA}"

    echo "[OK] Completed Exact E(λ = $LAMBDA) for $ENV_NAME"
    RUN_NUM=$((RUN_NUM + 1))
done

echo ""
echo "======================================================================"
echo "All 6 runs completed for environment: $ENV_NAME"
echo "Finished at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "======================================================================"

# Mark this environment as 100% finished
touch "${ENV_DIR}/.env_complete"

# ------------------------------------------------------------------------------
# 3. Check Suite Completion: Compile Vector PDF Once All 4 Tasks Finish
# ------------------------------------------------------------------------------
COMPLETED_COUNT=0
for E in "${ENVS[@]}"; do
    if [ -f "${SWEEP_SUITE_DIR}/${E}/.env_complete" ]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    fi
done

echo ""
echo "EightRooms Benchmark Progress: $COMPLETED_COUNT / ${#ENVS[@]} environments finished."

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Running in a SLURM array job
    if [ "$COMPLETED_COUNT" -eq "${#ENVS[@]}" ]; then
        if mkdir "${SWEEP_SUITE_DIR}/.suite_plot_lock" 2>/dev/null; then
            echo "======================================================================"
            echo "ALL ${#ENVS[@]} EIGHTROOMS ENVIRONMENTS COMPLETE! Compiling master PDF..."
            echo "======================================================================"
            $PYTHON scripts/eightrooms_lambda_sweep/plot_eightrooms_learning_curves.py --sweep-id "$SWEEP_ID"
        else
            echo "Master vector PDF already generated by another finished task."
        fi
    else
        # Partial completion: update progress plot
        $PYTHON scripts/eightrooms_lambda_sweep/plot_eightrooms_learning_curves.py --sweep-id "$SWEEP_ID"
    fi
else
    # Standalone CLI execution
    $PYTHON scripts/eightrooms_lambda_sweep/plot_eightrooms_learning_curves.py --sweep-id "$SWEEP_ID"
fi
