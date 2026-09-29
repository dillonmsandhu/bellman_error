#!/bin/bash
# ==============================================================================
# FourRooms & EightRooms Lambda Benchmark Array Worker
#
# Runs Ground Truth PPO and Exact E-Lambda PPO (lambda = 0.0, 0.8, 0.95, 1.0)
# for one specific environment (sub-job).
#
# Environments:
#   0: FourRooms-misc (regular 4 rooms, discrete)
#   1: continuous-fourrooms (continuous 4 rooms, sparse)
#   2: continuous-fourrooms-dense (continuous 4 rooms, dense PBRS)
#   3: EightRooms-Dense (discrete 8 rooms, dense PBRS)
#
# Exact settings enforced:
#   NUM_TIMESTEPS=1000, TOTAL_TIMESTEPS=1000, NUM_EPOCHS=1, NUM_MINIBATCHES=1, MINIBATCH_SIZE=1, N_SEEDS=8
#
# Arguments:
#   $1 - Environment selector (0, 1, 2, 3 or exact environment name)
#        Defaults to SLURM_ARRAY_TASK_ID if not provided
#   $2 - Sweep ID (optional, defaults to timestamp or $SWEEP_ID)
# ==============================================================================

set -e

ENV_ARG=$1
SWEEP_ID_ARG=$2

# Supported Environments
ENVS=(
    "FourRooms-misc"
    "continuous-fourrooms"
    "continuous-fourrooms-dense"
    "EightRooms-Dense"
)

if [ -n "$ENV_ARG" ]; then
    if [[ "$ENV_ARG" =~ ^[0-3]$ ]]; then
        ENV_NAME="${ENVS[$ENV_ARG]}"
    else
        ENV_NAME="$ENV_ARG"
    fi
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    echo "Error: No environment specified and SLURM_ARRAY_TASK_ID is unset."
    echo "Usage: $0 <0|1|2|3|ENV_NAME> [SWEEP_ID]"
    exit 1
fi

# Resolve Sweep ID
if [ -n "$SWEEP_ID_ARG" ]; then
    SWEEP_ID="$SWEEP_ID_ARG"
elif [ -z "$SWEEP_ID" ]; then
    SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
fi

# Exact Settings
N_SEEDS=${N_SEEDS:-8}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-1000}
NUM_EPOCHS=1
NUM_MINIBATCHES=1
MINIBATCH_SIZE=1

LAMBDA_VALUES=(0.0 0.8 0.95 1.0)

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"
PLOT_SCRIPT="$REPO_ROOT/scripts/fourrooms_lambda_sweep/plot_fourrooms_learning_curves.py"

# Resolve Python executable
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "======================================================================"
echo "ARRAY WORKER: SUB-JOB START"
echo "Host:             $(hostname)"
echo "Start Time:       $(date +"%Y-%m-%d %H:%M:%S")"
echo "Environment:      $ENV_NAME"
echo "Sub-job Index:    ${SLURM_ARRAY_TASK_ID:-'Manual'}"
echo "Sweep ID:         $SWEEP_ID"
echo "Seeds:            $N_SEEDS"
echo "Exact Settings:   NUM_TIMESTEPS=$NUM_TIMESTEPS, NUM_EPOCHS=$NUM_EPOCHS, NUM_MINIBATCHES=$NUM_MINIBATCHES"
echo "Lambdas to run:   ${LAMBDA_VALUES[*]}"
echo "Python:           $PYTHON"
echo "======================================================================"

SWEEP_BASE_SUFFIX="fourrooms_sweep/${SWEEP_ID}"

# ------------------------------------------------------------------------------
# 1. Run Ground Truth PPO (Baseline)
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[1/5] Running Ground Truth PPO on $ENV_NAME ($N_SEEDS seeds)..."
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
# 2. Run Exact E-Lambda PPO for each lambda in {0.0, 0.8, 0.95, 1.0}
# ------------------------------------------------------------------------------
RUN_NUM=2
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "[$RUN_NUM/5] Running Exact E-Lambda PPO (λ = $LAMBDA) on $ENV_NAME ($N_SEEDS seeds)..."
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
echo "All 5 runs completed for environment: $ENV_NAME"
echo "Finished at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 3. Opportunistic Post-Run Check: Trigger Plot if All 4 Environments Are Ready
# ------------------------------------------------------------------------------
ALL_COMPLETED=true
for env in "${ENVS[@]}"; do
    # Check Ground Truth
    GT_PKL="results/ppo/ground_truth/${SWEEP_BASE_SUFFIX}/ground_truth/${env}/out.pkl"
    if [ ! -f "$GT_PKL" ]; then
        ALL_COMPLETED=false
        break
    fi
    # Check all 4 lambdas
    for LAMBDA in "${LAMBDA_VALUES[@]}"; do
        LAMBDA_PKL="results/ppo/exact_E_lambda/${SWEEP_BASE_SUFFIX}/lambda_${LAMBDA}/${env}/out.pkl"
        if [ ! -f "$LAMBDA_PKL" ]; then
            ALL_COMPLETED=false
            break 2
        fi
    done
done

if [ "$ALL_COMPLETED" = true ]; then
    echo ""
    echo "======================================================================"
    echo "All environments have finished all 5 runs for Sweep $SWEEP_ID!"
    echo "Generating vector PDF learning curves plot..."
    echo "======================================================================"
    $PYTHON "$PLOT_SCRIPT" --sweep-id "$SWEEP_ID"
fi
