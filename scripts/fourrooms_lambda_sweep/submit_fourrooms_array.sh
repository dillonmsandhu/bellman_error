#!/bin/bash
# ==============================================================================
# Dispatcher for FourRooms & EightRooms Lambda Benchmark (SLURM Array + Chained Plotter)
#
# Runs Ground Truth PPO and Exact E-Lambda PPO (λ in {0, 0.8, 0.95, 1}) on:
#   1. Regular 4 rooms (FourRooms-misc)
#   2. Continuous 4 rooms (continuous-fourrooms)
#   3. Continuous dense 4 rooms (continuous-fourrooms-dense)
#   4. Discrete 8 rooms dense (EightRooms-Dense)
#
# Exact settings enforced:
#   NUM_TIMESTEPS=1000, NUM_EPOCHS=1, NUM_MINIBATCHES=1, 8 seeds
#
# Usage:
#   ./scripts/fourrooms_lambda_sweep/submit_fourrooms_array.sh [options]
#
# Options:
#   --seeds N         Number of seeds to run per condition (default: 8)
#   --timesteps N     Total PPO update iterations (default: 1000)
#   --sweep-id ID     Unique identifier for sweep directory (default: timestamp)
#   --dry-run         Print commands without submitting to SLURM
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEEDS=8
TIMESTEPS=1000
SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --sweep-id)
            SWEEP_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--seeds 8] [--timesteps 1000] [--sweep-id ID] [--dry-run]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p slurm

# Python executable
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

echo "======================================================================"
echo "BENCHMARK SWEEP SUBMISSION"
echo "Sweep ID:          $SWEEP_ID"
echo "Seeds:             $SEEDS"
echo "Exact Settings:    NUM_TIMESTEPS=$TIMESTEPS, NUM_EPOCHS=1, NUM_MINIBATCHES=1"
echo "Environments (4):  FourRooms-misc, continuous-fourrooms, continuous-fourrooms-dense, EightRooms-Dense"
echo "Algorithms (5):    Ground Truth PPO, Exact E(λ=0.0, 0.8, 0.95, 1.0) PPO"
echo "Mode:              $([ "$DRY_RUN" = true ] && echo 'DRY-RUN' || echo 'SLURM BATCH ARRAY')"
echo "======================================================================"

ARRAY_SCRIPT="$SCRIPT_DIR/run_slurm_fourrooms_array.sh"
PLOT_SCRIPT="$SCRIPT_DIR/plot_fourrooms_learning_curves.py"

SBATCH_ARRAY_CMD="sbatch --parsable \
    --export=ALL,N_SEEDS=$SEEDS,TOTAL_TIMESTEPS=$TIMESTEPS,NUM_TIMESTEPS=$TIMESTEPS,SWEEP_ID=$SWEEP_ID,SLURM_SUBMIT_DIR=$REPO_ROOT \
    \"$ARRAY_SCRIPT\""

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "--> [DRY-RUN] Array submission command (tasks 0-3):"
    echo "$SBATCH_ARRAY_CMD"
    echo ""
    echo "--> [DRY-RUN] Chained vector PDF generation command:"
    echo "sbatch --dependency=afterany:<ARRAY_JOB_ID> \
    --job-name=\"plot_${SWEEP_ID}\" \
    --output=\"slurm/%j_plot.out\" \
    --time=00:30:00 \
    --partition=compsci \
    --wrap=\"$PYTHON $PLOT_SCRIPT --sweep-id $SWEEP_ID\""
    exit 0
fi

# Submit SLURM array job
echo "Submitting SLURM Array Job (tasks 0-3)..."
ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] || [ -z "$ARRAY_JOB_ID" ]; then
    echo "Error: Failed to submit SLURM array job (exit code: $EXIT_CODE)"
    exit 1
fi

echo "--> SLURM Array Job Submitted: $ARRAY_JOB_ID"
echo "    Monitor jobs: squeue -j $ARRAY_JOB_ID"

# Submit dependent plotting job
PLOT_JOB_CMD="sbatch \
    --dependency=afterany:$ARRAY_JOB_ID \
    --job-name=\"plot_${SWEEP_ID}\" \
    --output=\"slurm/%j_plot_${SWEEP_ID}.out\" \
    --time=00:30:00 \
    --partition=compsci \
    --wrap=\"$PYTHON '$PLOT_SCRIPT' --sweep-id '$SWEEP_ID'\""

echo ""
echo "Submitting chained plotting job (runs after array completes)..."
PLOT_JOB_OUTPUT=$(eval "$PLOT_JOB_CMD")
PLOT_JOB_ID=$(echo "$PLOT_JOB_OUTPUT" | awk '{print $NF}')
echo "--> Plotting Job Submitted: $PLOT_JOB_ID"
echo ""
echo "When all jobs finish, the vector PDF learning curves and summary CSV will be saved to:"
echo "    results/fourrooms_sweep/${SWEEP_ID}/plots/"
echo "======================================================================"
