#!/bin/bash
#SBATCH --job-name=sweep_sampled
#SBATCH --output=slurm/%j.out
#SBATCH --time=4:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Sample-Based Algorithms (TD, Sampled E, Monte Carlo)
# Note: Monte Carlo runs TD with lambda=1 across the same learning rate grid,
# enabling direct cross-algorithm comparison of TD, MC, and Sampled E with their
# respective optimal learning rates.
#
# Usage:
#   sbatch scripts/run_slurm_sweep_sampled.sh
#   ./scripts/run_slurm_sweep_sampled.sh (for local test)
# ==============================================================================

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Python environment (defaults to cluster gymnax environment)
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

# Configuration
N_SEEDS=5
ENVS=("FourRooms-misc" "MountainCar-v0")
POLICIES=("random" "fixed")
SAMPLED_ALGOS=("td" "td0" "sampled_E" "monte_carlo" "unbiased_sampled_E")

# Base config overrides for sampled algorithms (NUM_ENVS, NUM_STEPS, TOTAL_TIMESTEPS, MINIBATCH_SIZE, etc.)
CONFIG='{"NUM_ENVS": 64, "NUM_STEPS": 256, "TOTAL_TIMESTEPS": 1000000, "MINIBATCH_SIZE": 1024, "NUM_EPOCHS": 1}'

# Common Learning Rate Grid to sweep over for TD, TD(0), Sampled E, and Monte Carlo
LR_GRID="0.001 0.0005 0.0001 0.00005 0.00001 0.000005 0.000001"

# Lambda Grid to sweep over for TD (GAE_LAMBDA)
LAMBDA_GRID="0.5 0.9 0.95"

# Per-environment evaluation policy placeholders for fixed policy evaluation.
# Replace with your trained policy run directories for each environment (e.g. "ground_truth/20260821_164541" or "short_run").
# If left as PLACEHOLDER, the pipeline will auto-resolve to the latest available trained checkpoint for that environment.
declare -A FIXED_MODEL_DIRS=(
    ["FourRooms-misc"]="ground_truth/20260823_122419"
    ["MountainCar-v0"]="ground_truth/20260823_123519"
)

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM SAMPLED ALGORITHMS SWEEP (TD, TD(0), Sampled E, Monte Carlo)"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Policies: ${POLICIES[*]}"
echo "Algorithms: ${SAMPLED_ALGOS[*]}"
echo "LR Grid: $LR_GRID"
echo "Lambda Grid: $LAMBDA_GRID"
echo "Config Overrides: $CONFIG"
echo "Working Directory: $REPO_ROOT"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    # Get model directory for this specific environment
    MODEL_DIR="${FIXED_MODEL_DIRS[$env]}"

    for policy in "${POLICIES[@]}"; do
        echo ""
        echo "======================================================================"
        echo "Running Sampled Sweep: Policy=$policy | Environment=$env"
        if [ "$policy" = "fixed" ]; then
            echo "Evaluation Policy Model Dir: $MODEL_DIR"
        fi
        echo "======================================================================"

        CMD="$PYTHON scripts/sweep_pipeline.py \
            --policy $policy \
            --env-name $env \
            --algos ${SAMPLED_ALGOS[*]} \
            --lr-grid $LR_GRID \
            --lambda-grid $LAMBDA_GRID \
            --n-seeds $N_SEEDS \
            --model-dir '$MODEL_DIR' \
            --config '$CONFIG' \
            --rank-by 'auc' \
            --higher-is-better \
            --metric nn_greedy_performance \
            --use-geom-mean"
            
        echo "Command: $CMD"
        eval "$CMD"
    done
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "Sampled Algorithms Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
