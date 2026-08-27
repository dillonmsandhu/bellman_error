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
#   sbatch run_slurm_sweep_sampled.sh
#   ./run_slurm_sweep_sampled.sh (for local test)
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
TOTAL_TIMESTEPS=2000
ENVS=("FourRooms-misc" "MountainCar-v0")
POLICIES=("random" "fixed")
SAMPLED_ALGOS=("td" "sampled_E" "monte_carlo")

# Common Learning Rate Grid to sweep over for TD, Sampled E, and Monte Carlo
LR_GRID="0.05 0.01 0.005 0.001 0.0005 0.0001"

# PLACEHOLDER for pretrained PPO evaluation policy directory (used if policy=fixed)
FIXED_MODEL_DIR="PLACEHOLDER_PPO_MODEL_DIR"

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM SAMPLED ALGORITHMS SWEEP (TD, Sampled E, Monte Carlo)"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Policies: ${POLICIES[*]}"
echo "Algorithms: ${SAMPLED_ALGOS[*]}"
echo "LR Grid: $LR_GRID"
echo "Seeds: $N_SEEDS | Timesteps: $TOTAL_TIMESTEPS"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    for policy in "${POLICIES[@]}"; do
        echo ""
        echo "======================================================================"
        echo "Running Sampled Sweep: Policy=$policy | Environment=$env"
        echo "======================================================================"

        CMD="$PYTHON sweep_pipeline.py \
            --policy $policy \
            --env-name $env \
            --algos ${SAMPLED_ALGOS[*]} \
            --lr-grid $LR_GRID \
            --n-seeds $N_SEEDS \
            --total-timesteps $TOTAL_TIMESTEPS \
            --model-dir '$FIXED_MODEL_DIR'"

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
