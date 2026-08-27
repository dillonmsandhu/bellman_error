#!/bin/bash
#SBATCH --job-name=sweep_exact
#SBATCH --output=slurm/%j.out
#SBATCH --time=4:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Exact Algorithms (TD, MC, E_gd, TD_lambda)
# Runs for both Fixed Policy and Random Policy on MountainCar and FourRooms.
#
# Usage:
#   sbatch run_slurm_sweep_exact.sh
#   ./run_slurm_sweep_exact.sh (for local test)
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
EXACT_ALGOS=("exact_td" "exact_mc" "exact_E_gd" "exact_td_lambda")

# PLACEHOLDER for pretrained PPO evaluation policy directory.
# Replace with your trained policy run directory (e.g. "ground_truth/20260821_164541" or "short_run"),
# or leave as placeholder to automatically resolve the latest available checkpoint.
FIXED_MODEL_DIR="PLACEHOLDER_PPO_MODEL_DIR"

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM EXACT ALGORITHMS SWEEP"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Policies: ${POLICIES[*]}"
echo "Algorithms: ${EXACT_ALGOS[*]}"
echo "Seeds: $N_SEEDS | Timesteps: $TOTAL_TIMESTEPS"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    for policy in "${POLICIES[@]}"; do
        echo ""
        echo "======================================================================"
        echo "Running Sweep: Policy=$policy | Environment=$env"
        echo "======================================================================"

        CMD="$PYTHON sweep_pipeline.py \
            --policy $policy \
            --env-name $env \
            --algos ${EXACT_ALGOS[*]} \
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
echo "Exact Algorithms Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
