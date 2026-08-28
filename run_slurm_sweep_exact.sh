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
N_SEEDS=3
TOTAL_TIMESTEPS=2000
ENVS=("FourRooms-misc" "MountainCar-v0")
POLICIES=("random" "fixed")
EXACT_ALGOS=("exact_td" "exact_mc" "exact_E_gd" "exact_td_lambda")

# Per-environment evaluation policy placeholders for fixed policy evaluation.
# Replace with your trained policy run directories for each environment (e.g. "ground_truth/20260821_164541" or "short_run").
# If left as PLACEHOLDER, the pipeline will auto-resolve to the latest available trained checkpoint for that environment.
declare -A FIXED_MODEL_DIRS=(
    ["FourRooms-misc"]="ground_truth/20260823_122419"
    ["MountainCar-v0"]="ground_truth/20260823_123519"
)

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
    # Get model directory for this specific environment
    MODEL_DIR="${FIXED_MODEL_DIRS[$env]}"

    for policy in "${POLICIES[@]}"; do
        echo ""
        echo "======================================================================"
        echo "Running Sweep: Policy=$policy | Environment=$env"
        if [ "$policy" = "fixed" ]; then
            echo "Evaluation Policy Model Dir: $MODEL_DIR"
        fi
        echo "======================================================================"

        CMD="$PYTHON sweep_pipeline.py \
            --policy $policy \
            --env-name $env \
            --algos ${EXACT_ALGOS[*]} \
            --n-seeds $N_SEEDS \
            --total-timesteps $TOTAL_TIMESTEPS \
            --model-dir '$MODEL_DIR' \
            --use-geom-mean \
            --rank-by 'final_window' \
            --window-size 400"

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
