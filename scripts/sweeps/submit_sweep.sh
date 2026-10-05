#!/bin/bash
# scripts/sweeps/submit_sweep.sh
# Submits the 7-environment SLURM array job and chains consolidated plotting upon completion.

set -e
mkdir -p slurm

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"

# Python interpreter resolution
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

SWEEP_ID="sweep_$(date +"%Y%m%d_%H%M%S")"
export SWEEP_ID

echo "================================================================"
echo "Submitting Bellman Error Rollout Horizon Sweep"
echo "Sweep ID:  $SWEEP_ID"
echo "Repo Root: $REPO_ROOT"
echo "================================================================"

# 1. Submit the 7-environment array job
ARRAY_SCRIPT="scripts/sweeps/run_slurm_array.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --export=ALL,SWEEP_ID=$SWEEP_ID,SLURM_SUBMIT_DIR=$REPO_ROOT \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"

# 2. Submit Chained Post-Processing Plotting Job (runs after all array tasks finish)
PLOT_SCRIPT="scripts/sweeps/plot_consolidated.py"
PLOT_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:$ARRAY_JOB_ID \
    --job-name="plot_${SWEEP_ID}" \
    --output="slurm/%j_plot_${SWEEP_ID}.out" \
    --error="slurm/%j_plot_${SWEEP_ID}.err" \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --wrap="$PYTHON '$PLOT_SCRIPT' --sweep_id '$SWEEP_ID'")

echo "Submitted Chained Plotting Job ID: $PLOT_JOB_ID (depends on $ARRAY_JOB_ID)"
echo ""
echo "Monitor status with: squeue -u \$USER"
echo "Results will be stored in: results/sweeps/$SWEEP_ID/"
