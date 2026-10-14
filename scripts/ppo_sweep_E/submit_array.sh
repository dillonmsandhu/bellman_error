#!/bin/bash
# scripts/ppo_sweep_E/submit_array.sh
#
# Dispatches the 10-environment SLURM array job for PPO Sampled E critic evaluation
# and chains a dependent job to automatically aggregate and plot results.

set -e
mkdir -p slurm

REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

ARRAY_SCRIPT="scripts/ppo_sweep_E/run_slurm_array.sh"
PLOT_SCRIPT="scripts/ppo_sweep_E/plot_results.py"
OUT_DIR=${OUT_DIR:-"results/ppo_sweep_E"}

echo "Submitting 10-environment array job (1 GPU per env) for PPO Sampled E Critic Benchmark..."
SBATCH_ARRAY_CMD="sbatch --parsable \
    --export=ALL,SLURM_SUBMIT_DIR=$REPO_ROOT,OUT_DIR=$OUT_DIR \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"

echo "Submitting chained plotting post-processing job..."
PLOT_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:$ARRAY_JOB_ID \
    --job-name="plot_ppo_sweep_E" \
    --output="slurm/%j_plot_ppo_sweep_E.out" \
    --error="slurm/%j_plot_ppo_sweep_E.err" \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=2 \
    --mem=8G \
    --wrap="$PYTHON $PLOT_SCRIPT --results_dir $OUT_DIR")

echo "Submitted Chained Plot Job ID: $PLOT_JOB_ID (will run when array finishes)"
echo "Track status with: squeue -u \$USER"
