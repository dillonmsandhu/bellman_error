#!/bin/bash
# scripts/jump_mode_sweep/submit_array.sh
#
# Submits the 9-environment SLURM array comparing:
# 1. e_lambda_drop (Drop Invalid Jump)
# 2. e_lambda_mc   (MC Invalid Jump)
# 3. bsmc          (1-step BSMC with matched weight for lambda=0.9)
#
# Automatically creates the slurm/ log directory and chains analyze_sweep.py as a dependent job.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p slurm

export VALUE_LAMBDA=${VALUE_LAMBDA:-0.9}
export TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-100000}
export N_SEEDS=${N_SEEDS:-5}
export OUT_DIR=${OUT_DIR:-"results/jump_mode_sweep"}

echo "================================================================"
echo "SUBMITTING JUMP MODES VS BSMC SWEEP (PPO)"
echo "Repo Root:       $REPO_ROOT"
echo "Value Lambda:    $VALUE_LAMBDA"
echo "Seeds:           $N_SEEDS"
echo "Total Timesteps: $TOTAL_TIMESTEPS"
echo "Output Root:     $OUT_DIR"
echo "Partition:       compsci-gpu (A5000:1)"
echo "Environments:    9 environments (Array 0-8)"
echo "Algorithms:      e_lambda_drop, e_lambda_mc, bsmc"
echo "================================================================"

ARRAY_JOB_OUTPUT=$(sbatch scripts/jump_mode_sweep/run_slurm_array.sh)
echo "$ARRAY_JOB_OUTPUT"

ARRAY_JOB_ID=$(echo "$ARRAY_JOB_OUTPUT" | grep -oE '[0-9]+')

if [ -n "$ARRAY_JOB_ID" ]; then
    echo "Submitted array job ID: $ARRAY_JOB_ID"
    echo "Chaining analysis post-processing job (afterany:$ARRAY_JOB_ID)..."

    ANALYSIS_CMD="python scripts/jump_mode_sweep/analyze_sweep.py --results_dir $OUT_DIR --value_lambda $VALUE_LAMBDA"

    ANALYSIS_JOB_OUTPUT=$(sbatch \
        --job-name=jump_mode_analysis \
        --partition=compsci-gpu \
        --time=00:30:00 \
        --mem=16G \
        --output=slurm/analysis_%j.out \
        --error=slurm/analysis_%j.err \
        --dependency=afterany:"$ARRAY_JOB_ID" \
        --wrap="$ANALYSIS_CMD")

    echo "$ANALYSIS_JOB_OUTPUT"
    echo "Analysis will run automatically when all tasks complete."
else
    echo "Warning: Could not parse array job ID; analysis job not queued."
fi

echo "================================================================"
echo "Monitor status with: squeue -u \$USER"
echo "================================================================"
