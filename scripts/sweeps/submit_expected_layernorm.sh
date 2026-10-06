#!/bin/bash
# scripts/sweeps/submit_expected_layernorm.sh
# Submits the 7-environment SLURM array job for the Expected PPO LayerNorm and LR sweep.

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
mkdir -p slurm
export PYTHONPATH="$REPO_ROOT"

SWEEP_ID="ln_sweep_$(date +"%Y%m%d_%H%M%S")"
export SWEEP_ID

echo "================================================================"
echo "Submitting Expected PPO LayerNorm & LR Sweep Array"
echo "Sweep ID:  $SWEEP_ID"
echo "Repo Root: $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/sweeps/run_slurm_expected_layernorm.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --export=ALL,SWEEP_ID=$SWEEP_ID,REPO_ROOT=$REPO_ROOT \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"
echo ""
echo "Monitor status with: squeue -u \$USER"
echo "Outputs and plots will be written to: results/sweeps/expected_layernorm/<env_name>/"
