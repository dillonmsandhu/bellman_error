#!/bin/bash
# scripts/sweeps/submit_oracle_geometric_lambda.sh
# Submits the 7-environment SLURM array job for geometric jump sampled E(lambda) oracle targets validation.

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="oracle_geometric_lambda"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT"

VALUE_LAMBDA=${1:-0.9}
export VALUE_LAMBDA

echo "================================================================"
echo "Submitting Geometric Sampled E(lambda) Oracle Validation Sweep"
echo "Value Lambda: $VALUE_LAMBDA"
echo "Repo Root:    $REPO_ROOT"
echo "Log Dir:      $LOG_DIR"
echo "================================================================"

ARRAY_SCRIPT="scripts/sweeps/run_slurm_oracle_geometric_lambda.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/%A_%a.out\" \
    --error=\"${LOG_DIR}/%A_%a.err\" \
    --export=ALL,VALUE_LAMBDA=$VALUE_LAMBDA,REPO_ROOT=$REPO_ROOT \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"
echo ""
echo "Monitor status with: squeue -u \$USER"
echo "Logs written to: $LOG_DIR/"
echo "Results will be stored in: results/sweeps/oracle_targets_e_lambda_geometric/*_lambda_${VALUE_LAMBDA}/"
