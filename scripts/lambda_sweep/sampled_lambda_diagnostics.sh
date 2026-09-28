#!/bin/bash
# ==============================================================================
# SLURM Diagnostics Script for Sampled E(lambda) Sweeps
#
# Arguments:
#   $1 - Sweep ID
# ==============================================================================

SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: $0 <SWEEP_ID>"
    exit 1
fi

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/../.."
fi
REPO_ROOT=$(pwd)

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
else
    PYTHON="python"
fi

echo "======================================================================"
echo "Starting Sampled Lambda Sweep Diagnostics for Sweep ID: $SWEEP_ID"
echo "======================================================================"

export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

$PYTHON -m scripts.lambda_sweep.sampled_lambda_diagnostics --sweep-id "$SWEEP_ID"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Diagnostics completed successfully. Now compressing results..."
    cd "$REPO_ROOT/results/lambda_sweep"
    tar -czvf "sampled_lambda_sweep_${SWEEP_ID}.tar.gz" "$SWEEP_ID"
    echo "Compression complete: results/lambda_sweep/sampled_lambda_sweep_${SWEEP_ID}.tar.gz"

    cd "$REPO_ROOT"
    echo "Sending results and comparison plots to ds541@cs.duke.edu..."
    # 1. Email the master 12-task vector PDF and PNG summary
    $PYTHON -c "from core.mail import email_results_file; import os; [email_results_file(f) for f in ['results/lambda_sweep/${SWEEP_ID}/sampled_lambda_comparison_summary.pdf', 'results/lambda_sweep/${SWEEP_ID}/sampled_lambda_comparison_summary.png', 'results/lambda_sweep/${SWEEP_ID}/sampled_lambda_comparison_metrics.csv'] if os.path.exists(f)]"

    # 2. Email the compressed full archive
    $PYTHON -c "from core.mail import email_results_file; email_results_file('results/lambda_sweep/sampled_lambda_sweep_${SWEEP_ID}.tar.gz')"
else
    echo "Diagnostics FAILED with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
