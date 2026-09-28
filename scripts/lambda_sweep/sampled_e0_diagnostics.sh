#!/bin/bash
# ==============================================================================
# Sampled E(lambda=0) Diagnostics, Packaging & Email Dispatcher
#
# Generates:
#   1. Value learning curves & Greedy accuracy curves for Policy Eval tasks
#   2. Performance return curves for PPO tasks
#   3. Master summary PDF & PNG
#   4. Compact zip archive of all plots
#   5. Automatically emails the plots and archive to ds541@cs.duke.edu
# ==============================================================================

SWEEP_ID=$1
RECIPIENT=${2:-"ds541@cs.duke.edu"}

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: $0 <SWEEP_ID> [RECIPIENT]"
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

export PYTHONPATH="$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"

SWEEP_DIR="results/lambda_sweep/${SWEEP_ID}"
echo "======================================================================"
echo "Running Sampled E(lambda=0) Diagnostics for Sweep: $SWEEP_ID"
echo "Results Directory: $SWEEP_DIR"
echo "Recipient:         $RECIPIENT"
echo "======================================================================"

# 1. Run Python Diagnostics
$PYTHON scripts/lambda_sweep/sampled_e0_diagnostics.py \
    --sweep-id "$SWEEP_ID" \
    --base-dir "results/lambda_sweep"

# 2. Create Zip Archive of all plots
PLOTS_DIR="${SWEEP_DIR}/plots"
ZIP_FILE="${SWEEP_DIR}/sampled_e0_plots.zip"

if [ -d "$PLOTS_DIR" ]; then
    echo "Creating zip archive of plots: $ZIP_FILE"
    (cd "$SWEEP_DIR" && zip -q -r "sampled_e0_plots.zip" "plots" "sampled_e0_master_summary.pdf" "sampled_e0_master_summary.png" "sampled_e0_comparison_metrics.csv")
fi

# 3. Email Results
echo "Dispatching results via email to $RECIPIENT..."
$PYTHON - <<EOF
from core.mail import email_results_file
import os

recipient = "${RECIPIENT}"
sweep_dir = "${SWEEP_DIR}"

files_to_email = [
    os.path.join(sweep_dir, "sampled_e0_master_summary.pdf"),
    os.path.join(sweep_dir, "sampled_e0_master_summary.png"),
    os.path.join(sweep_dir, "sampled_e0_comparison_metrics.csv"),
    os.path.join(sweep_dir, "sampled_e0_plots.zip"),
]

for f in files_to_email:
    if os.path.exists(f):
        print(f"Sending: {f}")
        email_results_file(f, recipient=recipient)
    else:
        print(f"Skipping absent file: {f}")
EOF

echo "Sampled E(0) diagnostics and notification completed."
