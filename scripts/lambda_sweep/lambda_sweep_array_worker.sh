#!/bin/bash
# ==============================================================================
# SLURM Array Worker Script for Lambda Sweeps
#
# Reads the task corresponding to SLURM_ARRAY_TASK_ID from the manifest file.
# Arguments:
#   $1 - Manifest file path (containing "ENV POLICY ALGO [LAMBDAS...]" on each line)
#   $2 - Sweep ID
#   "${@:3}" - Extra arguments passed to lambda_sweep_pipeline (e.g. --config)
# ==============================================================================

MANIFEST_FILE=$1
SWEEP_ID=$2
shift 2

if [ -z "$MANIFEST_FILE" ] || [ -z "$SWEEP_ID" ]; then
    echo "Usage: $0 <MANIFEST_FILE> <SWEEP_ID> [pipeline_args...]"
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    exit 1
fi

# 1-indexed line number for sed
TASK_IDX=$((SLURM_ARRAY_TASK_ID + 1))
TASK_LINE=$(sed -n "${TASK_IDX}p" "$MANIFEST_FILE")

if [ -z "$TASK_LINE" ]; then
    echo "Error: No task found at line $TASK_IDX in manifest $MANIFEST_FILE"
    exit 1
fi

read -r ENV_NAME POLICY_TYPE ALGO ALGO_LAMBDAS <<< "$TASK_LINE"

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR/../.."
fi
REPO_ROOT=$(pwd)

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

echo "======================================================================"
echo "Starting SLURM Lambda Sweep Array Task: $SLURM_ARRAY_TASK_ID (Line $TASK_IDX)"
echo "Environment: $ENV_NAME"
echo "Policy:      $POLICY_TYPE"
echo "Algorithm:   $ALGO"
if [ -n "$ALGO_LAMBDAS" ]; then
    echo "Lambdas:     $ALGO_LAMBDAS"
fi
echo "Sweep ID:    $SWEEP_ID"
echo "Job ID:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Host:        $(hostname)"
echo "Start Time:  $(date +"%Y-%m-%d %H:%M:%S")"
echo "======================================================================"

export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Assemble Python command
CMD=($PYTHON -m scripts.lambda_sweep.lambda_sweep_pipeline
    --env-name "$ENV_NAME"
    --policy "$POLICY_TYPE"
    --algo "$ALGO"
    --sweep-id "$SWEEP_ID"
)

if [ -n "$ALGO_LAMBDAS" ]; then
    CMD+=(--lambdas $ALGO_LAMBDAS)
fi

CMD+=("$@")

echo "Running command: ${CMD[*]}"
"${CMD[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Sweep pipeline completed successfully for $ENV_NAME / $POLICY_TYPE / $ALGO"
else
    echo "Sweep pipeline FAILED for $ENV_NAME / $POLICY_TYPE / $ALGO with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
