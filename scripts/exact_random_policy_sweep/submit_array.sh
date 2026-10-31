#!/bin/bash
# scripts/exact_random_policy_sweep/submit_array.sh
# Submits the 9-environment SLURM array job and chains automated consolidated plotting.

set -e

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="exact_random_policy_sweep"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT"

NUM_UPDATES=${1:-100}
N_SEEDS=${2:-10}
SEED=${3:-42}
K_VALUES=${4:-"8 64"}
LRS=${5:-"0.0003 0.001 0.003"}
VAL_LAMBDA=${6:-0.8}
OUT_DIR=${7:-"results/exact_random_policy_sweep"}

export NUM_UPDATES N_SEEDS SEED K_VALUES LRS VAL_LAMBDA OUT_DIR REPO_ROOT

echo "================================================================"
echo "Submitting Exact Random Policy Evaluation SLURM Array"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "k Values:           $K_VALUES"
echo "Learning Rates:     $LRS"
echo "Value Lambda:       $VAL_LAMBDA"
echo "Output Directory:   $OUT_DIR"
echo "Log Directory:      $LOG_DIR"
echo "Repo Root:          $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/exact_random_policy_sweep/run_slurm_array.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/%A_%a.out\" \
    --error=\"${LOG_DIR}/%A_%a.err\" \
    --export=ALL,NUM_UPDATES=$NUM_UPDATES,N_SEEDS=$N_SEEDS,SEED=$SEED,K_VALUES=\"$K_VALUES\",LRS=\"$LRS\",VAL_LAMBDA=$VAL_LAMBDA,OUT_DIR=\"$OUT_DIR\",REPO_ROOT=\"$REPO_ROOT\" \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"

# Python interpreter resolution for chained plotting job
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

POST_SCRIPT="scripts/exact_random_policy_sweep/plot_results.py"
SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_exact_rand_pol \
    --output=\"${LOG_DIR}/%A_plot.out\" \
    --error=\"${LOG_DIR}/%A_plot.err\" \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=2 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --results_dir '$OUT_DIR' && $PYTHON scripts/exact_random_policy_sweep/analyze_results.py --results_dir '$OUT_DIR'\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Chained Post-Processing Job ID: $POST_JOB_ID (will run after $ARRAY_JOB_ID completes)"
echo "Logs written to:    $LOG_DIR/"
echo "Track status with: squeue -u \$USER"
