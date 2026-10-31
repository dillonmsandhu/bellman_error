#!/bin/bash
# scripts/sampled_E_lambda_test/submit_array.sh
# Submits the 9-environment SLURM array job and chains automated consolidated plotting.

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sampled_E_lambda_test"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT"

VALUE_LAMBDA=${1:-0.8}
NUM_UPDATES=${2:-100}
N_SEEDS=${3:-3}
NUM_TRAJECTORIES=${4:-128}
NUM_STEPS=${5:-128}
OUT_DIR=${6:-"results/sampled_E_lambda_test"}

export VALUE_LAMBDA NUM_UPDATES N_SEEDS NUM_TRAJECTORIES NUM_STEPS OUT_DIR

echo "================================================================"
echo "Submitting Sampled E(lambda) Test Sweep (9 Benchmark Environments)"
echo "Value Lambda:       $VALUE_LAMBDA"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Rollouts:           $NUM_TRAJECTORIES trajs x $NUM_STEPS steps"
echo "Output Directory:   $OUT_DIR"
echo "Log Directory:      $LOG_DIR"
echo "Repo Root:          $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/sampled_E_lambda_test/run_slurm_array.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/%A_%a.out\" \
    --error=\"${LOG_DIR}/%A_%a.err\" \
    --export=ALL,VALUE_LAMBDA=$VALUE_LAMBDA,NUM_UPDATES=$NUM_UPDATES,N_SEEDS=$N_SEEDS,NUM_TRAJECTORIES=$NUM_TRAJECTORIES,NUM_STEPS=$NUM_STEPS,OUT_DIR=$OUT_DIR,REPO_ROOT=$REPO_ROOT \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"

# Python interpreter resolution for post-processing job
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

POST_SCRIPT="scripts/sampled_E_lambda_test/plot_comparison.py"
RESULTS_SUBDIR="$OUT_DIR/lambda_$VALUE_LAMBDA"
SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_sampled_E_lambda \
    --output=\"${LOG_DIR}/%A_post.out\" \
    --error=\"${LOG_DIR}/%A_post.err\" \
    --time=00:30:00 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
    --cpus-per-task=2 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --results_dir $RESULTS_SUBDIR\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Post-Processing Job ID: $POST_JOB_ID (Runs after $ARRAY_JOB_ID)"
echo ""
echo "Monitor status with: squeue -u \$USER"
echo "Logs will be stored in:    $LOG_DIR/"
echo "Results will be stored in: $RESULTS_SUBDIR/"
