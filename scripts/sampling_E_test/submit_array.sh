#!/bin/bash
# scripts/sampling_E_test/submit_array.sh
# Submits the 7-environment SLURM array job and chains automated consolidated plotting.

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
mkdir -p slurm
export PYTHONPATH="$REPO_ROOT"

NUM_UPDATES=${1:-50}
N_SEEDS=${2:-3}
NUM_TRAJECTORIES=${3:-128}
NUM_STEPS=${4:-128}
OUT_DIR=${5:-"results/sampling_E_test"}

export NUM_UPDATES N_SEEDS NUM_TRAJECTORIES NUM_STEPS OUT_DIR

echo "================================================================"
echo "Submitting Sampling E Test Sweep (7 Benchmark Environments)"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Rollouts:           $NUM_TRAJECTORIES trajs x $NUM_STEPS steps"
echo "Output Directory:   $OUT_DIR"
echo "Repo Root:          $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/sampling_E_test/run_slurm_array.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --export=ALL,NUM_UPDATES=$NUM_UPDATES,N_SEEDS=$N_SEEDS,NUM_TRAJECTORIES=$NUM_TRAJECTORIES,NUM_STEPS=$NUM_STEPS,OUT_DIR=$OUT_DIR,REPO_ROOT=$REPO_ROOT \
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

POST_SCRIPT="scripts/sampling_E_test/plot_comparison.py"
SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_sampling_E \
    --output=slurm/%A.out \
    --error=slurm/%A.err \
    --time=00:30:00 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
    --cpus-per-task=2 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --results_dir $OUT_DIR\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Post-Processing Job ID: $POST_JOB_ID (Runs after $ARRAY_JOB_ID)"
echo ""
echo "Monitor status with: squeue -u \$USER"
echo "Results will be stored in: $OUT_DIR/"
