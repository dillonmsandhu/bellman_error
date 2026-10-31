#!/bin/bash
# scripts/sampled_E_aux_random_sweep/submit_array.sh
# 1. Submits the 9-environment SLURM array job for sampled random policy evaluation.
# 2. Chains automated consolidated plotting and analysis tables to run after array finishes.

set -e

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sampled_E_aux_random_sweep"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT"

NUM_UPDATES=${1:-100}
N_SEEDS=${2:-10}
SEED=${3:-42}
K_VALUES=${4:-"8 64"}
CRITIC_LRS=${5:-"0.0005 0.001 0.003"}
NUM_ENVS=${6:-16}
NUM_STEPS=${7:-64}
MINIBATCH_SIZE=${8:-256}
NUM_EPOCHS=${9:-4}
VAL_LAMBDA=${10:-1.0}
OUT_DIR=${11:-"results/sampled_E_aux_random_sweep"}

export NUM_UPDATES N_SEEDS SEED K_VALUES CRITIC_LRS NUM_ENVS NUM_STEPS MINIBATCH_SIZE NUM_EPOCHS VAL_LAMBDA OUT_DIR REPO_ROOT

# Resolve Python interpreter
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

echo "================================================================"
echo "Submitting Sampled E-Aux Random Policy Evaluation SLURM Array"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "k Values:           $K_VALUES"
echo "Critic LRs:         $CRITIC_LRS"
echo "Rollout Config:     $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE | Epochs: $NUM_EPOCHS"
echo "Value Lambda:       $VAL_LAMBDA"
echo "Output Directory:   $OUT_DIR"
echo "Log Directory:      $LOG_DIR"
echo "Repo Root:          $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/sampled_E_aux_random_sweep/run_slurm_array.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/%A_%a.out\" \
    --error=\"${LOG_DIR}/%A_%a.err\" \
    --export=ALL,NUM_UPDATES=$NUM_UPDATES,N_SEEDS=$N_SEEDS,SEED=$SEED,K_VALUES=\"$K_VALUES\",CRITIC_LRS=\"$CRITIC_LRS\",NUM_ENVS=$NUM_ENVS,NUM_STEPS=$NUM_STEPS,MINIBATCH_SIZE=$MINIBATCH_SIZE,NUM_EPOCHS=$NUM_EPOCHS,VAL_LAMBDA=$VAL_LAMBDA,OUT_DIR=\"$OUT_DIR\",REPO_ROOT=\"$REPO_ROOT\" \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"

POST_SCRIPT="scripts/sampled_E_aux_random_sweep/plot_results.py"
SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_smp_e_aux \
    --output=\"${LOG_DIR}/%A_plot.out\" \
    --error=\"${LOG_DIR}/%A_plot.err\" \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=2 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --results_dir '$OUT_DIR' && $PYTHON scripts/sampled_E_aux_random_sweep/analyze_results.py --results_dir '$OUT_DIR'\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Chained Post-Processing Job ID: $POST_JOB_ID (will run after $ARRAY_JOB_ID completes)"
echo "Logs written to:    $LOG_DIR/"
echo "Track status with: squeue -u \$USER"
