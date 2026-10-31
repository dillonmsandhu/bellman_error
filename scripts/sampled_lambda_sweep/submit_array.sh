#!/bin/bash
# scripts/sampled_lambda_sweep/submit_array.sh
# Submits the 9-environment SLURM array job for Sampled PPO Lambda Sweep (TD(lambda) vs E(lambda) with v_aux).
# Chains automated consolidated plotting and analysis tables to run after array finishes.

set -e

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sampled_lambda_sweep"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT"

NUM_UPDATES=${1:-100}
N_SEEDS=${2:-10}
SEED=${3:-42}
K_VAL=${4:-32}
ACTOR_LR=${5:-0.0003}
CRITIC_LRS=${6:-"0.0005 0.001 0.003"}
LAMBDAS=${7:-"0.4 0.8 0.95 0.99"}
NUM_ENVS=${8:-32}
NUM_STEPS=${9:-256}
MINIBATCH_SIZE=${10:-256}
NUM_EPOCHS=${11:-4}
GAE_LAMBDA=${12:-0.8}
CLIP_EPS=${13:-0.2}
RUNWAY_LENGTH=${14:-adaptive}
OUT_DIR=${15:-"results/sampled_lambda_sweep"}
ALGOS=${16:-"td_lambda e_lambda e_lambda_fwd"}

export NUM_UPDATES N_SEEDS SEED K_VAL ACTOR_LR CRITIC_LRS LAMBDAS ALGOS NUM_ENVS NUM_STEPS MINIBATCH_SIZE NUM_EPOCHS GAE_LAMBDA CLIP_EPS RUNWAY_LENGTH OUT_DIR REPO_ROOT

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
echo "Submitting Sampled PPO Lambda Sweep SLURM Array"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Critic Feature k:   $K_VAL"
echo "Actor LR:           $ACTOR_LR (fixed)"
echo "Critic LRs:         $CRITIC_LRS"
echo "Lambdas:            $LAMBDAS"
echo "Algorithms:         $ALGOS"
echo "Actor GAE Lambda:   $GAE_LAMBDA"
echo "Clip Epsilon:       $CLIP_EPS"
echo "Rollout Config:     $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE | Epochs: $NUM_EPOCHS"
echo "Output Directory:   $OUT_DIR"
echo "Log Directory:      $LOG_DIR"
echo "Python:             $PYTHON"
echo "================================================================"

ARRAY_SCRIPT="scripts/sampled_lambda_sweep/run_slurm_array.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/%A_%a.out\" \
    --error=\"${LOG_DIR}/%A_%a.err\" \
    --export=ALL,NUM_UPDATES=$NUM_UPDATES,N_SEEDS=$N_SEEDS,SEED=$SEED,K_VAL=$K_VAL,ACTOR_LR=$ACTOR_LR,CRITIC_LRS=\"$CRITIC_LRS\",LAMBDAS=\"$LAMBDAS\",ALGOS=\"$ALGOS\",NUM_ENVS=$NUM_ENVS,NUM_STEPS=$NUM_STEPS,MINIBATCH_SIZE=$MINIBATCH_SIZE,NUM_EPOCHS=$NUM_EPOCHS,GAE_LAMBDA=$GAE_LAMBDA,CLIP_EPS=$CLIP_EPS,RUNWAY_LENGTH=$RUNWAY_LENGTH,OUT_DIR=\"$OUT_DIR\",REPO_ROOT=\"$REPO_ROOT\" \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"

POST_SCRIPT="scripts/sampled_lambda_sweep/plot_results.py"
SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_smp_lam \
    --output=\"${LOG_DIR}/%A_plot.out\" \
    --error=\"${LOG_DIR}/%A_plot.err\" \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=2 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --results_dir '$OUT_DIR' && $PYTHON scripts/sampled_lambda_sweep/analyze_results.py --results_dir '$OUT_DIR'\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Chained Post-Processing Job ID: $POST_JOB_ID (will run after $ARRAY_JOB_ID completes)"
echo "Logs written to:    $LOG_DIR/"
echo "Track status with: squeue -u \$USER"
