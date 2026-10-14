#!/bin/bash
# scripts/exact_ppo_sweep/submit_array.sh
# Submits the 9-environment SLURM array job and chains automated consolidated plotting and analysis tables.

set -e

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
mkdir -p slurm
export PYTHONPATH="$REPO_ROOT"

NUM_UPDATES=${1:-100}
N_SEEDS=${2:-10}
SEED=${3:-42}
K_VAL=${4:-32}
ACTOR_LR=${5:-0.0001}
CRITIC_LR=${6:-0.001}
GAE_LAMBDA=${7:-0.2}
CLIP_EPS=${8:-0.2}
VAL_LAMBDA=${9:-0.8}
OUT_DIR=${10:-"results/exact_ppo_sweep"}

export NUM_UPDATES N_SEEDS SEED K_VAL ACTOR_LR CRITIC_LR GAE_LAMBDA CLIP_EPS VAL_LAMBDA OUT_DIR REPO_ROOT

echo "================================================================"
echo "Submitting Exact PPO Evaluation SLURM Array"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Critic k:           $K_VAL"
echo "Actor LR:           $ACTOR_LR"
echo "Critic LR:          $CRITIC_LR"
echo "GAE Lambda:         $GAE_LAMBDA"
echo "Clip Epsilon:       $CLIP_EPS"
echo "Value Lambda:       $VAL_LAMBDA"
echo "Output Directory:   $OUT_DIR"
echo "Repo Root:          $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/exact_ppo_sweep/run_slurm_array.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --export=ALL,NUM_UPDATES=$NUM_UPDATES,N_SEEDS=$N_SEEDS,SEED=$SEED,K_VAL=$K_VAL,ACTOR_LR=$ACTOR_LR,CRITIC_LR=$CRITIC_LR,GAE_LAMBDA=$GAE_LAMBDA,CLIP_EPS=$CLIP_EPS,VAL_LAMBDA=$VAL_LAMBDA,OUT_DIR=\"$OUT_DIR\",REPO_ROOT=\"$REPO_ROOT\" \
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

POST_SCRIPT="scripts/exact_ppo_sweep/plot_results.py"
SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_exact_ppo \
    --output=slurm/%A_plot.out \
    --error=slurm/%A_plot.err \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=2 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --results_dir '$OUT_DIR' && $PYTHON scripts/exact_ppo_sweep/analyze_results.py --results_dir '$OUT_DIR'\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Chained Post-Processing Job ID: $POST_JOB_ID (will run after $ARRAY_JOB_ID completes)"
echo "Track status with: squeue -u \$USER"
