#!/bin/bash
# scripts/sampled_E_surgical_ppo_sweep/submit_array.sh
# 1. Submits the 9-environment SLURM array job for full sampled PPO.
# 2. Chains automated consolidated plotting and analysis tables to run after array finishes.

set -e

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
mkdir -p slurm
export PYTHONPATH="$REPO_ROOT"

NUM_UPDATES=${1:-100}
N_SEEDS=${2:-10}
SEED=${3:-42}
K=${4:-32}
ACTOR_LR=${5:-0.0003}
CRITIC_LRS=${6:-"0.0005 0.001 0.003"}
NUM_ENVS=${7:-32}
NUM_STEPS=${8:-128}
MINIBATCH_SIZE=${9:-256}
NUM_EPOCHS=${10:-4}
GAE_LAMBDA=${11:-0.8}
CLIP_EPS=${12:-0.2}
GAMMA=${13:-0.99}
OUT_DIR=${14:-"results/sampled_E_surgical_ppo_sweep"}

export NUM_UPDATES N_SEEDS SEED K ACTOR_LR CRITIC_LRS NUM_ENVS NUM_STEPS MINIBATCH_SIZE NUM_EPOCHS GAE_LAMBDA CLIP_EPS GAMMA OUT_DIR REPO_ROOT

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
echo "Submitting Sampled E-Surgical PPO Evaluation SLURM Array"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Critic Dimension k: $K"
echo "Actor LR:           $ACTOR_LR"
echo "Critic LRs:         $CRITIC_LRS"
echo "Rollout Config:     $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE | Epochs: $NUM_EPOCHS"
echo "GAE Lambda:         $GAE_LAMBDA | Clip Epsilon: $CLIP_EPS"
echo "Output Directory:   $OUT_DIR"
echo "Repo Root:          $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/sampled_E_surgical_ppo_sweep/run_slurm_array.sh"
SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --export=ALL,NUM_UPDATES=$NUM_UPDATES,N_SEEDS=$N_SEEDS,SEED=$SEED,K=$K,ACTOR_LR=$ACTOR_LR,CRITIC_LRS=\"$CRITIC_LRS\",NUM_ENVS=$NUM_ENVS,NUM_STEPS=$NUM_STEPS,MINIBATCH_SIZE=$MINIBATCH_SIZE,NUM_EPOCHS=$NUM_EPOCHS,GAE_LAMBDA=$GAE_LAMBDA,CLIP_EPS=$CLIP_EPS,GAMMA=$GAMMA,OUT_DIR=\"$OUT_DIR\",REPO_ROOT=\"$REPO_ROOT\" \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID"

POST_SCRIPT="scripts/sampled_E_surgical_ppo_sweep/plot_results.py"
SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_surg_ppo \
    --output=slurm/%A_plot.out \
    --error=slurm/%A_plot.err \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=2 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --results_dir '$OUT_DIR' && $PYTHON scripts/sampled_E_surgical_ppo_sweep/analyze_results.py --results_dir '$OUT_DIR'\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Chained Post-Processing Job ID: $POST_JOB_ID (will run after $ARRAY_JOB_ID completes)"
echo "Track status with: squeue -u \$USER"
