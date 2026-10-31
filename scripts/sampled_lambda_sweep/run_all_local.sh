#!/bin/bash
# scripts/sampled_lambda_sweep/run_all_local.sh
# Runs Sampled PPO Lambda Sweep sequentially on local machine (or GPU workstation).

set -e

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Python interpreter resolution
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

ENVS=(
    "mountaincar"
    "FourRooms-misc"
    "fourrooms-mines"
    "fourrooms-mines-dense"
    "eightrooms-dense"
    "eightrooms-misc"
    "whirlpool-misc"
    "SpaceInvadersExactValue"
    "mountaincar-dense"
)

# Defaults
NUM_UPDATES=100
N_SEEDS=10
SEED=42
K_VAL=32
ACTOR_LR=0.0003
CRITIC_LRS="0.0005 0.001 0.003"
LAMBDAS="0.4 0.8 0.95 0.99"
ALGOS="td_lambda e_lambda e_lambda_fwd"
NUM_ENVS=32
NUM_STEPS=256
MINIBATCH_SIZE=256
NUM_EPOCHS=4
GAE_LAMBDA=0.8
CLIP_EPS=0.2
RUNWAY_LENGTH="adaptive"
OUT_DIR="results/sampled_lambda_sweep"
TARGET_ENV=""
QUICK_TEST=false

for arg in "$@"; do
    case "$arg" in
        --quick-test)
            QUICK_TEST=true
            NUM_UPDATES=2
            N_SEEDS=1
            CRITIC_LRS="0.001"
            LAMBDAS="0.9"
            NUM_ENVS=4
            NUM_STEPS=32
            MINIBATCH_SIZE=64
            OUT_DIR="results/sampled_lambda_sweep_test"
            ;;
        --env=*)
            TARGET_ENV="${arg#*=}"
            ;;
        --out-dir=*)
            OUT_DIR="${arg#*=}"
            ;;
        --updates=*)
            NUM_UPDATES="${arg#*=}"
            ;;
        --seeds=*)
            N_SEEDS="${arg#*=}"
            ;;
        -h|--help)
            echo "Usage: $0 [--quick-test] [--env=ENV_NAME] [--out-dir=DIR] [--updates=N] [--seeds=N]"
            exit 0
            ;;
    esac
done

echo "================================================================"
echo "Running Sampled PPO Lambda Sweep Locally"
echo "Target Env:         ${TARGET_ENV:-ALL 9 ENVS}"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Critic Feature k:   $K_VAL"
echo "Actor LR:           $ACTOR_LR (fixed)"
echo "Critic LRs:         $CRITIC_LRS"
echo "Lambdas:            $LAMBDAS"
echo "Rollout Config:     $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE | Epochs: $NUM_EPOCHS"
echo "Output Directory:   $OUT_DIR"
echo "Python:             $PYTHON"
echo "================================================================"

if [ -n "$TARGET_ENV" ]; then
    RUN_LIST=("$TARGET_ENV")
else
    RUN_LIST=("${ENVS[@]}")
fi

for env_name in "${RUN_LIST[@]}"; do
    echo -e "\n>>> Starting Environment: $env_name"
    $PYTHON scripts/sampled_lambda_sweep/run_single_env.py \
        --env_name "$env_name" \
        --num_updates "$NUM_UPDATES" \
        --n_seeds "$N_SEEDS" \
        --seed "$SEED" \
        --k "$K_VAL" \
        --actor_lr "$ACTOR_LR" \
        --critic_lrs $CRITIC_LRS \
        --lambdas $LAMBDAS \
        --algos $ALGOS \
        --num_envs "$NUM_ENVS" \
        --num_steps "$NUM_STEPS" \
        --minibatch_size "$MINIBATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --gae_lambda "$GAE_LAMBDA" \
        --clip_eps "$CLIP_EPS" \
        --runway_length "$RUNWAY_LENGTH" \
        --out_dir "$OUT_DIR"
done

echo -e "\nAll environments finished. Running consolidated plotting and analysis..."
$PYTHON scripts/sampled_lambda_sweep/plot_results.py --results_dir "$OUT_DIR"
$PYTHON scripts/sampled_lambda_sweep/analyze_results.py --results_dir "$OUT_DIR"

echo "Complete! Results available in $OUT_DIR"
