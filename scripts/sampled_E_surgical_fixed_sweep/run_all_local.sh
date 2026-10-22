#!/bin/bash
# scripts/sampled_E_surgical_fixed_sweep/run_all_local.sh
# Runs all 9 environments sequentially on local machine, then compiles figures and tables.

set -e

[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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
OUT_DIR=${11:-"results/sampled_E_surgical_fixed_sweep"}
POLICY_DIR=${12:-"ground_truth/fixed_policies"}

if [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

ALL_ENVS=(
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

echo "=================================================================="
echo "Running All 9 Environments Locally: Sampled E-Surgical Fixed Sweep"
echo "Python: $PYTHON"
echo "Policy Dir: $POLICY_DIR"
echo "Output Directory: $OUT_DIR"
echo "=================================================================="

for idx in "${!ALL_ENVS[@]}"; do
    ENV_NAME="${ALL_ENVS[$idx]}"
    echo ""
    echo ">>> [$((idx + 1))/9] Starting Environment: $ENV_NAME <<<"
    $PYTHON scripts/sampled_E_surgical_fixed_sweep/run_single_env.py \
        --env_name "$ENV_NAME" \
        --policy_dir "$POLICY_DIR" \
        --num_updates "$NUM_UPDATES" \
        --n_seeds "$N_SEEDS" \
        --seed "$SEED" \
        --k_values $K_VALUES \
        --critic_lrs $CRITIC_LRS \
        --num_envs "$NUM_ENVS" \
        --num_steps "$NUM_STEPS" \
        --minibatch_size "$MINIBATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --val_lambda "$VAL_LAMBDA" \
        --out_dir "$OUT_DIR"
done

echo ""
echo ">>> Generating Consolidated Plots & Analysis Tables <<<"
$PYTHON scripts/sampled_E_surgical_fixed_sweep/plot_results.py --results_dir "$OUT_DIR"
$PYTHON scripts/sampled_E_surgical_fixed_sweep/analyze_results.py --results_dir "$OUT_DIR"

echo "All fixed policy experiments, plots, and analysis tables completed successfully!"
