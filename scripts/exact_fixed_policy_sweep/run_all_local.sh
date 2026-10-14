#!/usr/bin/env bash
# scripts/exact_fixed_policy_sweep/run_all_local.sh
#
# Runs Exact Fixed Policy Evaluation Sweep across all 9 environments locally.
# First verifies / generates fixed policies (Step 0), then evaluates the 5 algorithms.
# Automatically updates consolidated publication plots upon completion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Resolve python interpreter
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

NUM_UPDATES=${NUM_UPDATES:-100}
N_SEEDS=${N_SEEDS:-10}
SEED=${SEED:-42}
K_VALUES=${K_VALUES:-"8 64"}
LRS=${LRS:-"0.0003 0.001 0.003"}
VAL_LAMBDA=${VAL_LAMBDA:-0.8}
POLICY_DIR=${POLICY_DIR:-"results/ppo/ground_truth/fixed_policies"}
POLICY_UPDATES=${POLICY_UPDATES:-50}
OUT_DIR=${OUT_DIR:-"results/exact_fixed_policy_sweep"}
SKIP_COMPLETED=${SKIP_COMPLETED:-0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-completed)
            SKIP_COMPLETED=1
            shift
            ;;
        --force)
            SKIP_COMPLETED=0
            shift
            ;;
        --num-updates)
            NUM_UPDATES="$2"
            shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --val-lambda)
            VAL_LAMBDA="$2"
            shift 2
            ;;
        --policy-dir)
            POLICY_DIR="$2"
            shift 2
            ;;
        --policy-updates)
            POLICY_UPDATES="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

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

mkdir -p "$OUT_DIR"

echo "================================================================"
echo "Step 0: Generating / Verifying Target Fixed Policies"
echo "Policy Directory:   $POLICY_DIR"
echo "Policy Updates:     $POLICY_UPDATES"
echo "================================================================"

$PYTHON scripts/exact_fixed_policy_sweep/generate_fixed_policies.py \
    --policy_dir "$POLICY_DIR" \
    --num_updates "$POLICY_UPDATES" \
    --seed "$SEED"

echo ""
echo "================================================================"
echo "Starting Local Exact Fixed Policy Evaluation Sweep"
echo "Python:             $PYTHON"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Base Seed:          $SEED"
echo "k Values:           $K_VALUES"
echo "Learning Rates:     $LRS"
echo "Value Lambda:       $VAL_LAMBDA"
echo "Output Directory:   $OUT_DIR"
echo "Skip Completed:     $SKIP_COMPLETED"
echo "Environments:       ${#ENVS[@]}"
echo "================================================================"

for idx in "${!ENVS[@]}"; do
    ENV_NAME="${ENVS[$idx]}"
    ENV_OUT_DIR="$OUT_DIR/$ENV_NAME"
    OUT_PKL="$ENV_OUT_DIR/out.pkl"

    echo ""
    echo "================================================================"
    echo "[$((idx + 1))/${#ENVS[@]}] Processing Environment: $ENV_NAME"
    echo "================================================================"

    if [ "$SKIP_COMPLETED" -eq 1 ] && [ -f "$OUT_PKL" ]; then
        echo "Found existing results at $OUT_PKL. Skipping due to --skip-completed."
        continue
    fi

    $PYTHON scripts/exact_fixed_policy_sweep/run_single_env.py \
        --env_name "$ENV_NAME" \
        --num_updates "$NUM_UPDATES" \
        --n_seeds "$N_SEEDS" \
        --seed "$SEED" \
        --k_values $K_VALUES \
        --lrs $LRS \
        --val_lambda "$VAL_LAMBDA" \
        --policy_dir "$POLICY_DIR" \
        --policy_updates "$POLICY_UPDATES" \
        --out_dir "$OUT_DIR"

    echo "Finished $ENV_NAME. Updating plots..."
    $PYTHON scripts/exact_fixed_policy_sweep/plot_results.py \
        --results_dir "$OUT_DIR" || echo "[Warning] Plot generation encountered an error, continuing sweep..."
done

echo ""
echo "================================================================"
echo "All environments completed successfully!"
echo "Generating final consolidated publication figures..."
echo "================================================================"

$PYTHON scripts/exact_fixed_policy_sweep/plot_results.py --results_dir "$OUT_DIR"

echo "Done! Results and figures available in: $OUT_DIR"
