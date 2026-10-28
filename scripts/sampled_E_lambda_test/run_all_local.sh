#!/usr/bin/env bash
# scripts/sampled_E_lambda_test/run_all_local.sh
#
# Sequentially executes Sampled E(lambda) tests on all environments locally.
# Automatically updates cross-environment comparison plots after each environment finishes.

set -euo pipefail

# Determine script and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Resolve python interpreter
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo "Error: Python interpreter not found." >&2
    exit 1
fi

# Configuration defaults
VALUE_LAMBDA=${VALUE_LAMBDA:-0.8}
NUM_UPDATES=${NUM_UPDATES:-50}
N_SEEDS=${N_SEEDS:-3}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-128}
NUM_STEPS=${NUM_STEPS:-128}
OUT_DIR=${OUT_DIR:-"results/sampled_E_lambda_test"}
SKIP_COMPLETED=${SKIP_COMPLETED:-0}

# Parse optional command line flags
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
        --value-lambda)
            VALUE_LAMBDA="$2"
            shift 2
            ;;
        --num-updates)
            NUM_UPDATES="$2"
            shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-completed    Skip environments that already have out.pkl in OUT_DIR"
            echo "  --force             Re-run all environments even if out.pkl exists (default)"
            echo "  --value-lambda L    Trace parameter lambda (default: $VALUE_LAMBDA)"
            echo "  --num-updates N     Number of policy updates per env (default: $NUM_UPDATES)"
            echo "  --n-seeds N         Number of seeds (default: $N_SEEDS)"
            echo "  --out-dir PATH      Output directory (default: $OUT_DIR)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ENVS=(
    "fourrooms-dense"
    "FourRooms-misc"
    "fourrooms-mines"
    "fourrooms-mines-dense"
    "eightrooms-dense"
    "eightrooms-misc"
    "whirlpool-misc"
    "SpaceInvadersExactValue"
    "mountaincar-dense"
)

TOTAL_ENVS=${#ENVS[@]}
RESULTS_SUBDIR="$OUT_DIR/lambda_$VALUE_LAMBDA"

echo "======================================================================"
echo "Sampled E(lambda) Test: Local Sequential Runner"
echo "Repository Root:    $REPO_ROOT"
echo "Python:             $PYTHON"
echo "Value Lambda:       $VALUE_LAMBDA"
echo "Total Environments: $TOTAL_ENVS"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Rollout Config:     $NUM_TRAJECTORIES trajs x $NUM_STEPS steps"
echo "Output Directory:   $RESULTS_SUBDIR"
echo "Skip Completed:     $SKIP_COMPLETED"
echo "======================================================================"

for idx in "${!ENVS[@]}"; do
    ENV_NAME="${ENVS[$idx]}"
    STEP_NUM=$((idx + 1))
    ENV_OUT="$RESULTS_SUBDIR/$ENV_NAME/out.pkl"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "[$STEP_NUM/$TOTAL_ENVS] Environment: $ENV_NAME (lambda=$VALUE_LAMBDA)"
    echo "Started at: $(date)"
    echo "----------------------------------------------------------------------"

    if [ "$SKIP_COMPLETED" -eq 1 ] && [ -f "$ENV_OUT" ]; then
        echo "--> $ENV_NAME already completed ($ENV_OUT exists). Skipping run."
    else
        $PYTHON scripts/sampled_E_lambda_test/run_single_env.py \
            --env_name "$ENV_NAME" \
            --value_lambda "$VALUE_LAMBDA" \
            --num_updates "$NUM_UPDATES" \
            --n_seeds "$N_SEEDS" \
            --num_trajectories "$NUM_TRAJECTORIES" \
            --num_steps "$NUM_STEPS" \
            --out_dir "$OUT_DIR" \
            --save_metrics
        echo "--> Finished training run for $ENV_NAME."
    fi

    echo "--> Updating cross-environment plots & summary table..."
    $PYTHON scripts/sampled_E_lambda_test/plot_comparison.py \
        --results_dir "$RESULTS_SUBDIR"

    echo "--> Plots updated successfully."
done

echo ""
echo "======================================================================"
echo "All $TOTAL_ENVS environments finished! Final plots generated in: $RESULTS_SUBDIR"
echo "Finished at: $(date)"
echo "======================================================================"
