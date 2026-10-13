#!/usr/bin/env bash
# scripts/random_policy_sweep_E/run_all_local.sh
#
# Runs Random Policy Sampled E Benchmark across all environments sequentially.
# Automatically updates cross-environment comparison plots after each environment completes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Resolve python interpreter
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo "Error: Python interpreter not found." >&2
    exit 1
fi

NUM_UPDATES=${NUM_UPDATES:-50}
N_SEEDS=${N_SEEDS:-3}
NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-1024}
OUT_DIR=${OUT_DIR:-"results/random_policy_sweep_E"}
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
            echo "  --num-updates N     Number of updates per env (default: $NUM_UPDATES)"
            echo "  --n-seeds N         Number of random seeds (default: $N_SEEDS)"
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

echo "======================================================================"
echo "Random Policy Sampled E Benchmark: Local Sequential Runner"
echo "Repository Root:    $REPO_ROOT"
echo "Python:             $PYTHON"
echo "Total Environments: $TOTAL_ENVS"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Rollout Config:     $NUM_ENVS envs x $NUM_STEPS steps"
echo "Output Directory:   $OUT_DIR"
echo "Skip Completed:     $SKIP_COMPLETED"
echo "======================================================================"

for idx in "${!ENVS[@]}"; do
    ENV_NAME="${ENVS[$idx]}"
    STEP_NUM=$((idx + 1))
    ENV_OUT="$OUT_DIR/$ENV_NAME/out.pkl"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "[$STEP_NUM/$TOTAL_ENVS] Environment: $ENV_NAME"
    echo "Started at: $(date)"
    echo "----------------------------------------------------------------------"

    if [ "$SKIP_COMPLETED" -eq 1 ] && [ -f "$ENV_OUT" ]; then
        echo "--> $ENV_NAME already completed ($ENV_OUT exists). Skipping run."
    else
        $PYTHON scripts/random_policy_sweep_E/run_single_env.py \
            --env_name "$ENV_NAME" \
            --num_updates "$NUM_UPDATES" \
            --n_seeds "$N_SEEDS" \
            --num_envs "$NUM_ENVS" \
            --num_steps "$NUM_STEPS" \
            --minibatch_size "$MINIBATCH_SIZE" \
            --out_dir "$OUT_DIR"
        echo "--> Finished benchmark run for $ENV_NAME."
    fi

    echo "--> Updating cross-environment plots & summary table..."
    $PYTHON scripts/random_policy_sweep_E/plot_results.py \
        --results_dir "$OUT_DIR"

    echo "--> Plots updated successfully."
done

echo ""
echo "======================================================================"
echo "All $TOTAL_ENVS environments finished! Final plots generated in: $OUT_DIR"
echo "Finished at: $(date)"
echo "======================================================================"
