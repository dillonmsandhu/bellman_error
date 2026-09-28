#!/bin/bash
# ==============================================================================
# Run TD(lambda), E(lambda), E, and MC on EightRooms and EightRooms-Dense
#
# Usage:
#   ./scripts/run_dense_vs_sparse.sh [--suffix SUFFIX] [--timesteps N] [--n-seeds N] [--save-video]
# ==============================================================================

set -e

# Default values
RUN_SUFFIX="dense_vs_sparse"
TIMESTEPS=1000
N_SEEDS=1
EXTRA_FLAGS=()
SAVE_VIDEO=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suffix)
            RUN_SUFFIX="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --save-video)
            SAVE_VIDEO="--save-video"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --suffix SUFFIX       Run suffix directory (default: dense_vs_sparse)"
            echo "  --timesteps N         Total timesteps / updates (default: 1000)"
            echo "  --n-seeds N           Number of seeds to run (default: 1)"
            echo "  --save-video          Save environment rollout video GIFs"
            exit 0
            ;;
        *)
            if [[ -z "$RUN_SUFFIX_SET" ]]; then
                RUN_SUFFIX="$1"
                RUN_SUFFIX_SET=1
                shift
            else
                echo "Unknown option: $1"
                exit 1
            fi
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Determine python command (prioritize purejaxrl pyenv if available)
if [ -f "/Users/dillonsandhu/.pyenv/versions/3.10.9/envs/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/3.10.9/envs/purejaxrl/bin/python"
elif [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

CONFIG="{\"TOTAL_TIMESTEPS\": ${TIMESTEPS}, \"LIGHT_METRICS\": false}"
ENV_IDS="EightRooms EightRooms-Dense"

# Algorithms to run:
# 1. TD(lambda)
# 2. E(lambda)
# 3. E (Exact Bellman Error minimization)
# 4. MC (Exact Monte Carlo)
ALGOS=(
    "ppo.exact_td_lambda"
    "ppo.exact_E_lambda"
    "ppo.exact_E"
    "ppo.exact_mc"
)

echo "======================================================================"
echo "RUNNING DENSE VS SPARSE COMPARISON"
echo "Algorithms:  TD(λ), E(λ), E, MC"
echo "Environments: EightRooms, EightRooms-Dense"
echo "Run Suffix:   $RUN_SUFFIX"
echo "Timesteps:    $TIMESTEPS"
echo "Seeds:        $N_SEEDS"
echo "Python:       $PYTHON"
echo "======================================================================"

for ALGO in "${ALGOS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Starting Algorithm: $ALGO"
    echo "======================================================================"
    
    CMD="$PYTHON -m $ALGO \
        --env-ids $ENV_IDS \
        --run-suffix $RUN_SUFFIX \
        --n-seeds $N_SEEDS \
        --config '$CONFIG' \
        --save-metrics $SAVE_VIDEO"
    
    echo "Running: $CMD"
    eval "$CMD"
    echo "Finished Algorithm: $ALGO"
done

echo ""
echo "======================================================================"
echo "All runs completed successfully for suffix: $RUN_SUFFIX"
echo "To analyze and compare results, run:"
echo "  marimo edit notebooks/compare_dense_vs_sparse.py"
echo "======================================================================"
