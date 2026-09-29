#!/bin/bash
#SBATCH --job-name=fourrooms_lambda_array
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

# ==============================================================================
# SLURM Job Array: FourRooms & EightRooms Benchmark
#
# Environments (1 sub-job per environment, array 0-3):
#   - Task 0: FourRooms-misc (regular discrete 4 rooms, sparse)
#   - Task 1: continuous-fourrooms (continuous 4 rooms, sparse)
#   - Task 2: continuous-fourrooms-dense (continuous 4 rooms, dense PBRS)
#   - Task 3: EightRooms-Dense (discrete 8 rooms, dense PBRS)
#
# In each sub-job, runs 5 distinct runs with 8 seeds using exact settings:
#   NUM_TIMESTEPS=1000, NUM_EPOCHS=1, NUM_MINIBATCHES=1
#
# Algorithms:
#   1. Ground Truth PPO
#   2. Exact E-Lambda PPO (λ = 0.0)
#   3. Exact E-Lambda PPO (λ = 0.8)
#   4. Exact E-Lambda PPO (λ = 0.95)
#   5. Exact E-Lambda PPO (λ = 1.0)
#
# Usage:
#   sbatch scripts/fourrooms_lambda_sweep/run_slurm_fourrooms_array.sh
#   Or test locally:
#   ./scripts/fourrooms_lambda_sweep/run_slurm_fourrooms_array.sh [0|1|2|3]
# ==============================================================================

set -e

mkdir -p slurm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Default configuration
export N_SEEDS=${N_SEEDS:-8}
export TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000}
export NUM_TIMESTEPS=${NUM_TIMESTEPS:-1000}

# Sweep ID shared across array tasks
if [ -z "$SWEEP_ID" ]; then
    if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
        export SWEEP_ID="fourrooms_${SLURM_ARRAY_JOB_ID}"
    else
        export SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
    fi
fi

# Determine target environment from CLI arg ($1) or SLURM_ARRAY_TASK_ID
TARGET_ENV=$1
if [ -z "$TARGET_ENV" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    TARGET_ENV="$SLURM_ARRAY_TASK_ID"
fi

if [ -n "$TARGET_ENV" ]; then
    # Run single assigned environment sub-job
    exec "$SCRIPT_DIR/fourrooms_array_worker.sh" "$TARGET_ENV" "$SWEEP_ID"
else
    # Sequential local fallback for testing without SLURM
    echo "No task index provided; running all 4 environments sequentially..."
    for task_idx in 0 1 2 3; do
        "$SCRIPT_DIR/fourrooms_array_worker.sh" "$task_idx" "$SWEEP_ID"
    done
fi
