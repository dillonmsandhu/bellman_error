#!/bin/bash
# ==============================================================================
# Sampled E(lambda=0.0) Comparison Sweep Launcher (SLURM Job Array)
#
# Runs all 4 sampling implementations of E at lambda=0.0 across all 12 tasks:
#   1. sampled_E (baseline formulation)
#   2. E_lambda_fixed (fixed / stop-gradient)
#   3. E_lambda_geometric (geometric jumps)
#   4. E_lambda_differentiable (differentiable scan)
#
# Tasks (12 total):
#   - Policy Eval Tasks: [EightRooms, FourRooms-misc, Whirlpool, MountainCar-v0] x [random, fixed]
#   - PPO Tasks:        [EightRooms, FourRooms-misc, Whirlpool, MountainCar-v0] x [ppo]
#
# Diagnostics & Outputs:
#   - Value learning curves & Greedy accuracy curves for Policy Eval tasks
#   - Performance return curves for PPO tasks
#   - Master summary PDF & PNG
#   - Zip archive containing all task plots
#   - Automatically emailed to recipient (default: ds541@cs.duke.edu)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")
DEFAULT_POLICIES=("random" "fixed" "ppo")
DEFAULT_ALGOS=("sampled_E" "E_lambda_fixed" "E_lambda_geometric" "E_lambda_differentiable")

PARTITION="compsci-gpu"
TIME_LIMIT="4:00:00"
GPU_GRES="gpu:a5000:1"
WORKER_SCRIPT="scripts/lambda_sweep/lambda_sweep_array_worker.sh"
DIAGNOSTICS_SCRIPT="scripts/lambda_sweep/sampled_e0_diagnostics.sh"

mkdir -p slurm slurm/manifests

DRY_RUN=false
SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
RECIPIENT="ds541@cs.duke.edu"
LAMBDAS="0.0"
MAX_CONCURRENT=10

# ==============================================================================
# Hard-coded Training Hyperparameters for Sampled E
# ==============================================================================
NUM_EPOCHS=4
MINIBATCH_SIZE=1024
TOTAL_TIMESTEPS=1000000
NUM_ENVS=128
NUM_STEPS=128
GAE_LAMBDA=0.8
CONFIG="{\"NUM_ENVS\":$NUM_ENVS,\"NUM_STEPS\":$NUM_STEPS,\"TOTAL_TIMESTEPS\":$TOTAL_TIMESTEPS,\"MINIBATCH_SIZE\":$MINIBATCH_SIZE,\"NUM_EPOCHS\":$NUM_EPOCHS,\"LIGHT_METRICS\":true,\"GAE_LAMBDA\":$GAE_LAMBDA}"

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --sweep-id=*)
            SWEEP_ID="${arg#*=}"
            ;;
        --recipient=*)
            RECIPIENT="${arg#*=}"
            ;;
        --max-concurrent=*)
            MAX_CONCURRENT="${arg#*=}"
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--sweep-id=ID] [--recipient=EMAIL] [--max-concurrent=10]"
            exit 0
            ;;
    esac
done

# Build task manifest (env policy algo)
TASKS=()
for env in "${DEFAULT_ENVS[@]}"; do
    for policy in "${DEFAULT_POLICIES[@]}"; do
        for algo in "${DEFAULT_ALGOS[@]}"; do
            TASKS+=("$env $policy $algo")
        done
    done
done
NUM_TASKS=${#TASKS[@]}

MANIFEST_FILE="slurm/manifests/tasks_e0_${SWEEP_ID}.txt"
printf "%s\n" "${TASKS[@]}" > "$MANIFEST_FILE"

echo "======================================================================"
echo "SAMPLED E(LAMBDA=0.0) 12-TASK COMPARISON DISPATCHER (SLURM ARRAY)"
echo "Sweep ID:       $SWEEP_ID"
echo "Total Tasks:    $NUM_TASKS"
echo "Max Concurrent: $MAX_CONCURRENT"
echo "Lambdas:        $LAMBDAS"
echo "GAE Lambda:     $GAE_LAMBDA"
echo "Envs x Steps:   $NUM_ENVS x $NUM_STEPS"
echo "Manifest:       $MANIFEST_FILE"
echo "Config:         $CONFIG"
echo "Recipient:      $RECIPIENT"
if [ "$DRY_RUN" = true ]; then
    echo "Mode:           DRY-RUN"
else
    echo "Mode:           SLURM BATCH ARRAY"
fi
echo "======================================================================"

ARRAY_CMD="sbatch --parsable \
    --array=0-$((NUM_TASKS - 1))%${MAX_CONCURRENT} \
    --job-name=\"e0_${SWEEP_ID}\" \
    --output=\"slurm/%A_%a.out\" \
    --time=\"$TIME_LIMIT\" \
    --partition=\"$PARTITION\" \
    --gres=\"$GPU_GRES\" \
    \"$WORKER_SCRIPT\" \"$MANIFEST_FILE\" \"$SWEEP_ID\" --lambdas $LAMBDAS --config '$CONFIG'"

if [ "$DRY_RUN" = true ]; then
    echo "--> [DRY-RUN] Array Command:"
    echo "$ARRAY_CMD"
    echo ""
    echo "--> [DRY-RUN] Chained Diagnostics Command:"
    echo "sbatch --dependency=afterany:<ARRAY_JOB_ID> --job-name=\"diag_sampled_e0_${SWEEP_ID}\" --partition=\"compsci\" \"$DIAGNOSTICS_SCRIPT\" \"$SWEEP_ID\" \"$RECIPIENT\""
else
    echo "--> Submitting SLURM Job Array (0-$((NUM_TASKS - 1))%${MAX_CONCURRENT})..."
    ARRAY_JOB_ID=$(eval "$ARRAY_CMD")
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ] && [ -n "$ARRAY_JOB_ID" ]; then
        echo "    Status: SUBMITTED (Array Job ID: $ARRAY_JOB_ID)"
        echo "    Monitor progress: squeue -j $ARRAY_JOB_ID"
        
        echo ""
        echo "Submitting chained diagnostics job (triggers after array $ARRAY_JOB_ID completes)..."
        DIAG_JOB_NAME="diag_sampled_e0_${SWEEP_ID}"
        DIAG_LOG_OUT="slurm/%j_diag_sampled_e0_${SWEEP_ID}.out"
        
        DIAG_CMD="sbatch \
            --dependency=afterany:$ARRAY_JOB_ID \
            --job-name=\"$DIAG_JOB_NAME\" \
            --output=\"$DIAG_LOG_OUT\" \
            --time=\"0:30:00\" \
            --partition=\"compsci\" \
            --cpus-per-task=4 \
            \"$DIAGNOSTICS_SCRIPT\" \"$SWEEP_ID\" \"$RECIPIENT\""
        
        DIAG_OUTPUT=$(eval "$DIAG_CMD")
        DIAG_JOB_ID=$(echo "$DIAG_OUTPUT" | awk '{print $NF}')
        echo "Diagnostics job scheduled with Job ID: $DIAG_JOB_ID"
        echo "Master plots, CSV ranking, and zip archive will be emailed to $RECIPIENT upon completion."
    else
        echo "    Status: FAILED ARRAY SUBMISSION (Exit code: $EXIT_CODE)"
        exit 1
    fi
fi
echo "======================================================================"
