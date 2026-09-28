#!/bin/bash
# ==============================================================================
# Multi-GPU SLURM Sampled E(lambda) Sweep Launcher (SLURM Job Array)
#
# Loops over all 12 tasks (4 envs x 3 policies) and submits a single SLURM job array
# for the 4 sampling implementations of E(lambda):
#   1. sampled_E (E(0.0) original formulation)
#   2. E_lambda_fixed (sweeping lambda in {0.0, 0.9})
#   3. E_lambda_geometric (sweeping lambda in {0.0, 0.9})
#   4. E_lambda_differentiable (sweeping lambda in {0.0, 0.9})
#
# After jobs complete, runs sampled_lambda_diagnostics to produce the 2-column
# master summary comparison and rankings.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Config
DEFAULT_ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")
DEFAULT_POLICIES=("random" "fixed" "ppo")
DEFAULT_ALGOS=("sampled_E" "E_lambda_fixed" "E_lambda_geometric" "E_lambda_differentiable")

PARTITION="compsci-gpu"
TIME_LIMIT="4:00:00"
GPU_GRES="gpu:a5000:1"
WORKER_SCRIPT="scripts/lambda_sweep/lambda_sweep_array_worker.sh"
DIAGNOSTICS_SCRIPT="scripts/lambda_sweep/sampled_lambda_diagnostics.sh"

mkdir -p slurm slurm/manifests

DRY_RUN=false
SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
LAMBDAS="0.0 0.9"
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
        --lambdas=*)
            LAMBDAS="${arg#*=}"
            ;;
        --max-concurrent=*)
            MAX_CONCURRENT="${arg#*=}"
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--sweep-id=ID] [--lambdas=\"0.0 0.9\"] [--max-concurrent=10]"
            exit 0
            ;;
    esac
done

# Build task manifest (env policy algo lambdas...)
TASKS=()
for env in "${DEFAULT_ENVS[@]}"; do
    for policy in "${DEFAULT_POLICIES[@]}"; do
        for algo in "${DEFAULT_ALGOS[@]}"; do
            if [ "$algo" = "sampled_E" ]; then
                ALGO_LAMBDAS="0.0"
            else
                ALGO_LAMBDAS="$LAMBDAS"
            fi
            TASKS+=("$env $policy $algo $ALGO_LAMBDAS")
        done
    done
done
NUM_TASKS=${#TASKS[@]}

MANIFEST_FILE="slurm/manifests/tasks_lambda_${SWEEP_ID}.txt"
printf "%s\n" "${TASKS[@]}" > "$MANIFEST_FILE"

echo "======================================================================"
echo "SAMPLED E(LAMBDA) SWEEP DISPATCHER (SLURM ARRAY)"
echo "Sweep ID:       $SWEEP_ID"
echo "Total Tasks:    $NUM_TASKS"
echo "Max Concurrent: $MAX_CONCURRENT"
echo "Lambdas:        $LAMBDAS"
echo "GAE Lambda:     $GAE_LAMBDA"
echo "Envs x Steps:   $NUM_ENVS x $NUM_STEPS"
echo "Manifest:       $MANIFEST_FILE"
echo "Config:         $CONFIG"
if [ "$DRY_RUN" = true ]; then
    echo "Mode:           DRY-RUN"
else
    echo "Mode:           SLURM BATCH ARRAY"
fi
echo "======================================================================"

ARRAY_CMD="sbatch --parsable \
    --array=0-$((NUM_TASKS - 1))%${MAX_CONCURRENT} \
    --job-name=\"samp_e_${SWEEP_ID}\" \
    --output=\"slurm/%A_%a.out\" \
    --time=\"$TIME_LIMIT\" \
    --partition=\"$PARTITION\" \
    --gres=\"$GPU_GRES\" \
    \"$WORKER_SCRIPT\" \"$MANIFEST_FILE\" \"$SWEEP_ID\" --config '$CONFIG'"

if [ "$DRY_RUN" = true ]; then
    echo "--> [DRY-RUN] Array Command:"
    echo "$ARRAY_CMD"
    echo ""
    echo "--> [DRY-RUN] Chained Diagnostics Command:"
    echo "sbatch --dependency=afterany:<ARRAY_JOB_ID> --job-name=\"samp_e_diag_${SWEEP_ID}\" --partition=\"$PARTITION\" \"$DIAGNOSTICS_SCRIPT\" \"$SWEEP_ID\""
else
    echo "--> Submitting SLURM Job Array (0-$((NUM_TASKS - 1))%${MAX_CONCURRENT})..."
    ARRAY_JOB_ID=$(eval "$ARRAY_CMD")
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ] && [ -n "$ARRAY_JOB_ID" ]; then
        echo "    Status: SUBMITTED (Array Job ID: $ARRAY_JOB_ID)"
        echo "    Monitor progress: squeue -j $ARRAY_JOB_ID"
        
        echo ""
        echo "Submitting chained diagnostics job (triggers after array $ARRAY_JOB_ID completes)..."
        DIAG_JOB_NAME="samp_e_diag_${SWEEP_ID}"
        DIAG_LOG_OUT="slurm/%j_samp_e_diag_${SWEEP_ID}.out"
        
        DIAG_CMD="sbatch \
            --dependency=afterany:$ARRAY_JOB_ID \
            --job-name=\"$DIAG_JOB_NAME\" \
            --output=\"$DIAG_LOG_OUT\" \
            --time=\"01:00:00\" \
            --partition=\"$PARTITION\" \
            \"$DIAGNOSTICS_SCRIPT\" \"$SWEEP_ID\""
        
        DIAG_OUTPUT=$(eval "$DIAG_CMD")
        DIAG_JOB_ID=$(echo "$DIAG_OUTPUT" | awk '{print $NF}')
        echo "Diagnostics job scheduled with Job ID: $DIAG_JOB_ID"
    else
        echo "    Status: FAILED ARRAY SUBMISSION (Exit code: $EXIT_CODE)"
        exit 1
    fi
fi
echo "======================================================================"
