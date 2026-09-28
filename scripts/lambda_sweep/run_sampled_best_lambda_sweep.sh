#!/bin/bash
# ==============================================================================
# Sampled E Lambda-Tuning Sweep Launcher
#
# Sweeps over a spread of VALUE_LAMBDA values (including 0.0) across all 12 tasks:
#   1. sampled_E (baseline lambda=0.0)
#   2. E_lambda_fixed (swept over lambdas)
#   3. E_lambda_geometric (swept over lambdas)
#   4. E_lambda_differentiable (swept over lambdas)
#
# Tasks (12 total):
#   - Policy Eval Tasks: [EightRooms, FourRooms-misc, Whirlpool, MountainCar-v0] x [random, fixed]
#   - PPO Tasks:        [EightRooms, FourRooms-misc, Whirlpool, MountainCar-v0] x [ppo]
#
# Post-Processing & Automated Diagnostics:
#   - Automatically identifies the best-performing lambda for each algorithm
#   - Plots Value Learning Curves & Greedy Accuracy for Policy Eval tasks (at best lambda)
#   - Plots Performance Return Curves (V_start) for PPO tasks (at best lambda)
#   - Outputs best_lambda_comparison_metrics.csv indicating which lambda was best
#   - Generates 5-column master summary PDF & PNG
#   - Compresses all task plots into best_lambda_plots.zip
#   - Automatically emails all results and the zip archive to recipient
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
WORKER_SCRIPT="scripts/lambda_sweep/lambda_sweep_worker.sh"
DIAGNOSTICS_SCRIPT="scripts/lambda_sweep/sampled_best_lambda_diagnostics.sh"

mkdir -p slurm

DRY_RUN=false
SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
RECIPIENT="ds541@cs.duke.edu"
LAMBDAS="0.0 0.2 0.5 0.8 0.9 0.95 0.99"

# ==============================================================================
# Hard-coded Training Hyperparameters for Sampled E
# ==============================================================================
NUM_EPOCHS=4
MINIBATCH_SIZE=1024
TOTAL_TIMESTEPS=1000000
NUM_ENVS=64
NUM_STEPS=256
CONFIG="{\"NUM_ENVS\":$NUM_ENVS,\"NUM_STEPS\":$NUM_STEPS,\"TOTAL_TIMESTEPS\":$TOTAL_TIMESTEPS,\"MINIBATCH_SIZE\":$MINIBATCH_SIZE,\"NUM_EPOCHS\":$NUM_EPOCHS,\"LIGHT_METRICS\":true}"

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
        --recipient=*)
            RECIPIENT="${arg#*=}"
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--sweep-id=ID] [--lambdas=\"0.0 0.2 ...\"] [--recipient=EMAIL]"
            exit 0
            ;;
    esac
done

echo "======================================================================"
echo "SAMPLED E LAMBDA-TUNING SWEEP DISPATCHER"
echo "Sweep ID:   $SWEEP_ID"
echo "Envs:       ${DEFAULT_ENVS[*]}"
echo "Policies:   ${DEFAULT_POLICIES[*]}"
echo "Algos:      ${DEFAULT_ALGOS[*]}"
echo "Lambdas:    $LAMBDAS"
echo "Config:     $CONFIG"
echo "Recipient:  $RECIPIENT"
if [ "$DRY_RUN" = true ]; then
    echo "Mode:       DRY-RUN"
else
    echo "Mode:       SLURM BATCH"
fi
echo "======================================================================"

JOB_IDS=""
SUBMITTED_COUNT=0
MAX_CONCURRENT=10
declare -a SUBMITTED_JOBS

for env in "${DEFAULT_ENVS[@]}"; do
    for policy in "${DEFAULT_POLICIES[@]}"; do
        for algo in "${DEFAULT_ALGOS[@]}"; do
            
            # sampled_E only implements lambda=0.0
            if [ "$algo" = "sampled_E" ]; then
                ALGO_LAMBDAS="0.0"
            else
                ALGO_LAMBDAS="$LAMBDAS"
            fi

            JOB_NAME="tune_${env}_${policy}_${algo}"
            LOG_OUT="slurm/%j_tune_${env}_${policy}_${algo}.out"
            
            DEP=""
            if [ $SUBMITTED_COUNT -ge $MAX_CONCURRENT ]; then
                DEP_JOB_IDX=$((SUBMITTED_COUNT - MAX_CONCURRENT))
                DEP_JOB_ID=${SUBMITTED_JOBS[$DEP_JOB_IDX]}
                DEP="--dependency=afterany:$DEP_JOB_ID"
            fi
            
            SBATCH_CMD="sbatch \
                $DEP \
                --job-name=\"$JOB_NAME\" \
                --output=\"$LOG_OUT\" \
                --time=\"$TIME_LIMIT\" \
                --partition=\"$PARTITION\" \
                --gres=\"$GPU_GRES\" \
                \"$WORKER_SCRIPT\" \"$env\" \"$policy\" \"$algo\" \"$SWEEP_ID\" --lambdas $ALGO_LAMBDAS --config '$CONFIG'"
            
            echo "--> Submitting: Env=$env, Policy=$policy, Algo=$algo (lambdas: $ALGO_LAMBDAS)"
            
            if [ "$DRY_RUN" = true ]; then
                echo "    [DRY-RUN] Command: $SBATCH_CMD"
            else
                JOB_OUTPUT=$(eval "$SBATCH_CMD")
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 0 ]; then
                    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')
                    echo "    Status: SUBMITTED (Job ID: $JOB_ID)"
                    if [ -z "$JOB_IDS" ]; then
                        JOB_IDS="$JOB_ID"
                    else
                        JOB_IDS="${JOB_IDS}:${JOB_ID}"
                    fi
                    SUBMITTED_JOBS+=($JOB_ID)
                    SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
                else
                    echo "    Status: FAILED SUBMISSION (Exit code: $EXIT_CODE)"
                fi
            fi

        done
    done
done

echo "======================================================================"
echo "Submitted $SUBMITTED_COUNT jobs for Sampled E Lambda-Tuning Sweep."

if [ "$DRY_RUN" = false ] && [ -n "$JOB_IDS" ]; then
    echo "Submitting chained diagnostics job (triggers after all tasks complete)..."
    DIAG_JOB_NAME="diag_best_lambda_${SWEEP_ID}"
    DIAG_LOG_OUT="slurm/%j_diag_best_lambda_${SWEEP_ID}.out"
    
    DIAG_CMD="sbatch \
        --dependency=afterany:$JOB_IDS \
        --job-name=\"$DIAG_JOB_NAME\" \
        --output=\"$DIAG_LOG_OUT\" \
        --time=\"0:30:00\" \
        --partition=\"compsci\" \
        --cpus-per-task=4 \
        \"$DIAGNOSTICS_SCRIPT\" \"$SWEEP_ID\" \"$RECIPIENT\""
    
    DIAG_OUTPUT=$(eval "$DIAG_CMD")
    DIAG_JOB_ID=$(echo "$DIAG_OUTPUT" | awk '{print $NF}')
    echo "Diagnostics job scheduled with Job ID: $DIAG_JOB_ID"
    echo "Winning lambda curves, CSV rankings, and zip archive will be emailed to $RECIPIENT upon completion."
elif [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Chained diagnostics command:"
    echo "sbatch --dependency=afterany:<ALL_JOBS> --job-name=\"diag_best_lambda_${SWEEP_ID}\" --partition=\"compsci\" \"$DIAGNOSTICS_SCRIPT\" \"$SWEEP_ID\" \"$RECIPIENT\""
fi
echo "======================================================================"
