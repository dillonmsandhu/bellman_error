#!/bin/bash
# ==============================================================================
# Sampled E(lambda=0.0) Comparison Sweep Launcher
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
WORKER_SCRIPT="scripts/lambda_sweep/lambda_sweep_worker.sh"
DIAGNOSTICS_SCRIPT="scripts/lambda_sweep/sampled_e0_diagnostics.sh"

mkdir -p slurm

DRY_RUN=false
SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
RECIPIENT="ds541@cs.duke.edu"
LAMBDAS="0.0"

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
        -h|--help)
            echo "Usage: $0 [--dry-run] [--sweep-id=ID] [--recipient=EMAIL]"
            exit 0
            ;;
    esac
done

echo "======================================================================"
echo "SAMPLED E(LAMBDA=0.0) 12-TASK COMPARISON DISPATCHER"
echo "Sweep ID:  $SWEEP_ID"
echo "Envs:      ${DEFAULT_ENVS[*]}"
echo "Policies:  ${DEFAULT_POLICIES[*]}"
echo "Algos:     ${DEFAULT_ALGOS[*]}"
echo "Lambdas:   $LAMBDAS"
echo "Recipient: $RECIPIENT"
if [ "$DRY_RUN" = true ]; then
    echo "Mode:      DRY-RUN"
else
    echo "Mode:      SLURM BATCH"
fi
echo "======================================================================"

JOB_IDS=""
SUBMITTED_COUNT=0
MAX_CONCURRENT=10
declare -a SUBMITTED_JOBS

for env in "${DEFAULT_ENVS[@]}"; do
    for policy in "${DEFAULT_POLICIES[@]}"; do
        for algo in "${DEFAULT_ALGOS[@]}"; do
            
            JOB_NAME="e0_${env}_${policy}_${algo}"
            LOG_OUT="slurm/%j_e0_${env}_${policy}_${algo}.out"
            
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
                \"$WORKER_SCRIPT\" \"$env\" \"$policy\" \"$algo\" \"$SWEEP_ID\" --lambdas $LAMBDAS"
            
            echo "--> Submitting: Env=$env, Policy=$policy, Algo=$algo (lambda=$LAMBDAS)"
            
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
echo "Submitted $SUBMITTED_COUNT jobs for Sampled E(0.0) comparison."

if [ "$DRY_RUN" = false ] && [ -n "$JOB_IDS" ]; then
    echo "Submitting chained diagnostics job (triggers after all tasks complete)..."
    DIAG_JOB_NAME="diag_sampled_e0_${SWEEP_ID}"
    DIAG_LOG_OUT="slurm/%j_diag_sampled_e0_${SWEEP_ID}.out"
    
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
    echo "Master plots, CSV ranking, and zip archive will be emailed to $RECIPIENT upon completion."
elif [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Chained diagnostics command:"
    echo "sbatch --dependency=afterany:<ALL_JOBS> --job-name=\"diag_sampled_e0_${SWEEP_ID}\" --partition=\"compsci\" \"$DIAGNOSTICS_SCRIPT\" \"$SWEEP_ID\" \"$RECIPIENT\""
fi
echo "======================================================================"
