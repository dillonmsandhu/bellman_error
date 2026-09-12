#!/bin/bash
#SBATCH --job-name=sweep_exact_ppo
#SBATCH --output=slurm/%j.out
#SBATCH --time=12:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Exact PPO Control Algorithms
# Compares E minimization vs TD(lambda) for policy improvement.
#
# Environments: FourRooms-misc and MountainCar-v0
# Optimization Metric: AUC of start-state value V_start (higher is better)
#
# Usage:
#   sbatch scripts/run_slurm_sweep_exact_ppo.sh
#   ./scripts/run_slurm_sweep_exact_ppo.sh (for local test)
# ==============================================================================
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Python environment (defaults to cluster gymnax environment)
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

# Configuration
N_SEEDS=12
TOTAL_TIMESTEPS=2000
ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")
EXACT_ALGOS=("exact_E" "exact_td_lambda")


FIXED_GAE_LAMBDA=0.1
# Grids (2 critic LRs, 2 actor LRs, fixed lambda=0.9 -> 4 configs per seed)
LR_GRID="0.01 0.005 0.001 0.0003"
ACTOR_LR_GRID="0.001 0.0003 0.0001"
VALUE_LAMBDA_GRID="0.9 0.99 1.0"

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM EXACT PPO / CONTROL ALGORITHMS SWEEP"
echo "Comparing: E minimization vs TD(lambda)"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Algorithms: ${EXACT_ALGOS[*]}"
echo "Seeds: $N_SEEDS | Timesteps: $TOTAL_TIMESTEPS"
echo "Critic LR Grid: $LR_GRID | Actor LR Grid: $ACTOR_LR_GRID | Lambda: $VALUE_LAMBDA_GRID"
echo "Optimization Metric: V_start (AUC, higher is better)"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Running PPO Control Sweep: Environment=$env"
    echo "======================================================================"
    
    CMD="$PYTHON scripts/sweep_pipeline.py \
        --policy ppo \
        --env-name $env \
        --algos ${EXACT_ALGOS[*]} \
        --lr-grid $LR_GRID \
        --actor-lr-grid $ACTOR_LR_GRID \
        --lambda-grid $VALUE_LAMBDA_GRID \
        --config '{\"GAE_LAMBDA\": $FIXED_GAE_LAMBDA}' \
        --n-seeds $N_SEEDS \
        --total-timesteps $TOTAL_TIMESTEPS \
        --metric V_start \
        --rank-by auc \
        --higher-is-better \
        --sweep-suffix exact \
        --no-log-scale"
    echo "Command: $CMD"
    eval "$CMD"
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "Exact PPO Control Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
