#!/bin/bash
#SBATCH --job-name=sweep_hybrid_ppo
#SBATCH --output=slurm/%j.out
#SBATCH --time=8:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Hybrid PPO Control Algorithms
# Compares E minimization vs TD(lambda) vs MC for exact value function
# optimization while using sampled GAE rollouts for the policy.
#
# Environments: FourRooms-misc and MountainCar-v0
# Optimization Metric: AUC of mean reward (higher is better)
#
# Usage:
#   sbatch scripts/run_slurm_sweep_hybrid_ppo.sh
#   ./scripts/run_slurm_sweep_hybrid_ppo.sh (for local test)
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
N_SEEDS=10
TOTAL_TIMESTEPS=1000000
ENVS=("EightRooms" "FourRooms-misc" "Whirlpool" "MountainCar-v0")
HYBRID_ALGOS=("hybrid_exact_E" "hybrid_exact_td_lambda" "hybrid_exact_mc")

FIXED_GAE_LAMBDA=0.8
# Grids (2 critic LRs, 2 actor LRs, fixed lambda=0.9 -> 4 configs per seed)
LR_GRID="0.001 0.0003"
ACTOR_LR_GRID="0.001 0.0003"
VALUE_LAMBDA_GRID="0.9 0.95"
CONFIG="{\"GAE_LAMBDA\": $FIXED_GAE_LAMBDA, \"NUM_STEPS\": 256, \"NUM_ENVS\": 64, \"MINIBATCH_SIZE\": 1024, \"TOTAL_TIMESTEPS\": 1000000, \"NUM_EPOCHS\": 4}"

mkdir -p slurm

echo "======================================================================"
echo "STARTING SLURM HYBRID PPO / CONTROL ALGORITHMS SWEEP"
echo "Comparing: Hybrid E minimization vs TD(lambda) vs MC"
echo "Start Time: $START_TIME"
echo "Environments: ${ENVS[*]}"
echo "Algorithms: ${HYBRID_ALGOS[*]}"
echo "Seeds: $N_SEEDS | Timesteps: $TOTAL_TIMESTEPS"
echo "Critic LR Grid: $LR_GRID | Actor LR Grid: $ACTOR_LR_GRID | Lambda: $VALUE_LAMBDA_GRID"
echo "Optimization Metric: V_start (AUC, higher is better)"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Running Hybrid PPO Control Sweep: Environment=$env"
    echo "======================================================================"
    
    CMD="$PYTHON scripts/sweep_pipeline.py \
        --policy ppo \
        --env-name $env \
        --algos ${HYBRID_ALGOS[*]} \
        --lr-grid $LR_GRID \
        --actor-lr-grid $ACTOR_LR_GRID \
        --lambda-grid $VALUE_LAMBDA_GRID \
        --config '$CONFIG' \
        --n-seeds $N_SEEDS \
        --total-timesteps $TOTAL_TIMESTEPS \
        --metric V_start \
        --rank-by auc \
        --higher-is-better \
        --sweep-suffix hybrid \
        --no-log-scale"
    echo "Command: $CMD"
    eval "$CMD"
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "Hybrid PPO Control Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "Job finished at: $END_TIME"
echo "======================================================================"
