#!/bin/bash
#SBATCH --job-name=sweep_epsilon
#SBATCH --output=slurm/%j.out
#SBATCH --time=4:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1

# ==============================================================================
# Hyperparameter Sweep over Epsilon values for Sampled Algorithms 
# Evaluates TD(lambda) and Exact_E under varying levels of action stochasticity.
# ==============================================================================

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON="python"
fi

# Configuration
N_SEEDS=5
ENVS=("FourRooms-misc" "MountainCar-v0")
SAMPLED_ALGOS=("td" "unbiased_sampled_E")
EPSILON_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

CONFIG='{"NUM_ENVS": 64, "NUM_STEPS": 256, "TOTAL_TIMESTEPS": 1000000, "MINIBATCH_SIZE": 1024, "NUM_EPOCHS": 1, "FAIL_PROB": 0.0}'

# The sweep_pipeline will test ALL of these LRs and Lambdas, automatically selecting 
# the best performing LR for EACH individual epsilon and lambda combination!
LR_GRID="0.001 0.0005 0.0001 0.00001"
LAMBDA_GRID="0.5 0.9 0.95"

mkdir -p slurm

echo "======================================================================"
echo "STARTING EPSILON NOISE SWEEP"
echo "Algorithms: ${SAMPLED_ALGOS[*]}"
echo "Epsilon Values: ${EPSILON_VALUES[*]}"
echo "LR Grid: $LR_GRID"
echo "Lambda Grid: $LAMBDA_GRID"
echo "======================================================================"

for env in "${ENVS[@]}"; do
    for epsilon in "${EPSILON_VALUES[@]}"; do
        echo ""
        echo "======================================================================"
        echo "Running Sweep: Environment=$env | Epsilon=$epsilon"
        echo "======================================================================"

        CMD="$PYTHON scripts/sweep_pipeline.py \
            --policy fixed \
            --env-name $env \
            --algos ${SAMPLED_ALGOS[*]} \
            --lr-grid $LR_GRID \
            --lambda-grid $LAMBDA_GRID \
            --n-seeds $N_SEEDS \
            --config '$CONFIG' \
            --rank-by 'auc' \
            --higher-is-better \
            --metric nn_greedy_performance \
            --use-geom-mean \
            --use-greedy-policy \
            --policy-epsilon $epsilon"
            
        echo "Command: $CMD"
        eval "$CMD"
    done
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo ""
echo "======================================================================"
echo "Epsilon Sweep Completed!"
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"
echo "======================================================================"
