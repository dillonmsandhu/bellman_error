#!/bin/bash
#SBATCH --job-name=is_ppo_seeds
#SBATCH --output=slurm/%j.out
#SBATCH --time=1:00:00
#SBATCH --partition compsci-gpu
#SBATCH --gres=gpu:a5000:1 

# Run like:
# sbatch run_slurm.sh ppo.ppo_lstd mc
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # start timer
FILE=$1
SUFFIX=$2

# Define batch sizes (Bash arrays use space separation)
BATCH_SIZES=(1 2 4 8 16 32 64 128)

# Base configuration template
BASE_CONFIG='{"TOTAL_TIMESTEPS": 1000, "NUM_ENVS": 1, "NUM_STEPS": 1, "NUM_EPOCHS": 1, "MINIBATCH_SIZE": 1, "ENV_NAME": "FourRooms-misc", "MODEL_LOAD_DIR": "250_steps_layer_norm", "FAIL_PROB": 0.15}'

for bs in "${BATCH_SIZES[@]}"; do
    echo "========================================"
    echo "Running sweep for Batch Size: $bs"
    echo "========================================"

    # Dynamically update the batch size parameter in the JSON config using jq
    # Note: Change '.BATCH_SIZE' to '.MINIBATCH_SIZE' if that is the key your script expects.
    CONFIG=$(echo "$BASE_CONFIG" | jq --argjson bs "$bs" '.BATCH_SIZE = $bs')

    # Update suffix to include the current batch size
    CURRENT_SUFFIX="${SUFFIX}_sweep_b_${bs}"

    CMD="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python -m ${FILE} --run-suffix ${CURRENT_SUFFIX} --config '${CONFIG}' --save-metrics"
    
    echo "$CMD"
    eval "$CMD"
    echo ""
done

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$SECONDS
echo "Total runtime: $(($DURATION / 3600))h $((($DURATION % 3600) / 60))m $(($DURATION % 60))s"`
echo "Job finished at: $END_TIME"