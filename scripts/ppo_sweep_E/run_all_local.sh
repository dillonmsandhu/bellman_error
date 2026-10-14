#!/bin/bash
# scripts/ppo_sweep_E/run_all_local.sh
#
# Sequentially executes all 10 environments locally for PPO Sampled E Critic Benchmark,
# then aggregates and plots the comparative results.

set -e

REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

ENVS=(
    "FourRooms-misc"
    "fourrooms-dense"
    "fourrooms-mines"
    "fourrooms-mines-dense"
    "eightrooms-misc"
    "eightrooms-dense"
    "whirlpool-misc"
    "MountainCar-v0"
    "mountaincar-dense"
    "SpaceInvadersExactValue"
)

OUT_DIR=${OUT_DIR:-"results/ppo_sweep_E"}
NUM_UPDATES=${NUM_UPDATES:-100}
N_SEEDS=${N_SEEDS:-3}
NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-128}

echo "=========================================================="
echo "Starting Sequential Local Run of PPO Sampled E Benchmark"
echo "Total Environments: ${#ENVS[@]}"
echo "Updates per Env:    $NUM_UPDATES"
echo "Seeds per Env:      $N_SEEDS"
echo "Rollout:            ${NUM_ENVS} envs x ${NUM_STEPS} steps"
echo "Output Directory:   $OUT_DIR"
echo "=========================================================="

for idx in "${!ENVS[@]}"; do
    env_name="${ENVS[$idx]}"
    echo ""
    echo "----------------------------------------------------------"
    echo "[$((idx + 1))/${#ENVS[@]}] Processing: $env_name"
    echo "----------------------------------------------------------"
    
    $PYTHON scripts/ppo_sweep_E/run_single_env.py \
        --env_name "$env_name" \
        --num_updates "$NUM_UPDATES" \
        --n_seeds "$N_SEEDS" \
        --num_envs "$NUM_ENVS" \
        --num_steps "$NUM_STEPS" \
        --out_dir "$OUT_DIR"
done

echo ""
echo "=========================================================="
echo "All environments completed. Generating cross-env plots..."
echo "=========================================================="

$PYTHON scripts/ppo_sweep_E/plot_results.py --results_dir "$OUT_DIR"

echo "Done! All results and plots generated in $OUT_DIR."
