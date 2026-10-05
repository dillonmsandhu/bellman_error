#!/bin/bash
#SBATCH --job-name=horizon_suite
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Ensure working directory is repo root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

# Detect Python interpreter
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
else
    PYTHON=$(which python)
fi

echo "================================================================================"
echo "Job ID:           $SLURM_JOB_ID"
echo "Host:             $(hostname)"
echo "Directory:        $(pwd)"
echo "Python:           $PYTHON"
echo "Start Time:       $(date)"
echo "================================================================================"

mkdir -p slurm
mkdir -p results/full_horizon_suite_multiseed

# Run multi-seed fixed-batch horizon investigation across all environments
$PYTHON scripts/variance_investigation/run_cross_env_horizon_suite.py \
    --envs fourrooms-dense FourRooms-misc eightrooms-dense eightrooms-misc whirlpool-misc SpaceInvadersExactValue mountaincar-dense \
    --total_batch_size 16384 \
    --horizons 8 64 128 512 1024 \
    --num_updates 35 \
    --num_seeds 4 \
    --seed 42 \
    --out_dir results/full_horizon_suite_multiseed

echo "================================================================================"
echo "End Time:         $(date)"
echo "Completed Successfully!"
echo "================================================================================"
