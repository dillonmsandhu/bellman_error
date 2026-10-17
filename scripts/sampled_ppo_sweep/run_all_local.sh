#!/usr/bin/env bash
# scripts/sampled_ppo_sweep/run_all_local.sh
#
# Runs full Sampled PPO Evaluation Sweep across all 9 environments sequentially on the local machine.
# Automatically updates consolidated publication plots and analysis tables upon completion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Resolve python interpreter
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

NUM_UPDATES=${NUM_UPDATES:-100}
N_SEEDS=${N_SEEDS:-10}
SEED=${SEED:-42}
K_VAL=${K_VAL:-32}
ACTOR_LR=${ACTOR_LR:-0.0003}
CRITIC_LRS=${CRITIC_LRS:-"0.0005 0.001 0.003"}
NUM_ENVS=${NUM_ENVS:-16}
NUM_STEPS=${NUM_STEPS:-64}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-256}
NUM_EPOCHS=${NUM_EPOCHS:-4}
GAE_LAMBDA=${GAE_LAMBDA:-0.8}
RETURN_LAMBDAS=${RETURN_LAMBDAS:-"0.95 1.0"}
CLIP_EPS=${CLIP_EPS:-0.2}
OUT_DIR=${OUT_DIR:-"results/sampled_ppo_sweep"}
SKIP_COMPLETED=${SKIP_COMPLETED:-0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-completed)
            SKIP_COMPLETED=1
            shift
            ;;
        --force)
            SKIP_COMPLETED=0
            shift
            ;;
        --num-updates)
            NUM_UPDATES="$2"
            shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --k)
            K_VAL="$2"
            shift 2
            ;;
        --actor-lr)
            ACTOR_LR="$2"
            shift 2
            ;;
        --critic-lrs)
            CRITIC_LRS="$2"
            shift 2
            ;;
        --return-lambdas)
            RETURN_LAMBDAS="$2"
            shift 2
            ;;
        --gae-lambda)
            GAE_LAMBDA="$2"
            shift 2
            ;;
        --clip-eps)
            CLIP_EPS="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ENVS=(
    "mountaincar"
    "FourRooms-misc"
    "fourrooms-mines"
    "fourrooms-mines-dense"
    "eightrooms-dense"
    "eightrooms-misc"
    "whirlpool-misc"
    "SpaceInvadersExactValue"
    "mountaincar-dense"
)

echo "================================================================="
echo "Full Sampled PPO Local Sequential Sweep"
echo "Num Updates:      $NUM_UPDATES"
echo "Num Seeds:        $N_SEEDS"
echo "Critic Feature k: $K_VAL"
echo "Actor LR:         $ACTOR_LR"
echo "Critic LRs:       $CRITIC_LRS"
echo "GAE Lambda:       $GAE_LAMBDA"
echo "Clip Epsilon:     $CLIP_EPS"
echo "Rollout Config:   $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE | Epochs: $NUM_EPOCHS"
echo "Output Directory: $OUT_DIR"
echo "Skip Completed:   $SKIP_COMPLETED"
echo "Python:           $PYTHON"
echo "================================================================="

mkdir -p "$OUT_DIR"

for idx in "${!ENVS[@]}"; do
    env_name="${ENVS[$idx]}"
    env_out="$OUT_DIR/$env_name"
    summary_file="$env_out/summary.csv"

    if [[ "$SKIP_COMPLETED" -eq 1 && -f "$summary_file" ]]; then
        echo ""
        echo ">>> [$(($idx + 1))/${#ENVS[@]}] Skipping $env_name (already completed: $summary_file)"
        continue
    fi

    echo ""
    echo ">>> [$(($idx + 1))/${#ENVS[@]}] Running $env_name"
    $PYTHON scripts/sampled_ppo_sweep/run_single_env.py \
        --env_name "$env_name" \
        --num_updates "$NUM_UPDATES" \
        --n_seeds "$N_SEEDS" \
        --seed "$SEED" \
        --k "$K_VAL" \
        --actor_lr "$ACTOR_LR" \
        --critic_lrs $CRITIC_LRS \
        --return_lambdas $RETURN_LAMBDAS \
        --num_envs "$NUM_ENVS" \
        --num_steps "$NUM_STEPS" \
        --minibatch_size "$MINIBATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --gae_lambda "$GAE_LAMBDA" \
        --clip_eps "$CLIP_EPS" \
        --out_dir "$OUT_DIR"
done

echo ""
echo "================================================================="
echo "Post-Processing: Generating Publication Figures & Tables"
echo "================================================================="

$PYTHON scripts/sampled_ppo_sweep/plot_results.py --results_dir "$OUT_DIR"
$PYTHON scripts/sampled_ppo_sweep/analyze_results.py --results_dir "$OUT_DIR"

echo ""
echo "================================================================="
echo "Full Sampled PPO Evaluation Sweep Pipeline Finished Successfully!"
echo "Artifacts are ready under: $OUT_DIR"
echo "================================================================="
