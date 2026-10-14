#!/bin/bash
# scripts/exact_ppo_sweep/run_all_local.sh
# Runs the Exact PPO Sweep locally across all 9 environments.

set -e

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Resolve Python interpreter
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
ACTOR_LR=${ACTOR_LR:-0.0001}
CRITIC_LR=${CRITIC_LR:-0.001}
GAE_LAMBDA=${GAE_LAMBDA:-0.2}
CLIP_EPS=${CLIP_EPS:-0.2}
VAL_LAMBDA=${VAL_LAMBDA:-0.8}
OUT_DIR=${OUT_DIR:-"results/exact_ppo_sweep"}
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
        --seed)
            SEED="$2"
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
        --critic-lr)
            CRITIC_LR="$2"
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
        --val-lambda)
            VAL_LAMBDA="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

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

mkdir -p "$OUT_DIR"

echo "================================================================"
echo "Starting Local Exact PPO Evaluation Sweep"
echo "Python:             $PYTHON"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS"
echo "Base Seed:          $SEED"
echo "Critic k:           $K_VAL"
echo "Actor LR:           $ACTOR_LR"
echo "Critic LR:          $CRITIC_LR"
echo "GAE Lambda:         $GAE_LAMBDA"
echo "Clip Epsilon:       $CLIP_EPS"
echo "Value Lambda:       $VAL_LAMBDA"
echo "Output Directory:   $OUT_DIR"
echo "Skip Completed:     $SKIP_COMPLETED"
echo "Environments (9):   ${ENVS[*]}"
echo "================================================================"

for i in "${!ENVS[@]}"; do
    env_name="${ENVS[$i]}"
    env_out_dir="$OUT_DIR/$env_name"
    pkl_path="$env_out_dir/out.pkl"
    csv_path="$env_out_dir/summary.csv"

    echo ""
    echo "----------------------------------------------------------------"
    echo "[$((i + 1))/${#ENVS[@]}] Environment: $env_name"
    echo "----------------------------------------------------------------"

    if [ "$SKIP_COMPLETED" -eq 1 ] && [ -f "$pkl_path" ] && [ -f "$csv_path" ]; then
        echo "  [Skipping] Valid results already exist at: $env_out_dir"
        continue
    fi

    $PYTHON scripts/exact_ppo_sweep/run_single_env.py \
        --env_name "$env_name" \
        --num_updates "$NUM_UPDATES" \
        --n_seeds "$N_SEEDS" \
        --seed "$SEED" \
        --k "$K_VAL" \
        --actor_lr "$ACTOR_LR" \
        --critic_lr "$CRITIC_LR" \
        --gae_lambda "$GAE_LAMBDA" \
        --clip_eps "$CLIP_EPS" \
        --val_lambda "$VAL_LAMBDA" \
        --out_dir "$OUT_DIR"
done

echo ""
echo "================================================================"
echo "All 9 environments completed. Generating plots and tables..."
echo "================================================================"

$PYTHON scripts/exact_ppo_sweep/plot_results.py --results_dir "$OUT_DIR"
$PYTHON scripts/exact_ppo_sweep/analyze_results.py --results_dir "$OUT_DIR"

echo "Local Exact PPO Evaluation Sweep fully complete!"
