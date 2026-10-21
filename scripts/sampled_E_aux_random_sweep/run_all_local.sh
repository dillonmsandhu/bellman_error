#!/usr/bin/env bash
# scripts/sampled_E_aux_random_sweep/run_all_local.sh
#
# Runs Sampled E-Aux Random Policy Evaluation Sweep across all 9 environments locally.
# Evaluates:
#   1. MC (Sampled)
#   2. OG Oct 8 (Biased)
#   3. Full v_aux (Oct 8)
#   4. Bilinear v_aux
#   5. Surgical v_aux
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
K_VALUES=${K_VALUES:-"8 64"}
CRITIC_LRS=${CRITIC_LRS:-"0.0005 0.001 0.003"}
NUM_ENVS=${NUM_ENVS:-16}
NUM_STEPS=${NUM_STEPS:-64}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-256}
NUM_EPOCHS=${NUM_EPOCHS:-4}
VAL_LAMBDA=${VAL_LAMBDA:-1.0}
OUT_DIR=${OUT_DIR:-"results/sampled_E_aux_random_sweep"}
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
        --k-values)
            K_VALUES="$2"
            shift 2
            ;;
        --critic-lrs)
            CRITIC_LRS="$2"
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
echo "Sampled E-Aux Random Policy Evaluation Sweep (All 9 Environments)"
echo "Python:             $PYTHON"
echo "Num Updates:        $NUM_UPDATES"
echo "Num Seeds:          $N_SEEDS (base: $SEED)"
echo "k Values:           $K_VALUES"
echo "Critic LRs:         $CRITIC_LRS"
echo "Rollout Config:     $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE | Epochs: $NUM_EPOCHS"
echo "Value Lambda:       $VAL_LAMBDA"
echo "Output Directory:   $OUT_DIR"
echo "Skip Completed:     $SKIP_COMPLETED"
echo "================================================================"

for env_name in "${ENVS[@]}"; do
    env_dir="$OUT_DIR/$env_name"
    out_pkl="$env_dir/out.pkl"
    summary_csv="$env_dir/summary_auc.csv"

    if [ "$SKIP_COMPLETED" -eq 1 ] && [ -f "$out_pkl" ] && [ -f "$summary_csv" ]; then
        echo ">>> [SKIP] Environment '$env_name' already completed at $env_dir"
        continue
    fi

    echo ""
    echo ">>> Running Environment: $env_name"
    $PYTHON scripts/sampled_E_aux_random_sweep/run_single_env.py \
        --env_name "$env_name" \
        --num_updates "$NUM_UPDATES" \
        --n_seeds "$N_SEEDS" \
        --seed "$SEED" \
        --k_values $K_VALUES \
        --critic_lrs $CRITIC_LRS \
        --num_envs "$NUM_ENVS" \
        --num_steps "$NUM_STEPS" \
        --minibatch_size "$MINIBATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --val_lambda "$VAL_LAMBDA" \
        --out_dir "$OUT_DIR"
done

echo ""
echo "================================================================"
echo "Generating Consolidated Publication Figures"
echo "================================================================"
$PYTHON scripts/sampled_E_aux_random_sweep/plot_results.py --results_dir "$OUT_DIR"

echo ""
echo "================================================================"
echo "Generating Publication Analysis Tables (ICML / NeurIPS)"
echo "================================================================"
$PYTHON scripts/sampled_E_aux_random_sweep/analyze_results.py --results_dir "$OUT_DIR"

echo ""
echo "All sampled E-aux random policy sweeps, plots, and tables completed successfully!"
