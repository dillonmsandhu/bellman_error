"""
scripts/sampling_E_test/run_single_env.py

Runs ppo/exact_E.py on a single environment with sampling E metrics logging.
At every policy update, computes:
- Exact E matrix loss and critic gradient
- Expected corrected E loss and gradient
- Expected uncorrected E loss and gradient
- Sampled corrected E loss and gradient (128 rollouts of 128 steps)
- Sampled uncorrected E loss and gradient
- Cosine similarities and gradient norms

Integrates with core/evaluate.py for standardized metrics saving and plotting.
"""

from __future__ import annotations
import os
import sys
import argparse
import time
from typing import Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
from core.config import config as default_config
from core.evaluate import evaluate
from ppo.exact_E import make_train

ALL_ENVS = [
    "fourrooms-dense",
    "FourRooms-misc",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sampling E Test on a Single Environment")
    parser.add_argument("--env_name", type=str, default=None, help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index of the environment (0-6)")
    parser.add_argument("--num_updates", type=int, default=50, help="Number of policy updates")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num_trajectories", type=int, default=128, help="Number of parallel rollout trajectories")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout steps per trajectory")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="results/sampling_E_test", help="Output directory root")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save complete train state checkpoints")
    parser.add_argument("--save_metrics", action="store_true", default=True, help="Save metrics dictionary")
    parser.add_argument("--save_video", action="store_true", help="Save policy GIF video")
    return parser.parse_args()


def build_config(env_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    is_mountaincar = "mountaincar" in env_name.lower()

    cfg = default_config.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "TOTAL_TIMESTEPS": args.num_updates,
        "NUM_UPDATES": args.num_updates,
        "NUM_ENVS": 1,
        "NUM_STEPS": 1,
        "N_SEEDS": args.n_seeds,
        "SEED": args.seed,
        "NETWORK_TYPE": "mlp" if is_mountaincar else "cnn",
        "USE_VISUAL_OBS": False if is_mountaincar else True,
        "LR": args.lr,
        "NUM_EPOCHS": 4,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": 0.6,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
        "LOG_SAMPLING_E_METRICS": True,
        "SAMPLING_E_NUM_TRAJECTORIES": args.num_trajectories,
        "SAMPLING_E_NUM_STEPS": args.num_steps,
    })
    return cfg


def main():
    args = parse_args()

    # Determine environment name
    if args.env_name is not None:
        env_name = args.env_name
    elif args.env_idx is not None and 0 <= args.env_idx < len(ALL_ENVS):
        env_name = ALL_ENVS[args.env_idx]
    else:
        env_name = "eightrooms-misc"

    print("=" * 70)
    print("Sampling E Test: Exact vs Expected vs Sampled E")
    print(f"Environment:       {env_name}")
    print(f"Updates:           {args.num_updates}")
    print(f"Seeds:             {args.n_seeds} (base seed {args.seed})")
    print(f"Rollout Sampling:  {args.num_trajectories} trajectories x {args.num_steps} steps")
    print(f"Gamma:             {args.gamma}")
    print(f"Output Root:       {args.out_dir}")
    print("=" * 70)

    run_config = build_config(env_name, args)
    rng = jax.random.PRNGKey(args.seed)

    start_time = time.time()
    evaluate(
        run_config=run_config,
        make_train=make_train,
        run_dir=args.out_dir,
        args=args,
        rng=rng,
    )
    elapsed = time.time() - start_time
    print(f"\n[DONE] Completed {env_name} in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
