"""
scripts/exact_fixed_policy_sweep/generate_fixed_policies.py

Step 0: Pre-generates and saves fixed target policies for all environments
using a modest number of exact PPO updates (ppo/ground_truth.py).

The generated policies represent a non-random, directed policy
(inducing an irreversible Markov chain with non-zero skew-symmetric component K).

Storage destination:
  results/ppo/ground_truth/fixed_policies/<env_name>/out.pkl
  results/ppo/ground_truth/fixed_policies/<env_name>/config.json

Automatically skips environments where a valid checkpoint is already detected,
unless --force is specified.
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import json
import cloudpickle
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from core.config import config as base_cfg
from ppo.ground_truth import make_train as make_ground_truth_train

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

DEFAULT_POLICY_DIR = "results/ppo/ground_truth/fixed_policies"


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-generate fixed policies for exact evaluation using ppo/ground_truth.py")
    parser.add_argument("--envs", type=str, nargs="+", default=ALL_ENVS, help="Environments to generate policies for")
    parser.add_argument("--num_updates", type=int, default=50, help="Number of ground truth PPO updates to reach a directed policy (default: 50)")
    parser.add_argument("--policy_dir", type=str, default=DEFAULT_POLICY_DIR, help="Directory to save fixed policy models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for policy training")
    parser.add_argument("--actor_lr", type=float, default=0.001, help="Actor learning rate for ground truth PPO")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if checkpoint exists")
    return parser.parse_args()


def is_checkpoint_valid(env_dir: str) -> bool:
    """Checks if out.pkl exists and contains valid runner_state train state."""
    pkl_path = os.path.join(env_dir, "out.pkl")
    cfg_path = os.path.join(env_dir, "config.json")
    if not (os.path.exists(pkl_path) and os.path.exists(cfg_path)):
        return False
    try:
        with open(pkl_path, "rb") as f:
            data = cloudpickle.load(f)
        return isinstance(data, dict) and "runner_state" in data
    except Exception:
        return False


def generate_policy_for_env(env_name: str, args: argparse.Namespace) -> bool:
    """Trains a directed policy on a single environment using exact ground truth PPO and saves the checkpoint."""
    env_dir = os.path.join(os.path.abspath(args.policy_dir), env_name)
    os.makedirs(env_dir, exist_ok=True)

    if not args.force and is_checkpoint_valid(env_dir):
        print(f"[Cached] Valid fixed policy already exists for '{env_name}' at: {env_dir}")
        return True

    print(f"\n--> Training Fixed Policy for '{env_name}' ({args.num_updates} exact PPO updates)...")
    is_mc = "mountaincar" in env_name.lower()

    cfg = base_cfg.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": 1,
        "NUM_ENVS": 1,
        "MINIBATCH_SIZE": 1,
        "NUM_EPOCHS": 1,
        "TOTAL_TIMESTEPS": args.num_updates,
        "NUM_UPDATES": args.num_updates,
        "N_SEEDS": 1,
        "SEED": args.seed,
        "ACTOR_LR": args.actor_lr,
        "ACTOR_LR_END": args.actor_lr,
        "LR": args.actor_lr,
        "LR_END": args.actor_lr,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "GAE_LAMBDA": 1.0,
    })

    t0 = time.time()
    # Use vmap over 1 seed so parameters have shape (1, ...) expected by helpers.get_evaluation_policies
    train_builder = make_ground_truth_train(cfg)
    vmapped_train = jax.vmap(train_builder)
    rng = jax.random.PRNGKey(args.seed)
    rngs = jax.random.split(rng, 1)

    out = vmapped_train(rngs)
    dt = time.time() - t0

    # Extract metrics summary
    metrics = out.get("metrics", {})
    v_start = float(np.mean(metrics.get("V_start", [0.0])))
    v_mean = float(np.mean(metrics.get("V_mean", [0.0])))

    # Save checkpoint out.pkl and config.json
    pkl_path = os.path.join(env_dir, "out.pkl")
    cfg_path = os.path.join(env_dir, "config.json")

    with open(pkl_path, "wb") as f:
        cloudpickle.dump(out, f)

    # Save serializable config
    serializable_cfg = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list, dict)) or v is None}
    with open(cfg_path, "w") as f:
        json.dump(serializable_cfg, f, indent=4)

    print(f"[Saved] Fixed policy checkpoint for '{env_name}' in {dt:.1f}s -> {pkl_path}")
    print(f"        V_start: {v_start:.4f} | V_mean: {v_mean:.4f}")
    return True


def main():
    args = parse_args()
    policy_dir = os.path.abspath(args.policy_dir)
    os.makedirs(policy_dir, exist_ok=True)

    print("=" * 75)
    print("Fixed Policy Generator (Step 0)")
    print(f"Target Environments: {args.envs}")
    print(f"PPO Updates:         {args.num_updates}")
    print(f"Output Directory:    {policy_dir}")
    print(f"Force Re-generate:   {args.force}")
    print("=" * 75)

    all_success = True
    for env_name in args.envs:
        success = generate_policy_for_env(env_name, args)
        if not success:
            all_success = False

    if all_success:
        print("\n[Complete] All target fixed policies are verified and ready for evaluation.")
    else:
        print("\n[Warning] Some environment policies failed to generate.")


if __name__ == "__main__":
    main()
