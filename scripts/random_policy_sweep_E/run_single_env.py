"""
scripts/random_policy_sweep/run_single_env.py

Runs the 3 Sampled E variants (Sept 14, Oct 5, Oct 8) on a single environment for Random Policy Evaluation:
- Sept 14 (Uncorrected): Pure Dirichlet smoothness + magnitude
- Oct 5 (Corrected): Adds empirical difference of second moments
- Oct 8 (Start-Anchored): Adds boundary anchor on initial state distribution

Tracks and compares:
- Value Error MSE (nn_weighted_VE) across updates
- Gradient MSE to Exact E gradient (grad_mse_exact_E) across updates
- Gradient cosine similarity to Exact E (grad_cos_exact_E)
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import pickle
from typing import Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from core.config import config as default_config
from random_policy.sep14 import make_train as make_sep14
from random_policy.oct5 import make_train as make_oct5
from random_policy.oct8 import make_train as make_oct8

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

VARIANTS = {
    "sep14": ("Sept 14 (Uncorr)", make_sep14),
    "oct5": ("Oct 5 (Corr)", make_oct5),
    "oct8": ("Oct 8 (Start-Anc)", make_oct8),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Random Policy Sampled E Evaluation on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--num_updates", type=int, default=50, help="Number of critic updates")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel rollout environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout steps per environment")
    parser.add_argument("--minibatch_size", type=int, default=1024, help="Minibatch size for updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="results/random_policy_sweep_E", help="Output directory root")
    return parser.parse_args()


def run_variant(variant_name, make_train_fn, env_config, n_seeds, base_seed):
    """Executes a single variant across multiple seeds and aggregates metrics."""
    train_fn = make_train_fn(env_config)
    seed_metrics = []

    for s_idx in range(n_seeds):
        rng = jax.random.PRNGKey(base_seed + s_idx)
        t0 = time.time()
        res = train_fn(rng)
        dt = time.time() - t0
        # Convert JAX arrays to NumPy for saving
        np_metrics = jax.tree.map(lambda x: np.asarray(x), res["metrics"])
        seed_metrics.append(np_metrics)
        print(f"    Seed {s_idx + 1}/{n_seeds} completed in {dt:.2f}s")

    # Stack metrics across seeds: (n_seeds, num_updates, ...)
    keys = seed_metrics[0].keys()
    stacked = {}
    for k in keys:
        try:
            stacked[k] = np.stack([sm[k] for sm in seed_metrics], axis=0)
        except Exception:
            pass

    return stacked


def main():
    args = parse_args()
    env_name = args.env_name
    out_dir = os.path.abspath(args.out_dir)
    env_out_dir = os.path.join(out_dir, env_name)
    os.makedirs(env_out_dir, exist_ok=True)

    print("=" * 70)
    print("Random Policy Sampled E Benchmark")
    print(f"Environment:  {env_name}")
    print(f"Num Updates:  {args.num_updates}")
    print(f"Num Seeds:    {args.n_seeds}")
    print(f"Rollout:      {args.num_envs} envs x {args.num_steps} steps")
    print(f"Output:       {env_out_dir}")
    print("=" * 70)

    cfg = default_config.copy()
    cfg["ENV_NAME"] = env_name
    cfg["NUM_UPDATES"] = args.num_updates
    cfg["NUM_ENVS"] = args.num_envs
    cfg["NUM_STEPS"] = args.num_steps
    cfg["MINIBATCH_SIZE"] = args.minibatch_size
    cfg["TOTAL_TIMESTEPS"] = args.num_updates * args.num_envs * args.num_steps
    cfg["GAMMA"] = args.gamma
    cfg["LR"] = args.lr

    results = {}
    for v_key, (v_label, make_fn) in VARIANTS.items():
        print(f"\n--> Running Variant: {v_label} ({v_key})")
        results[v_key] = run_variant(
            variant_name=v_key,
            make_train_fn=make_fn,
            env_config=cfg,
            n_seeds=args.n_seeds,
            base_seed=args.seed,
        )

    out_file = os.path.join(env_out_dir, "out.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved all variant metrics to: {out_file}")


if __name__ == "__main__":
    main()
