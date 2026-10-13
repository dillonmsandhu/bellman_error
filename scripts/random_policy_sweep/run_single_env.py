"""
scripts/random_policy_sweep/run_single_env.py

Runs random policy evaluation benchmark comparing:
1. Sampled MC (TD with GAE_LAMBDA=1.0)
2. TD(0) (SARSA-style TD(0) value learning)
3. TD(lambda) (TD with GAE_LAMBDA=0.8 or user specified)
4. E(0) (October 8 start-anchored formulation)

Tracks metrics during learning:
- VE: nn_weighted_VE
- Advantage Cosine Sim: nn_advantage_cossim
- Greedy Policy V_start: nn_greedy_performance
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
from random_policy.td import make_train as make_td
from random_policy.td0 import make_train as make_td0
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Random Policy Benchmark (MC vs TD(0) vs TD(lambda) vs E(0)) on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--num_updates", type=int, default=50, help="Number of critic updates")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel rollout environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout steps per environment")
    parser.add_argument("--minibatch_size", type=int, default=1024, help="Minibatch size for updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--td_lambda", type=float, default=0.8, help="Lambda value for TD(lambda)")
    parser.add_argument("--out_dir", type=str, default="results/random_policy_sweep", help="Output directory root")
    return parser.parse_args()


def run_variant(variant_name: str, make_train_fn, env_config: dict, n_seeds: int, base_seed: int):
    """Executes a single algorithm variant across multiple seeds and aggregates metrics."""
    train_fn = make_train_fn(env_config)
    seed_metrics = []

    for s_idx in range(n_seeds):
        rng = jax.random.PRNGKey(base_seed + s_idx)
        t0 = time.time()
        res = train_fn(rng)
        dt = time.time() - t0
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
    print("Random Policy Benchmark: MC vs TD(0) vs TD(lambda) vs E(0)")
    print(f"Environment:  {env_name}")
    print(f"Num Updates:  {args.num_updates}")
    print(f"Num Seeds:    {args.n_seeds}")
    print(f"TD Lambda:    {args.td_lambda}")
    print(f"Rollout:      {args.num_envs} envs x {args.num_steps} steps")
    print(f"Output:       {env_out_dir}")
    print("=" * 70)

    cfg_base = default_config.copy()
    cfg_base["ENV_NAME"] = env_name
    cfg_base["NUM_UPDATES"] = args.num_updates
    cfg_base["NUM_ENVS"] = args.num_envs
    cfg_base["NUM_STEPS"] = args.num_steps
    cfg_base["MINIBATCH_SIZE"] = args.minibatch_size
    cfg_base["TOTAL_TIMESTEPS"] = args.num_updates * args.num_envs * args.num_steps
    cfg_base["GAMMA"] = args.gamma
    cfg_base["LR"] = args.lr

    # Define the 4 algorithm variants
    # 1. Sampled MC: td.py with GAE_LAMBDA=1.0
    cfg_mc = cfg_base.copy()
    cfg_mc["GAE_LAMBDA"] = 1.0

    # 2. TD(lambda): td.py with specified lambda
    cfg_td_lambda = cfg_base.copy()
    cfg_td_lambda["GAE_LAMBDA"] = args.td_lambda

    # 3. TD(0): td0.py
    cfg_td0 = cfg_base.copy()

    # 4. E(0): oct8.py
    cfg_oct8 = cfg_base.copy()

    variants = [
        ("mc", "Sampled MC (λ=1.0)", make_td, cfg_mc),
        ("td0", "TD(0)", make_td0, cfg_td0),
        ("td_lambda", f"TD(λ={args.td_lambda})", make_td, cfg_td_lambda),
        ("oct8", "E(0) (Oct 8)", make_oct8, cfg_oct8),
    ]

    results = {}
    for v_key, v_label, make_fn, cfg_v in variants:
        print(f"\n--> Running Variant: {v_label} ({v_key})")
        results[v_key] = run_variant(
            variant_name=v_key,
            make_train_fn=make_fn,
            env_config=cfg_v,
            n_seeds=args.n_seeds,
            base_seed=args.seed,
        )

    out_file = os.path.join(env_out_dir, "out.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved all algorithm metrics to: {out_file}")


if __name__ == "__main__":
    main()
