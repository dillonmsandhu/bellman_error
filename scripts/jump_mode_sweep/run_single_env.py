"""
scripts/jump_mode_sweep/run_single_env.py

Executes PPO comparing 3 formulations for a fixed lambda (0.9) across 5 seeds:
1. e_lambda_drop: Geometric jump E(lambda=0.9) with drop_invalid_jump (actor uses all data, critic drops invalid jumps)
2. e_lambda_mc:   Geometric jump E(lambda=0.9) with mc_invalid_jump (actor & critic use all data; invalid jumps zero out Dirichlet smoothing and keep regular MC weight)
3. bsmc:          1-Step Bellman-Smoothed Monte Carlo with equivalent calibrated weight for lambda=0.9

Primary metric: True ground truth value of the start state V^pi(s_0) [V_start] across 5 random seeds.
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import pickle
from typing import Dict, Any, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from core.config import config as base_cfg
import core.helpers as helpers
from ppo.E_lambda_geometric import make_train as make_e_lambda_train
from ppo.sampled_bsmc import make_train as make_bsmc_train

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

ALGORITHMS = {
    "e_lambda_drop": {
        "display_name": "E(λ=0.95) [Drop Invalid Jump]",
        "builder": make_e_lambda_train,
        "config_overrides": {
            "E_LAMBDA_VARIANT": "aux",
            "E_LAMBDA_JUMP_MODE": "drop_invalid_jump",
            "RETURN_LAMBDA": 1.0,
        },
        "color": "#9467bd",  # Purple
    },
    "e_lambda_mc": {
        "display_name": "E(λ=0.95) [MC Invalid Jump]",
        "builder": make_e_lambda_train,
        "config_overrides": {
            "E_LAMBDA_VARIANT": "aux",
            "E_LAMBDA_JUMP_MODE": "mc_invalid_jump",
            "RETURN_LAMBDA": 1.0,
        },
        "color": "#d62728",  # Red / Coral
    },
    "e_lambda_full_mc": {
        "display_name": "E(λ=0.95) [Full MC Invalid Jump]",
        "builder": make_e_lambda_train,
        "config_overrides": {
            "E_LAMBDA_VARIANT": "aux",
            "E_LAMBDA_JUMP_MODE": "full_mc_invalid_jump",
            "RETURN_LAMBDA": 1.0,
        },
        "color": "#ff7f0e",  # Orange
    },
    "bsmc": {
        "display_name": "BSMC [1-step, Calibrated λ=0.95]",
        "builder": make_bsmc_train,
        "config_overrides": {
            "BSMC_VARIANT": "aux",
            "RETURN_LAMBDA": 1.0,
        },
        "color": "#1f77b4",  # Blue
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Jump Modes vs BSMC Comparison on a Single Environment")
    parser.add_argument("--env_name", type=str, default=None, help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index of the environment (0-8)")
    parser.add_argument("--algo", type=str, default=None, choices=list(ALGORITHMS.keys()), help="Algorithm to run (default: runs all algorithms)")
    parser.add_argument("--value_lambda", type=float, default=0.95, help="Trace parameter lambda / equivalent BSMC weight parameter (default: 0.95)")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total environment timesteps per seed (default: 100,000)")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--num_steps", type=int, default=64, help="Rollout steps per env step (default: 64)")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--minibatch_size", type=int, default=64, help="Minibatch size (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=4, help="PPO update epochs per rollout (default: 4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-3 for MountainCar, 3e-4 for others)")
    parser.add_argument("--out_dir", type=str, default="results/jump_mode_sweep", help="Output directory root")
    return parser.parse_args()


def build_env_config(env_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    is_mc = "mountaincar" in env_name.lower()
    lr = args.lr if args.lr is not None else (0.001 if is_mc else 0.0003)

    cfg = base_cfg.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_STEPS": args.num_steps,
        "NUM_ENVS": args.num_envs,
        "MINIBATCH_SIZE": args.minibatch_size,
        "NUM_EPOCHS": args.num_epochs,
        "GAMMA": args.gamma,
        "VALUE_LAMBDA": args.value_lambda,
        "RETURN_LAMBDA": 1.0,
        "GAE_LAMBDA": 0.8,
        "CRITIC_LR": lr,
        "ACTOR_LR": lr,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "RUNWAY_LENGTH": None,  # Adaptive runway
    })
    return cfg


def run_algorithm(algo_key: str, env_name: str, args: argparse.Namespace, base_env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    algo_meta = ALGORITHMS[algo_key]
    cfg = base_env_cfg.copy()
    cfg.update(algo_meta["config_overrides"])

    print(f"\n--- Running {algo_meta['display_name']} on {env_name} ({args.n_seeds} seeds) ---")
    t0 = time.time()
    train_builder = algo_meta["builder"]
    train_fn = train_builder(cfg)
    vmapped_train = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))

    rngs = jax.random.split(jax.random.PRNGKey(args.seed), args.n_seeds)
    out = vmapped_train(rngs, None)
    # Ensure computation completes
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - t0
    print(f"[{algo_key}] Completed in {elapsed:.2f}s")

    metrics = out["metrics"]
    res = {
        "algo": algo_key,
        "display_name": algo_meta["display_name"],
        "color": algo_meta["color"],
        "env_name": env_name,
        "value_lambda": args.value_lambda,
        "n_seeds": args.n_seeds,
        "elapsed_seconds": elapsed,
        # Performance metrics
        "V_start": np.array(metrics.get("V_start", 0.0)),
        "returns": np.array(metrics.get("returned_episode_returns", 0.0)),
        "value_loss": np.array(metrics.get("value_loss", 0.0)),
        "policy_tv": np.array(metrics.get("policy_tv", 0.0)) if "policy_tv" in metrics else None,
    }
    return res


def main():
    args = parse_args()

    if args.env_name is not None:
        env_name = args.env_name
    elif args.env_idx is not None and 0 <= args.env_idx < len(ALL_ENVS):
        env_name = ALL_ENVS[args.env_idx]
    else:
        env_name = "FourRooms-misc"

    algos_to_run = [args.algo] if args.algo is not None else list(ALGORITHMS.keys())
    base_env_cfg = build_env_config(env_name, args)

    print("=" * 80)
    print(f"RUNWAY JUMP MODE & BSMC PPO COMPARISON: {env_name}")
    print(f"Fixed Lambda:      {args.value_lambda}")
    print(f"Seeds:             {args.n_seeds} (base seed {args.seed})")
    print(f"Total Timesteps:   {args.total_timesteps}")
    print(f"Algorithms:        {algos_to_run}")
    print("=" * 80)

    env_out_dir = os.path.join(args.out_dir, f"lambda_{args.value_lambda}", env_name)
    os.makedirs(env_out_dir, exist_ok=True)

    for algo_key in algos_to_run:
        result = run_algorithm(algo_key, env_name, args, base_env_cfg)
        save_path = os.path.join(env_out_dir, f"{algo_key}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved: {save_path}")

        # Summary statistics on start-state value V_start
        v_arr = result["V_start"]
        final_v = v_arr[:, -1]
        mean_v = float(np.mean(final_v))
        sem_v = float(np.std(final_v) / np.sqrt(len(final_v)))
        auc_v = float(np.mean(np.mean(v_arr, axis=1)))
        print(f"  -> Final V^pi(s0): {mean_v:.4f} +/- {sem_v:.4f} | Mean AUC: {auc_v:.4f}")

    print(f"\n[DONE] Finished {env_name} for all algorithms.")


if __name__ == "__main__":
    main()
