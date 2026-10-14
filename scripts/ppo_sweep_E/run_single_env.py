"""
scripts/ppo_sweep_E/run_single_env.py

Runs the 3 Sampled E critic formulations (Sept 14, Oct 5, Oct 8) for PPO
on a single environment across multiple random seeds:
- Sept 14 (Uncorrected): Pure Dirichlet smoothness + magnitude
- Oct 5 (Corrected): Adds empirical difference of second moments
- Oct 8 (Start-Anchored): Adds boundary anchor on initial state distribution
  with Adam-style bias-corrected nu estimate

Tracks and saves:
- returned_episode_returns (evaluation return during PPO learning)
- V_start (true policy value from start state)
- nn_weighted_VE (weighted value error MSE)
- E (exact matrix Bellman error E)
- value_loss, magnitude_loss, laplacian_loss, actor_loss, entropy
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import config as default_config
from ppo.sep14 import make_train as make_sep14
from ppo.oct5 import make_train as make_oct5
from ppo.oct8 import make_train as make_oct8

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ALL_ENVS = [
    "FourRooms-misc",
    "fourrooms-dense",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-misc",
    "eightrooms-dense",
    "whirlpool-misc",
    "MountainCar-v0",
    "mountaincar-dense",
    "SpaceInvadersExactValue",
]

VARIANTS = {
    "sep14": ("Sept 14 (Uncorr)", make_sep14, "#1f77b4"),
    "oct5": ("Oct 5 (Corr)", make_oct5, "#d62728"),
    "oct8": ("Oct 8 (Start-Anc)", make_oct8, "#2ca02c"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO Sampled E Critic Benchmark on a Single Env")
    parser.add_argument("--env_name", type=str, default=None, help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-9)")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of PPO updates")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel rollout environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout steps per environment")
    parser.add_argument("--minibatch_size", type=int, default=1024, help="Minibatch size for updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="results/ppo_sweep_E", help="Output directory root")
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
        np_metrics = jax.tree.map(lambda x: np.asarray(x), res["metrics"])
        seed_metrics.append(np_metrics)
        print(f"    Seed {s_idx + 1}/{n_seeds} completed in {dt:.2f}s")

    keys = seed_metrics[0].keys()
    stacked = {}
    for k in keys:
        try:
            stacked[k] = np.stack([sm[k] for sm in seed_metrics], axis=0)
        except Exception:
            pass

    return stacked


def plot_single_env_summary(env_out_dir: str, env_name: str, results: Dict[str, Any]):
    """Generates per-environment summary plots as a vector PDF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    ax_ret, ax_ve, ax_e = axes[0], axes[1], axes[2]

    for v_key, (v_label, _, color) in VARIANTS.items():
        if v_key not in results:
            continue
        data = results[v_key]

        # 1. Return / True Value
        ret_metric = "returned_episode_returns" if "returned_episode_returns" in data else ("V_start" if "V_start" in data else None)
        if ret_metric is not None and ret_metric in data:
            arr = np.asarray(data[ret_metric])
            if arr.ndim >= 2:
                m = np.nanmean(arr, axis=0)
                sem = np.nanstd(arr, axis=0) / np.sqrt(max(arr.shape[0], 1))
                x = np.arange(len(m))
                ax_ret.plot(x, m, label=v_label, color=color, lw=2.0)
                ax_ret.fill_between(x, m - sem, m + sem, color=color, alpha=0.18)

        # 2. Value Error MSE
        if "nn_weighted_VE" in data:
            arr = np.asarray(data["nn_weighted_VE"])
            if arr.ndim >= 2:
                m = np.nanmean(arr, axis=0)
                sem = np.nanstd(arr, axis=0) / np.sqrt(max(arr.shape[0], 1))
                x = np.arange(len(m))
                ax_ve.plot(x, m, label=v_label, color=color, lw=2.0)
                ax_ve.fill_between(x, np.maximum(m - sem, 1e-12), m + sem, color=color, alpha=0.18)

        # 3. Exact Matrix E
        e_key = "E" if "E" in data else ("E_exact" if "E_exact" in data else None)
        if e_key is not None and e_key in data:
            arr = np.asarray(data[e_key])
            if arr.ndim >= 2:
                m = np.nanmean(arr, axis=0)
                sem = np.nanstd(arr, axis=0) / np.sqrt(max(arr.shape[0], 1))
                x = np.arange(len(m))
                ax_e.plot(x, m, label=v_label, color=color, lw=2.0)
                ax_e.fill_between(x, np.maximum(m - sem, 1e-12), m + sem, color=color, alpha=0.18)

    ax_ret.set_title(f"{env_name}: Returns / $V^\\pi(s_0)$", fontsize=11, fontweight="bold")
    ax_ret.set_xlabel("Updates", fontsize=10)
    ax_ret.set_ylabel("Evaluation Return", fontsize=10)
    ax_ret.grid(True, alpha=0.3)
    ax_ret.legend(frameon=True, fontsize=9)

    ax_ve.set_title(f"{env_name}: Value Error MSE", fontsize=11, fontweight="bold")
    ax_ve.set_xlabel("Updates", fontsize=10)
    ax_ve.set_ylabel("Weighted VE (Log)", fontsize=10)
    ax_ve.set_yscale("log")
    ax_ve.grid(True, alpha=0.3)
    ax_ve.legend(frameon=True, fontsize=9)

    ax_e.set_title(f"{env_name}: Exact Matrix $E$", fontsize=11, fontweight="bold")
    ax_e.set_xlabel("Updates", fontsize=10)
    ax_e.set_ylabel("Exact $E$ (Log)", fontsize=10)
    ax_e.set_yscale("log")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    pdf_path = os.path.join(env_out_dir, "env_summary.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved environment summary plot to: {pdf_path}")


def main():
    args = parse_args()

    if args.env_name is not None:
        env_name = args.env_name
    elif args.env_idx is not None and 0 <= args.env_idx < len(ALL_ENVS):
        env_name = ALL_ENVS[args.env_idx]
    else:
        env_name = "FourRooms-misc"

    out_dir = os.path.abspath(args.out_dir)
    env_out_dir = os.path.join(out_dir, env_name)
    os.makedirs(env_out_dir, exist_ok=True)

    print("=" * 70)
    print("PPO Sampled E Critic Benchmark")
    print(f"Environment:  {env_name}")
    print(f"Num Updates:  {args.num_updates}")
    print(f"Num Seeds:    {args.n_seeds}")
    print(f"Rollout:      {args.num_envs} envs x {args.num_steps} steps")
    print(f"Output:       {env_out_dir}")
    print("=" * 70)

    is_mc = "mountaincar" in env_name.lower()
    cfg = default_config.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_UPDATES": args.num_updates,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "MINIBATCH_SIZE": args.minibatch_size,
        "TOTAL_TIMESTEPS": args.num_updates * args.num_envs * args.num_steps,
        "GAMMA": args.gamma,
        "LR": args.lr,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
    })

    results = {}
    for v_key, (v_label, make_fn, _) in VARIANTS.items():
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

    plot_single_env_summary(env_out_dir, env_name, results)


if __name__ == "__main__":
    main()
