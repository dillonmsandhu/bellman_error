"""
scripts/expected_algo_sweep/run_single_env.py

Runs the 4 expected (exact dynamics / full expectation) algorithms:
  1. MC:          ppo/exact_mc.py
  2. TD(0):       ppo/exact_td.py
  3. TD(lambda):  ppo/exact_td_lambda.py with VALUE_LAMBDA = 0.8
  4. E(0):        ppo/exact_E.py

Configuration for true gradient descent:
  - NUM_ENVS = 1, NUM_STEPS = 1
  - TOTAL_TIMESTEPS = 500 (500 exact policy iteration updates)
  - N_SEEDS = 8 (higher seed count for smooth confidence bands)
  - Base critic learning rate: 0.001 (swept over 0.5x, 1.0x, 2.0x -> 0.0005, 0.001, 0.002)
  - Actor LR kept constant at 0.001
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import json
from typing import Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import config as base_cfg
from ppo.exact_mc import make_train as make_mc_train
from ppo.exact_td import make_train as make_td0_train
from ppo.exact_td_lambda import make_train as make_td_lambda_train
from ppo.exact_E import make_train as make_e_train

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

ALGORITHMS = [
    {
        "name": "MC",
        "display_name": "MC (Exact)",
        "train_builder": make_mc_train,
        "hparams": {},
        "color": "#1f77b4",
    },
    {
        "name": "TD0",
        "display_name": "TD(0) [Exact]",
        "train_builder": make_td0_train,
        "hparams": {},
        "color": "#ff7f0e",
    },
    {
        "name": "TD_lambda",
        "display_name": "TD($\\lambda=0.8$) [Exact]",
        "train_builder": make_td_lambda_train,
        "hparams": {"VALUE_LAMBDA": 0.8},
        "color": "#2ca02c",
    },
    {
        "name": "E0",
        "display_name": "E(0) [Exact]",
        "train_builder": make_e_train,
        "hparams": {},
        "color": "#d62728",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep expected algorithms over critic LRs on a single environment")
    parser.add_argument("--env_name", type=str, default=None, help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-9)")
    parser.add_argument("--total_timesteps", type=int, default=500, help="Total updates (default: 500)")
    parser.add_argument("--n_seeds", type=int, default=8, help="Number of random seeds (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base critic LR for true gradient descent (default: 0.001)")
    parser.add_argument("--lr_multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                        help="Multipliers on base critic LR (default: 0.5 1.0 2.0)")
    parser.add_argument("--out_dir", type=str, default="results/expected_algo_sweep", help="Root output directory")
    return parser.parse_args()


def build_env_config(env_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    is_mc = "mountaincar" in env_name.lower()
    cfg = base_cfg.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": 1,
        "NUM_ENVS": 1,
        "MINIBATCH_SIZE": 1,
        "NUM_EPOCHS": 4,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_UPDATES": args.total_timesteps,
        "N_SEEDS": args.n_seeds,
        "SEED": args.seed,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
        "ACTOR_LR": args.base_lr,
        "ACTOR_LR_END": args.base_lr,
        "GAE_LAMBDA": 0.8,
    })
    return cfg


def run_condition(
    train_builder,
    config: Dict[str, Any],
    algo_hparams: Dict[str, Any],
    critic_lr: float,
    num_seeds: int,
    base_seed: int,
) -> Dict[str, np.ndarray]:
    """Runs a single algorithm + critic LR condition across multiple seeds via jax.vmap."""
    hparams = algo_hparams.copy()
    hparams["LR"] = critic_lr
    hparams["LR_END"] = critic_lr

    train_fn = train_builder(config)
    vmapped_train = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))

    rng = jax.random.PRNGKey(base_seed)
    rngs = jax.random.split(rng, num_seeds)

    t0 = time.time()
    out = vmapped_train(rngs, hparams)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - t0

    metrics = out["metrics"]
    extracted = {"elapsed_seconds": elapsed}

    for key in ["V_start", "v_pred_start", "value_loss", "sq_err", "actor_loss", "entropy", "total_loss"]:
        if key in metrics:
            extracted[key] = np.array(metrics[key])

    return extracted


def main():
    args = parse_args()

    if args.env_name is not None:
        env_name = args.env_name
    elif args.env_idx is not None and 0 <= args.env_idx < len(ALL_ENVS):
        env_name = ALL_ENVS[args.env_idx]
    else:
        env_name = "FourRooms-misc"

    env_dir = os.path.join(args.out_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)

    print("=" * 80)
    print(f"Expected Algorithm Sweep: {env_name}")
    print(f"Updates:           {args.total_timesteps}")
    print(f"Seeds:             {args.n_seeds} (base seed {args.seed})")
    print(f"Base Critic LR:    {args.base_lr} (True Gradient Descent)")
    print(f"LR Multipliers:    {args.lr_multipliers}")
    print(f"Output Directory:  {env_dir}")
    print("=" * 80)

    cfg = build_env_config(env_name, args)

    # Save configuration
    cfg_to_save = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))}
    with open(os.path.join(env_dir, "config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    critic_lrs = [args.base_lr * m for m in args.lr_multipliers]
    all_results = {}

    for algo in ALGORITHMS:
        algo_name = algo["name"]
        all_results[algo_name] = {}
        print(f"\n--- Running Algorithm: {algo['display_name']} ---")

        for lr_mult, lr in zip(args.lr_multipliers, critic_lrs):
            print(f"  Critic LR = {lr:.6f} ({lr_mult}x base)...", end="", flush=True)
            res = run_condition(
                train_builder=algo["train_builder"],
                config=cfg,
                algo_hparams=algo["hparams"],
                critic_lr=lr,
                num_seeds=args.n_seeds,
                base_seed=args.seed,
            )
            all_results[algo_name][f"{lr_mult}x"] = res
            print(f" Done ({res['elapsed_seconds']:.1f}s)")

    # Save raw array metrics
    np.savez_compressed(os.path.join(env_dir, "results.npz"), **{
        f"{algo}_{mult}_{k}": v
        for algo, lrs in all_results.items()
        for mult, data in lrs.items()
        for k, v in data.items()
        if isinstance(v, np.ndarray)
    })

    # Find best LR for each algorithm
    best_results = {}
    for algo in ALGORITHMS:
        algo_name = algo["name"]
        best_mult = None
        best_score = -float("inf")

        for lr_mult in [f"{m}x" for m in args.lr_multipliers]:
            data = all_results[algo_name][lr_mult]
            if "V_start" in data:
                score = np.mean(data["V_start"][:, -10:])
            elif "returned_episode_returns" in data:
                score = np.mean(data["returned_episode_returns"][:, -10:])
            else:
                score = 0.0

            if score > best_score or best_mult is None:
                best_score = score
                best_mult = lr_mult

        best_results[algo_name] = {
            "best_mult": best_mult,
            "data": all_results[algo_name][best_mult],
            "display_name": algo["display_name"],
            "color": algo["color"],
        }
        print(f"Algorithm {algo['name']:12s} | Best Critic LR: {best_mult} (Score: {best_score:.4f})")

    # Generate Environment-level Plots
    plot_env_summary(env_dir, env_name, ALGORITHMS, all_results, best_results, args.lr_multipliers)
    print(f"\nResults successfully saved to: {env_dir}")


def plot_env_summary(env_dir, env_name, algorithms, all_results, best_results, lr_multipliers):
    """Generates comparison plots for the environment."""
    # 1. Best-LR Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    ax_val, ax_err = axes[0], axes[1]

    val_metric_name = "V_start"

    for algo in algorithms:
        name = algo["name"]
        best_info = best_results[name]
        data = best_info["data"]
        mult = best_info["best_mult"]
        color = best_info["color"]
        label = f"{best_info['display_name']} ({mult})"

        # True Value Metric (from evaluator)
        val_metric = "V_start" if "V_start" in data else ("returned_episode_returns" if "returned_episode_returns" in data else None)
        if val_metric is not None and val_metric in data:
            val_metric_name = val_metric
            arr = data[val_metric]
            x = np.arange(arr.shape[1])
            m = np.mean(arr, axis=0)
            sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
            ax_val.plot(x, m, label=label, color=color, lw=2.0)
            ax_val.fill_between(x, m - sem, m + sem, color=color, alpha=0.18)

        # Value Error Metric
        err_metric = "sq_err" if "sq_err" in data else ("value_loss" if "value_loss" in data else None)
        if err_metric is not None and err_metric in data:
            arr = data[err_metric]
            x = np.arange(arr.shape[1])
            m = np.mean(arr, axis=0)
            sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
            ax_err.plot(x, m, label=label, color=color, lw=2.0)
            ax_err.fill_between(x, m - sem, m + sem, color=color, alpha=0.18)

    val_title = "True Start Value $V^\\pi(s_0)$" if val_metric_name == "V_start" else "Episode Return"
    ax_val.set_title(f"{env_name}: {val_title}", fontsize=11, fontweight="bold")
    ax_val.set_xlabel("Updates", fontsize=10)
    ax_val.set_ylabel(val_title, fontsize=10)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(frameon=True, fontsize=9)

    ax_err.set_title(f"{env_name}: Mean Squared Value Error", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Updates", fontsize=10)
    ax_err.set_ylabel("Squared Error", fontsize=10)
    ax_err.set_yscale("log")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(env_dir, "best_lr_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(env_dir, "best_lr_comparison.png"), bbox_inches="tight", dpi=200)
    plt.close()

    # 2. Comprehensive LR Sweep Grid Plot
    fig, axes = plt.subplots(len(algorithms), len(lr_multipliers), figsize=(4 * len(lr_multipliers), 3 * len(algorithms)), dpi=120, sharex=True)
    if len(algorithms) == 1:
        axes = np.expand_dims(axes, 0)
    if len(lr_multipliers) == 1:
        axes = np.expand_dims(axes, 1)

    for r_idx, algo in enumerate(algorithms):
        name = algo["name"]
        color = algo["color"]
        for c_idx, mult_val in enumerate(lr_multipliers):
            ax = axes[r_idx, c_idx]
            mult_str = f"{mult_val}x"
            data = all_results[name][mult_str]

            val_metric = "V_start" if "V_start" in data else ("returned_episode_returns" if "returned_episode_returns" in data else None)
            if val_metric is not None and val_metric in data:
                arr = data[val_metric]
                x = np.arange(arr.shape[1])
                m = np.mean(arr, axis=0)
                sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
                ax.plot(x, m, color=color, lw=1.8, label=f"$V^\\pi(s_0)$")
                ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.2)

            ax.set_title(f"{algo['display_name']} | LR: {mult_str}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if c_idx == 0:
                ax.set_ylabel(algo["name"], fontsize=10, fontweight="bold")
            if r_idx == len(algorithms) - 1:
                ax.set_xlabel("Updates", fontsize=9)

    plt.suptitle(f"{env_name}: Critic LR Sweep Across Expected Algorithms", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(env_dir, "all_lrs_grid.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(env_dir, "all_lrs_grid.png"), bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
