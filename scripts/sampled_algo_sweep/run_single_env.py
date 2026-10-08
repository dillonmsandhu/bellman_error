"""
scripts/sampled_algo_sweep/run_single_env.py

Runs the 4 sampled algorithms:
  1. MC:          ppo/sampled_td_lambda.py with VALUE_LAMBDA = 1.0
  2. TD(0):       ppo/sampled_td_lambda.py with VALUE_LAMBDA = 0.0
  3. TD(lambda):  ppo/sampled_td_lambda.py with VALUE_LAMBDA = 0.8
  4. E(0):        ppo/sampled_E.py         with RETURN_LAMBDA = 0.0 (corrected boundary loss)

Sweeps 3 critic learning rates (0.5x, 1.0x, 2.0x base LR) per algorithm while keeping
actor LR fixed at the base config rate.
Uses appropriately long rollout horizons (e.g., NUM_STEPS = 256) to allow natural episode
transitions without horizon truncation artifacts.
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import json
from typing import Dict, Any, List

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
from ppo.sampled_td_lambda import make_train as make_td_train
from ppo.sampled_E import make_train as make_e_train

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
        "display_name": "MC ($\\lambda=1.0$)",
        "train_builder": make_td_train,
        "hparams": {"VALUE_LAMBDA": 1.0},
        "color": "#1f77b4",
    },
    {
        "name": "TD0",
        "display_name": "TD(0)",
        "train_builder": make_td_train,
        "hparams": {"VALUE_LAMBDA": 0.0},
        "color": "#ff7f0e",
    },
    {
        "name": "TD_lambda",
        "display_name": "TD($\\lambda=0.8$)",
        "train_builder": make_td_train,
        "hparams": {"VALUE_LAMBDA": 0.8},
        "color": "#2ca02c",
    },
    {
        "name": "E0",
        "display_name": "E(0) [Corrected]",
        "train_builder": make_e_train,
        "hparams": {"RETURN_LAMBDA": 0.0},
        "color": "#d62728",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep sampled algorithms over critic LRs on a single environment")
    parser.add_argument("--env_name", type=str, default=None, help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-9)")
    parser.add_argument("--num_steps", type=int, default=256, help="Rollout steps per trajectory (default: 256)")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments (default: 64)")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total environment steps (default: 1,000,000)")
    parser.add_argument("--minibatch_size", type=int, default=1024, help="Minibatch size for PPO updates (default: 1024)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Epochs per PPO update (default: 4)")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--base_lr", type=float, default=3e-4, help="Base critic learning rate (default: 3e-4)")
    parser.add_argument("--lr_multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                        help="Multipliers on base critic LR (default: 0.5 1.0 2.0)")
    parser.add_argument("--out_dir", type=str, default="results/sampled_algo_sweep", help="Root output directory")
    return parser.parse_args()


def build_env_config(env_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    is_mc = "mountaincar" in env_name.lower()
    cfg = base_cfg.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": args.num_steps,
        "NUM_ENVS": args.num_envs,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "MINIBATCH_SIZE": args.minibatch_size,
        "NUM_EPOCHS": args.num_epochs,
        "N_SEEDS": args.n_seeds,
        "SEED": args.seed,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
        "ACTOR_LR": args.base_lr,      # Actor LR fixed at base rate
        "ACTOR_LR_END": args.base_lr,
        "GAE_LAMBDA": 0.8,             # Advantage lambda for actor
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

    for key in ["v_pred_start", "v_true_start", "mean_rew", "value_loss", "sq_err", "actor_loss", "entropy"]:
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
    print(f"Sampled Algorithm Sweep: {env_name}")
    print(f"Rollout Config:    NUM_STEPS = {args.num_steps}, NUM_ENVS = {args.num_envs}")
    print(f"Total Timesteps:   {args.total_timesteps:,}")
    print(f"Seeds:             {args.n_seeds} (base seed {args.seed})")
    print(f"Base Critic LR:    {args.base_lr}")
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
    summary_data = []

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
            if "v_true_start" in data:
                # Score by final start value
                score = np.mean(data["v_true_start"][:, -5:])
            elif "mean_rew" in data:
                score = np.mean(data["mean_rew"][:, -5:])
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
    ax_val, ax_rew = axes[0], axes[1]

    for algo in algorithms:
        name = algo["name"]
        best_info = best_results[name]
        data = best_info["data"]
        mult = best_info["best_mult"]
        color = best_info["color"]
        label = f"{best_info['display_name']} ({mult})"

        # Value Metric
        val_metric = "v_true_start" if "v_true_start" in data else ("v_pred_start" if "v_pred_start" in data else None)
        if val_metric is not None and val_metric in data:
            arr = data[val_metric]
            x = np.arange(arr.shape[1])
            m = np.mean(arr, axis=0)
            sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
            ax_val.plot(x, m, label=label, color=color, lw=2.0)
            ax_val.fill_between(x, m - sem, m + sem, color=color, alpha=0.18)

        # Reward / Return Metric
        if "mean_rew" in data:
            arr = data["mean_rew"]
            x = np.arange(arr.shape[1])
            m = np.mean(arr, axis=0)
            sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
            ax_rew.plot(x, m, label=label, color=color, lw=2.0)
            ax_rew.fill_between(x, m - sem, m + sem, color=color, alpha=0.18)

    ax_val.set_title(f"{env_name}: Start State Value $V(s_0)$", fontsize=11, fontweight="bold")
    ax_val.set_xlabel("PPO Updates", fontsize=10)
    ax_val.set_ylabel("Estimated $V(s_0)$", fontsize=10)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(frameon=True, fontsize=9)

    ax_rew.set_title(f"{env_name}: Mean Step Reward", fontsize=11, fontweight="bold")
    ax_rew.set_xlabel("PPO Updates", fontsize=10)
    ax_rew.set_ylabel("Reward", fontsize=10)
    ax_rew.grid(True, alpha=0.3)
    ax_rew.legend(frameon=True, fontsize=9)

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

            val_metric = "v_true_start" if "v_true_start" in data else ("v_pred_start" if "v_pred_start" in data else None)
            if val_metric is not None and val_metric in data:
                arr = data[val_metric]
                x = np.arange(arr.shape[1])
                m = np.mean(arr, axis=0)
                sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
                ax.plot(x, m, color=color, lw=1.8, label=f"$V(s_0)$")
                ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.2)

            ax.set_title(f"{algo['display_name']} | LR: {mult_str}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if c_idx == 0:
                ax.set_ylabel(algo["name"], fontsize=10, fontweight="bold")
            if r_idx == len(algorithms) - 1:
                ax.set_xlabel("PPO Updates", fontsize=9)

    plt.suptitle(f"{env_name}: Critic LR Sweep Across Sampled Algorithms", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(env_dir, "all_lrs_grid.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(env_dir, "all_lrs_grid.png"), bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
