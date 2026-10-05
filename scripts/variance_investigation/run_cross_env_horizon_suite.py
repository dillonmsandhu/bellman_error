"""
run_cross_env_horizon_suite.py

Orchestrates the fixed-batch horizon experiment (N = 16384, T in [8, 64, 128, 512, 1024])
with multiple seeds in parallel via jax.vmap across specified environments, generating:
1. cross_env_squared_error_trajectories.png:
   Each environment as a subplot showing Mean +/- 95% CI band for ||g_sampled_E - g_exact_E||^2 across updates.
2. cross_env_sampled_vs_exact_E_trajectories.png:
   Each environment as a subplot showing Mean +/- 95% CI band for rho(g_sampled_E^(T), g_exact_E) across updates.
3. cross_env_avg_squared_error_vs_T.png:
   Average Squared Error on the vertical axis vs. rollout horizon T on the horizontal log-axis across envs.
4. cross_env_avg_cosine_vs_T.png:
   Average cosine similarity on the vertical axis vs. rollout horizon T on the horizontal log-axis across envs.
5. cross_env_alignment_relative_spectral_norm.png:
   Single plot showing ||K||_2 / ||S||_2 for all environments on the same plot (representative T=128).
"""

import os
import sys
import time
import argparse
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.variance_investigation.run_fixed_batch_horizon_investigation import (
    run_single_horizon_experiment,
    generate_all_investigation_plots,
)


def plot_cross_env_squared_error_trajectories(
    all_env_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    total_batch_size: int,
    num_seeds: int,
    out_path: str,
):
    """
    Subplots for each environment showing Mean +/- 95% CI band for ||g_sampled_E - g_exact_E||^2.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    env_names = list(all_env_data.keys())
    n_envs = len(env_names)

    n_cols = min(n_envs, 3)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, env_name in enumerate(env_names):
        ax = axes[i]
        env_res = all_env_data[env_name]
        sorted_steps = sorted(env_res.keys())
        colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(sorted_steps)))

        for idx, t in enumerate(sorted_steps):
            data = env_res[t]
            updates = data["update"]
            b = total_batch_size // t

            sq_mean = data["sq_err_mean"]
            sq_std = data["sq_err_std"] / np.sqrt(num_seeds)
            ax.plot(updates, sq_mean, color=colors[idx], lw=2.2, label=f"T = {t:4d} (B = {b:4d})")
            ax.fill_between(
                updates,
                np.maximum(sq_mean - 1.96 * sq_std, 1e-8),
                sq_mean + 1.96 * sq_std,
                color=colors[idx],
                alpha=0.15,
            )

        ax.set_yscale("log")
        ax.set_title(f"{env_name} ($N = {total_batch_size}$, {num_seeds} Seeds)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Policy Update Step", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"Squared Error $\|g_{\mathrm{sampled\_E}} - g_{\mathrm{exact\_E}}\|^2$", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    for j in range(n_envs, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        r"Empirical Gradient Squared Error $\|g_{\mathrm{sampled\_E}} - g_{\mathrm{exact\_E}}\|^2$ Across Learning"
        + f"\n(Fixed Batch $N = {total_batch_size}$, Mean $\\pm$ 95% CI across {num_seeds} Seeds)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Cross-Environment Squared Error Trajectories to: {out_path}")


def plot_cross_env_trajectories(
    all_env_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    total_batch_size: int,
    num_seeds: int,
    out_path: str,
):
    """
    Subplots for each environment showing rho(g_sampled_E, g_exact_E) across policy updates for each T.
    Y-limits strictly padded between -1.05 and 1.05.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    env_names = list(all_env_data.keys())
    n_envs = len(env_names)

    n_cols = min(n_envs, 3)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, env_name in enumerate(env_names):
        ax = axes[i]
        env_res = all_env_data[env_name]
        sorted_steps = sorted(env_res.keys())
        colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(sorted_steps)))

        for idx, t in enumerate(sorted_steps):
            data = env_res[t]
            updates = data["update"]
            b = total_batch_size // t

            rho_mean = data["rho_SE_EE_mean"]
            rho_std = data["rho_SE_EE_std"] / np.sqrt(num_seeds)
            ax.plot(
                updates,
                rho_mean,
                color=colors[idx],
                lw=2.0,
                label=f"T = {t:4d} (B = {b:4d})",
            )
            ax.fill_between(
                updates,
                rho_mean - 1.96 * rho_std,
                rho_mean + 1.96 * rho_std,
                color=colors[idx],
                alpha=0.15,
            )

        ax.axhline(1.0, color="#2ca02c", ls=":", lw=1.8, label=r"Perfect Target ($\rho = 1.0$)")
        ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
        ax.set_title(f"{env_name} ($N = {total_batch_size}$, {num_seeds} Seeds)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Policy Update Step", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"$\rho(g_{\mathrm{sampled\_E}}^{(T)}, g_{\mathrm{exact\_E}})$", fontsize=10, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_envs, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        r"Empirical Sampled $E$ Alignment to Exact $E$ Target Across Horizons $T$"
        + f"\n(Fixed Batch $N = {total_batch_size}$, Mean $\\pm$ 95% CI across {num_seeds} Seeds)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Cross-Environment Trajectories to: {out_path}")


def plot_cross_env_avg_squared_error_vs_T(
    all_env_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    out_path: str,
):
    """
    Single plot showing Average Squared Error ||g_samp - g_exact||^2 vs. Horizon T for each environment.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_env_data)))

    for idx, (env_name, env_res) in enumerate(all_env_data.items()):
        sorted_steps = sorted(env_res.keys())
        avg_sqs = [float(np.mean(env_res[t]["sq_err_mean"])) for t in sorted_steps]
        ax.plot(
            sorted_steps,
            avg_sqs,
            marker="d",
            lw=2.5,
            color=colors[idx],
            label=f"{env_name}",
        )

    all_t = sorted(list(next(iter(all_env_data.values())).keys()))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_t)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Rollout Horizon $T$ (Fixed Batch $N$)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Mean Squared Error $\|g_{\mathrm{sampled\_E}} - g_{\mathrm{exact\_E}}\|^2$", fontsize=11, fontweight="bold")
    ax.set_title(r"Truncation Squared Error Scaling vs. Horizon $T$ Across Environments", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Cross-Environment Avg Squared Error vs T to: {out_path}")


def plot_cross_env_avg_cosine_vs_T(
    all_env_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    out_path: str,
):
    """
    Single plot showing Average Cosine Similarity vs. Horizon T for each environment.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_env_data)))

    for idx, (env_name, env_res) in enumerate(all_env_data.items()):
        sorted_steps = sorted(env_res.keys())
        avg_rhos = [float(np.mean(env_res[t]["rho_SE_EE_mean"])) for t in sorted_steps]
        ax.plot(
            sorted_steps,
            avg_rhos,
            marker="o",
            lw=2.5,
            color=colors[idx],
            label=f"{env_name}",
        )

    ax.axhline(1.0, color="#27ae60", ls=":", lw=1.8, label=r"Target Alignment ($\rho = 1.0$)")
    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)

    all_t = sorted(list(next(iter(all_env_data.values())).keys()))
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_t)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Rollout Horizon $T$ (Fixed Batch $N$)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Average Directional Alignment $\bar{\rho}(g_{\mathrm{sampled\_E}}, g_{\mathrm{exact\_E}})$", fontsize=11, fontweight="bold")
    ax.set_title(r"Average Cosine Similarity vs. Horizon $T$ Across Environments", fontsize=13, fontweight="bold")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Cross-Environment Avg Cosine vs T to: {out_path}")


def plot_cross_env_alignment(
    all_env_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    rep_t: int,
    out_path: str,
):
    """
    Single plot showing Relative Spectral Norm ||K||_2 / ||S||_2 for all environments
    on the same plot, with policy update step on the x-axis for a representative horizon T.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_env_data)))

    for idx, (env_name, env_res) in enumerate(all_env_data.items()):
        t_key = rep_t if rep_t in env_res else sorted(env_res.keys())[len(env_res) // 2]
        data = env_res[t_key]
        updates = data["update"]
        spectral_ratios = data.get("relative_spectral_norm_mean", None)

        if spectral_ratios is not None and len(spectral_ratios) > 0:
            ax.plot(
                updates,
                spectral_ratios,
                color=colors[idx],
                lw=2.5,
                label=f"{env_name} (T={t_key})",
            )

    ax.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Relative Spectral Norm $\|K\|_2 / \|S\|_2$", fontsize=11, fontweight="bold")
    ax.set_title(
        r"Asymmetry / Alignment Dynamics $\|K\|_2 / \|S\|_2$ Across Learning",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Cross-Environment Alignment to: {out_path}")


def plot_cross_env_exact_e_vs_td(
    all_env_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    out_path: str,
):
    """
    Single plot showing Directional Cosine Similarity rho(g_exact_E, g_exact_TD)
    across policy optimization for each environment on the same axes.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_env_data)))

    for idx, (env_name, env_res) in enumerate(all_env_data.items()):
        # Any horizon T contains the exact closed-form gradient; pick representative T
        t_key = sorted(env_res.keys())[len(env_res) // 2]
        data = env_res[t_key]
        updates = data["update"]
        rho_e_td = data.get("rho_exactE_exactTD_mean", None)

        if rho_e_td is not None:
            ax.plot(
                updates,
                rho_e_td,
                color=colors[idx],
                lw=2.5,
                marker="o",
                markersize=4,
                label=f"{env_name}",
            )

    ax.axhline(1.0, color="#27ae60", ls=":", lw=1.8, label=r"Perfect Alignment ($\rho = 1.0$)")
    ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Cosine Similarity $\rho(g_{\mathrm{exact\_E}}, g_{\mathrm{exact\_TD}})$", fontsize=11, fontweight="bold")
    ax.set_title(
        r"Theoretical Directional Alignment Between Exact $E$ and Expected TD Semi-Gradient"
        + "\n"
        + r"$\rho(g_{\mathrm{exact\_E}}, g_{\mathrm{exact\_TD}}) = \frac{\langle 2(S+K)\Delta, (S+K)\Delta \rangle}{\|2(S+K)\Delta\| \|(S+K)\Delta\|}$",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Cross-Environment Exact E vs TD to: {out_path}")


def plot_cross_env_sampled_vs_exact_td_trajectories(
    all_env_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    total_batch_size: int,
    num_seeds: int,
    out_path: str,
):
    """
    Subplots for each environment showing rho(g_sampled_E^(T), g_exact_TD) across policy updates for each T.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    env_names = list(all_env_data.keys())
    n_envs = len(env_names)

    n_cols = min(n_envs, 3)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, env_name in enumerate(env_names):
        ax = axes[i]
        env_res = all_env_data[env_name]
        sorted_steps = sorted(env_res.keys())
        colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(sorted_steps)))

        for idx, t in enumerate(sorted_steps):
            data = env_res[t]
            updates = data["update"]
            b = total_batch_size // t

            rho_mean = data.get("rho_SE_ETD_mean", None)
            rho_std = data.get("rho_SE_ETD_std", np.zeros_like(updates)) / np.sqrt(num_seeds)

            if rho_mean is not None:
                ax.plot(
                    updates,
                    rho_mean,
                    color=colors[idx],
                    lw=2.0,
                    label=f"T = {t:4d} (B = {b:4d})",
                )
                ax.fill_between(
                    updates,
                    rho_mean - 1.96 * rho_std,
                    rho_mean + 1.96 * rho_std,
                    color=colors[idx],
                    alpha=0.15,
                )

        ax.axhline(1.0, color="#2ca02c", ls=":", lw=1.8, label=r"Perfect Target ($\rho = 1.0$)")
        ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
        ax.set_title(f"{env_name} ($N = {total_batch_size}$, {num_seeds} Seeds)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Policy Update Step", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"$\rho(g_{\mathrm{sampled\_E}}^{(T)}, g_{\mathrm{exact\_TD}})$", fontsize=10, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_envs, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        r"Empirical Sampled $E$ Alignment to Expected TD Semi-Gradient Across Horizons $T$"
        + f"\n(Fixed Batch $N = {total_batch_size}$, Mean $\\pm$ 95% CI across {num_seeds} Seeds)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Cross-Environment Sampled E vs TD Trajectories to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Environment Horizon Investigation Suite")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=[
            "fourrooms-dense",
            "FourRooms-misc",
            "eightrooms-dense",
            "eightrooms-misc",
            "whirlpool-misc",
            "SpaceInvadersExactValue",
            "mountaincar-dense",
        ],
        help="List of environments to run",
    )
    parser.add_argument("--total_batch_size", type=int, default=16384, help="Fixed batch size N = B * T")
    parser.add_argument("--horizons", nargs="+", type=int, default=[8, 64, 128, 512, 1024], help="Horizons T to sweep")
    parser.add_argument("--num_updates", type=int, default=30, help="Number of policy updates")
    parser.add_argument("--num_seeds", type=int, default=4, help="Number of random seeds in parallel via vmap")
    parser.add_argument("--seed", type=int, default=42, help="PRNG base seed")
    parser.add_argument("--out_dir", type=str, default=None, help="Root output directory")
    parser.add_argument("--replot_only", action="store_true", help="Re-generate plots from existing saved .npz files without re-running simulations")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, "results/horizon_investigation_16k_multiseed")
    os.makedirs(args.out_dir, exist_ok=True)

    N = args.total_batch_size
    valid_horizons = [t for t in args.horizons if N % t == 0]

    print("=" * 80)
    print("STARTING CROSS-ENVIRONMENT FIXED-BATCH HORIZON INVESTIGATION (MULTI-SEED VMAP)")
    print(f"Environments:     {args.envs}")
    print(f"Total Batch Size: {N} (STRICTLY FIXED)")
    print(f"Horizons (T):     {valid_horizons}")
    print(f"Parallel Envs(B): {[N // t for t in valid_horizons]}")
    print(f"Policy Updates:   {args.num_updates} | Seeds: {args.num_seeds} (vmapped) | Base Seed: {args.seed}")
    print(f"Output Directory: {args.out_dir}")
    print(f"Replot Only:      {args.replot_only}")
    print("=" * 80)

    all_env_data = {}

    for env_name in args.envs:
        print(f"\n==========================================")
        print(f"Processing Environment: {env_name}")
        print(f"==========================================")
        env_out_dir = os.path.join(args.out_dir, env_name)
        os.makedirs(env_out_dir, exist_ok=True)
        data_path = os.path.join(env_out_dir, f"{env_name}_fixed_batch_data.npz")

        env_results = {}

        if args.replot_only and os.path.exists(data_path):
            print(f"Loading existing data from {data_path}...")
            npz = np.load(data_path)
            for key in npz.files:
                parts = key.split("_")
                t = int(parts[1])
                k = "_".join(parts[2:])
                if t not in env_results:
                    env_results[t] = {}
                env_results[t][k] = npz[key]
        else:
            for t in valid_horizons:
                b = N // t
                print(f"\n--> Running Condition: {env_name} | T = {t:4d}, B = {b:4d} (N = {b*t}, Seeds = {args.num_seeds})...", flush=True)
                t_start = time.time()
                res = run_single_horizon_experiment(
                    env_name=env_name,
                    num_steps=t,
                    num_envs=b,
                    num_updates=args.num_updates,
                    seed=args.seed,
                    num_seeds=args.num_seeds,
                )
                elapsed = time.time() - t_start
                env_results[t] = res
                print(
                    f"    Completed in {elapsed:.1f}s | "
                    f"Final Return: {res['policy_return_mean'][-1]:.4f} +/- {res['policy_return_std'][-1]:.4f} | "
                    f"Mean SqErr: {np.mean(res['sq_err_mean']):.6f} | "
                    f"Mean rho(SE, EE): {np.mean(res['rho_SE_EE_mean']):.4f}"
                )

            # Save per-environment dataset
            save_dict = {f"T_{t}_{k}": v for t, d in env_results.items() for k, v in d.items()}
            np.savez(data_path, **save_dict)
            print(f"Saved dataset to {data_path}")

        generate_all_investigation_plots(
            env_results,
            env_name=env_name,
            total_batch_size=N,
            out_dir=env_out_dir,
            num_seeds=args.num_seeds,
        )
        all_env_data[env_name] = env_results

    # Generate Cross-Environment Plots
    print("\nGenerating Cross-Environment Consolidated Plots...")
    plot_cross_env_trajectories(
        all_env_data,
        total_batch_size=N,
        num_seeds=args.num_seeds,
        out_path=os.path.join(args.out_dir, "cross_env_sampled_vs_exact_E_trajectories.png"),
    )
    plot_cross_env_avg_cosine_vs_T(
        all_env_data,
        out_path=os.path.join(args.out_dir, "cross_env_avg_cosine_vs_T.png"),
    )
    plot_cross_env_exact_e_vs_td(
        all_env_data,
        out_path=os.path.join(args.out_dir, "cross_env_exact_E_vs_TD_all_tasks.png"),
    )
    plot_cross_env_sampled_vs_exact_td_trajectories(
        all_env_data,
        total_batch_size=N,
        num_seeds=args.num_seeds,
        out_path=os.path.join(args.out_dir, "cross_env_sampled_vs_exact_TD_trajectories.png"),
    )
    plot_cross_env_squared_error_trajectories(
        all_env_data,
        total_batch_size=N,
        num_seeds=args.num_seeds,
        out_path=os.path.join(args.out_dir, "cross_env_squared_error_trajectories.png"),
    )
    plot_cross_env_avg_squared_error_vs_T(
        all_env_data,
        out_path=os.path.join(args.out_dir, "cross_env_avg_squared_error_vs_T.png"),
    )
    plot_cross_env_alignment(
        all_env_data,
        rep_t=128,
        out_path=os.path.join(args.out_dir, "cross_env_alignment_relative_spectral_norm.png"),
    )
    print("\nAll multi-seed experiments and plots completed successfully!")


if __name__ == "__main__":
    main()
