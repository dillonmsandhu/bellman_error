"""
scripts/sampled_E_surgical_ppo_sweep/plot_results.py

Consolidated publication-quality plotting for Sampled E-Surgical PPO Sweeps.
Plots:
1. Learning curves across all envs:
   - Standardized Policy Return curves (std_perf_curves_all_algos.pdf & .png)
   - Raw Reward curves (reward_curves_all_algos.pdf & .png)
   - V_start tracking curves (vstart_curves_all_algos.pdf & .png)
2. Clustered bar charts across all envs:
   - AUC of Standardized Return (auc_std_perf_bars_all_algos.pdf)
   - Final Standardized Return (final_std_perf_bars_all_algos.pdf)
   - Final Raw Reward (final_raw_reward_bars_all_algos.pdf)
All figures saved to <results_dir>/figures/
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
matplotlib.rcParams["axes.edgecolor"] = "#333333"
matplotlib.rcParams["axes.linewidth"] = 0.8

ALL_ENVS = [
    "mountaincar",
    "FourRooms-misc",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]

ALGO_COLORS = {
    "mc": "#1f77b4",             # Steel Blue
    "td_lambda": "#ff7f0e",      # Safety Orange
    "oct8_biased": "#d62728",     # Crimson / Red
    "forward_e": "#00bcd4",      # Cyan / Teal
    "surgical_v_aux": "#2ca02c",  # Forest Green
}

ALGO_LABELS = {
    "mc": "MC (Sampled)",
    "td_lambda": "TD(λ=0.9)",
    "oct8_biased": "OG Oct 8 (Biased)",
    "forward_e": "Forward-Only E",
    "surgical_v_aux": "Surgical V_aux",
}

ENV_TITLES = {
    "mountaincar": "Mountain Car (Sparse)",
    "FourRooms-misc": "FourRooms (Sparse)",
    "fourrooms-mines": "FourRooms + Mines",
    "fourrooms-mines-dense": "FourRooms + Mines (Dense)",
    "eightrooms-dense": "EightRooms (Dense)",
    "eightrooms-misc": "EightRooms (Sparse)",
    "whirlpool-misc": "Whirlpool (Irreversible)",
    "SpaceInvadersExactValue": "Space Invaders",
    "mountaincar-dense": "Mountain Car (Dense)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Sampled E-Surgical PPO Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sampled_E_surgical_ppo_sweep",
        help="Root results directory containing environment subfolders",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <results_dir>/figures)",
    )
    return parser.parse_args()


def load_all_env_data(results_dir: str):
    env_data = {}
    summary_dfs = []

    for env_name in ALL_ENVS:
        env_dir = os.path.join(results_dir, env_name)
        pkl_path = os.path.join(env_dir, "out.pkl")
        sum_path = os.path.join(env_dir, "summary.csv")

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    env_data[env_name] = pickle.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load {pkl_path}: {e}")

        if os.path.exists(sum_path):
            try:
                sdf = pd.read_csv(sum_path)
                summary_dfs.append(sdf)
            except Exception as e:
                print(f"[Warning] Failed to load {sum_path}: {e}")

    master_df = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
    return env_data, master_df


def plot_learning_curves(env_data, metric_name: str, y_label: str, out_path_base: str):
    present_envs = [e for e in ALL_ENVS if e in env_data]
    if not present_envs:
        return

    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if env_name not in env_data:
            ax.text(0.5, 0.5, "Data Pending", ha="center", va="center", transform=ax.transAxes, color="#888888")
            continue

        runs = env_data[env_name]
        for algo_key, color in ALGO_COLORS.items():
            if algo_key not in runs:
                continue
            k_key = list(runs[algo_key].keys())[0]
            run = runs[algo_key][k_key]
            stacked = run["metrics"].get(metric_name)
            if stacked is None:
                continue

            mean_curve = np.mean(stacked, axis=0)
            sem_curve = np.std(stacked, axis=0) / np.sqrt(max(1, stacked.shape[0]))
            steps = np.arange(1, len(mean_curve) + 1)

            label = ALGO_LABELS.get(algo_key, algo_key)
            ax.plot(steps, mean_curve, color=color, label=label, linewidth=1.8)
            ax.fill_between(steps, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.15)

        if idx % n_cols == 0:
            ax.set_ylabel(y_label, fontsize=10, fontweight="semibold")
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Updates", fontsize=10, fontweight="semibold")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(handles),
            frameon=False,
            fontsize=12,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(out_path_base), exist_ok=True)
    fig.savefig(f"{out_path_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_path_base}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {out_path_base}.pdf / .png")


def plot_clustered_bars(df: pd.DataFrame, metric_col: str, sem_col: str, y_label: str, title: str, out_path_base: str):
    if df.empty:
        return

    present_envs = [e for e in ALL_ENVS if e in df["env_name"].unique()]
    if not present_envs:
        return

    n_envs = len(present_envs)
    algos = [a for a in ALGO_COLORS.keys() if a in df["algo"].unique()]
    n_algos = len(algos)
    if n_algos == 0:
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.8 / n_algos
    indices = np.arange(n_envs)

    for i, algo in enumerate(algos):
        algo_data = []
        algo_sems = []
        for env_name in present_envs:
            row = df[(df["env_name"] == env_name) & (df["algo"] == algo)]
            if not row.empty:
                algo_data.append(row[metric_col].values[0])
                algo_sems.append(row[sem_col].values[0] if sem_col in row and not np.isnan(row[sem_col].values[0]) else 0.0)
            else:
                algo_data.append(0.0)
                algo_sems.append(0.0)

        offset = (i - (n_algos - 1) / 2) * width
        ax.bar(
            indices + offset,
            algo_data,
            width,
            yerr=algo_sems,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS[algo],
            alpha=0.9,
            capsize=3,
            edgecolor="#333333",
            linewidth=0.6,
        )

    ax.set_xticks(indices)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in present_envs], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(y_label, fontsize=11, fontweight="semibold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path_base), exist_ok=True)
    fig.savefig(f"{out_path_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_path_base}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved: {out_path_base}.pdf / .png")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir if args.out_dir else os.path.join(results_dir, "figures"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: {results_dir}")
    env_data, df = load_all_env_data(results_dir)

    if not env_data and df.empty:
        print("[Warning] No experimental results found to plot.")
        return

    # 1. Learning Curves
    plot_learning_curves(
        env_data, "std_perf", "Standardized Return (0=Random, 1=Optimal)",
        os.path.join(out_dir, "std_perf_curves_all_algos"),
    )
    plot_learning_curves(
        env_data, "mean_rew", "Mean Step Reward",
        os.path.join(out_dir, "reward_curves_all_algos"),
    )
    plot_learning_curves(
        env_data, "v_pred_start", "$V_{\\mathrm{start}}$ Prediction",
        os.path.join(out_dir, "vstart_curves_all_algos"),
    )

    # 2. Clustered Bar Charts
    if not df.empty:
        plot_clustered_bars(
            df, "std_perf_auc_mean", "std_perf_auc_sem",
            "AUC of Standardized Return", "Standardized Return AUC (0=Random, 1=Optimal)",
            os.path.join(out_dir, "auc_std_perf_bars_all_algos"),
        )
        plot_clustered_bars(
            df, "std_perf_final_mean", "std_perf_final_sem",
            "Final Standardized Return", "Final Standardized Return (0=Random, 1=Optimal)",
            os.path.join(out_dir, "final_std_perf_bars_all_algos"),
        )
        if "vstart_auc_mean" in df.columns:
            plot_clustered_bars(
                df, "vstart_auc_mean", "vstart_auc_sem",
                "AUC of $V_{\\mathrm{start}}$", "$V_{\\mathrm{start}}$ Prediction AUC",
                os.path.join(out_dir, "auc_vstart_bars_all_algos"),
            )

    print(f"\nAll publication figures saved in: {out_dir}")


if __name__ == "__main__":
    main()
