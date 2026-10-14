"""
scripts/exact_ppo_sweep/plot_results.py

Generates publication-quality charts for the Exact PPO Evaluation Sweep:
1. standardized_performance_learning_curves.pdf:
   - 3x3 grid across all 9 environments.
   - Standardized policy performance (V_start - V_rand) / (V* - V_rand) over PPO updates.
   - Shaded ±1 SEM error regions across random seeds.
   - Reference lines at 1.0 (Optimal) and 0.0 (Random).
2. final_standardized_performance_bars.pdf:
   - Side-by-side grouped bar charts of final standardized performance across environments.
3. auc_standardized_performance_bars.pdf:
   - Side-by-side grouped bar charts of AUC standardized performance across environments.
"""

from __future__ import annotations
import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication font formatting
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
    "mc": "#1f77b4",          # Steel Blue
    "td0": "#ff7f0e",         # Dark Orange
    "e0": "#2ca02c",          # Forest Green
    "td_lambda": "#d62728",   # Crimson Red
    "e_lambda": "#9467bd",    # Purple
}

ALGO_LABELS = {
    "mc": "MC (Exact Critic)",
    "td0": "TD(0) [Exact Critic]",
    "e0": "E(0) [Exact Critic]",
    "td_lambda": "TD(λ=0.8) [Exact Critic]",
    "e_lambda": "E(λ=0.8) [Exact Critic]",
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
    parser = argparse.ArgumentParser(description="Generate Exact PPO Sweep Publication Plots")
    parser.add_argument("--results_dir", type=str, default="results/exact_ppo_sweep", help="Results directory root")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory (default: same as results_dir/plots)")
    return parser.parse_args()


def load_all_env_data(results_dir: str):
    """Loads all out.pkl checkpoints and summary.csv tables from results_dir."""
    env_data = {}
    summary_dfs = []

    for env_name in ALL_ENVS:
        env_dir = os.path.join(results_dir, env_name)
        pkl_path = os.path.join(env_dir, "out.pkl")
        csv_path = os.path.join(env_dir, "summary.csv")

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                env_data[env_name] = data
            except Exception as e:
                print(f"[Warning] Failed to load {pkl_path}: {e}")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                summary_dfs.append(df)
            except Exception as e:
                print(f"[Warning] Failed to read {csv_path}: {e}")

    combined_summary = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
    return env_data, combined_summary


def plot_standardized_learning_curves(
    env_data: dict,
    out_pdf_path: str,
):
    """Generates 3x3 grid of standardized policy performance learning curves across PPO updates."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    algos_ordered = ["mc", "td0", "e0", "td_lambda", "e_lambda"]
    legend_handles = {}

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.grid(True, linestyle="--", alpha=0.35)

        # Baseline horizontal reference lines
        ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2, alpha=0.7, label="Optimal $V^*$")
        ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7, label="Random $V^{\\pi_{\\mathrm{rand}}}$")

        if env_name not in env_data or "runs" not in env_data[env_name]:
            ax.text(0.5, 0.5, "Data Not Found", ha="center", va="center", transform=ax.transAxes, color="#888888")
            continue

        runs = env_data[env_name]["runs"]

        for algo_key in algos_ordered:
            if algo_key not in runs:
                continue
            run_dict = runs[algo_key]
            std_traj = run_dict.get("standardized_v_start")

            if std_traj is None:
                continue

            n_seeds = std_traj.shape[0]
            mean_curve = np.mean(std_traj, axis=0)
            sem_curve = np.std(std_traj, axis=0, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean_curve)
            steps = np.arange(1, len(mean_curve) + 1)

            color = ALGO_COLORS.get(algo_key, "#333333")
            label = ALGO_LABELS.get(algo_key, algo_key)

            line = ax.plot(steps, mean_curve, color=color, linewidth=1.8, label=label)[0]
            ax.fill_between(steps, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.18)

            if algo_key not in legend_handles:
                legend_handles[algo_key] = line

        ax.set_xlabel("PPO Updates", fontsize=10)
        ax.set_ylabel("Standardized Performance", fontsize=10)
        ax.tick_params(labelsize=9)

    # Global legend at the top
    handles = [legend_handles[k] for k in algos_ordered if k in legend_handles]
    labels = [ALGO_LABELS[k] for k in algos_ordered if k in legend_handles]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(handles), frameon=True, fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] {out_pdf_path}")


def plot_performance_bars(
    summary_df: pd.DataFrame,
    metric_col: str,
    sem_col: str,
    title: str,
    ylabel: str,
    out_pdf_path: str,
):
    """Generates grouped bar charts comparing standardized performance across environments."""
    if summary_df.empty:
        return

    envs_present = [e for e in ALL_ENVS if e in summary_df["env"].unique()]
    algos_ordered = ["mc", "td0", "e0", "td_lambda", "e_lambda"]
    algos_present = [a for a in algos_ordered if a in summary_df["algo"].unique()]

    if not envs_present or not algos_present:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    n_envs = len(envs_present)
    n_algos = len(algos_present)
    bar_width = 0.8 / n_algos
    x = np.arange(n_envs)

    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)

    for i, algo in enumerate(algos_present):
        means = []
        sems = []
        for e in envs_present:
            sub = summary_df[(summary_df["env"] == e) & (summary_df["algo"] == algo)]
            if not sub.empty:
                means.append(sub[metric_col].values[0])
                sems.append(sub[sem_col].values[0] if sem_col in sub else 0.0)
            else:
                means.append(0.0)
                sems.append(0.0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            yerr=sems,
            width=bar_width,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#333333"),
            edgecolor="#222222",
            linewidth=0.8,
            capsize=3,
            alpha=0.9,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs_present], rotation=30, ha="right", fontsize=9.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=10.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] {out_pdf_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else os.path.join(results_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("Exact PPO Sweep Plot Generation")
    print(f"Results Directory: {results_dir}")
    print(f"Save Directory:    {save_dir}")
    print("=" * 70)

    env_data, summary_df = load_all_env_data(results_dir)
    print(f"Loaded data for {len(env_data)} environments.")

    # 1. 3x3 Standardized Learning Curves
    learning_curves_pdf = os.path.join(save_dir, "standardized_performance_learning_curves.pdf")
    plot_standardized_learning_curves(env_data, learning_curves_pdf)

    # 2. Final Standardized Performance Bars
    final_bars_pdf = os.path.join(save_dir, "final_standardized_performance_bars.pdf")
    plot_performance_bars(
        summary_df=summary_df,
        metric_col="std_perf_final_mean",
        sem_col="std_perf_final_sem",
        title="Final Standardized Policy Performance (PPO Exact Evaluator)",
        ylabel="Standardized Performance (Random=0, Optimal=1)",
        out_pdf_path=final_bars_pdf,
    )

    # 3. AUC Standardized Performance Bars
    auc_bars_pdf = os.path.join(save_dir, "auc_standardized_performance_bars.pdf")
    plot_performance_bars(
        summary_df=summary_df,
        metric_col="std_perf_auc_mean",
        sem_col="std_perf_auc_sem",
        title="Area Under Curve (AUC) Standardized Policy Performance (PPO Exact Evaluator)",
        ylabel="AUC Standardized Performance",
        out_pdf_path=auc_bars_pdf,
    )

    print("\n[Complete] All PPO publication plots successfully generated.")


if __name__ == "__main__":
    main()
