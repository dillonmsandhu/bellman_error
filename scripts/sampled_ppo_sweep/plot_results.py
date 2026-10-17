"""
scripts/sampled_ppo_sweep/plot_results.py

Generates publication-quality vector PDF charts for the Sampled PPO Evaluation Sweep:
1. standardized_performance_learning_curves.pdf:
   - 3x3 grid across all 9 environments.
   - Standardized policy performance (V_start - V_rand) / (V* - V_rand) over PPO updates.
   - Shaded ±1 SEM error regions across random seeds.
   - Reference lines at 1.0 (Optimal Skyline) and 0.0 (Random Baseline).
2. e0_variants_comparison_curves.pdf:
   - Focused 3x3 grid specifically comparing the triad of E(0) sampling variants:
     Sept 14 (uncorrected), Oct 5 (corrected), Oct 8 (start-anchored).
3. value_error_learning_curves.pdf:
   - 3x3 grid of learned critic Value Error (nn_weighted_VE) across rollouts.
4. advantage_cossim_learning_curves.pdf:
   - 3x3 grid of policy advantage gradient alignment (Advantage Cosine Similarity).
5. final_standardized_performance_bars.pdf & auc_standardized_performance_bars.pdf:
   - Grouped bar charts across all 9 environments with ±1 SEM error bars.
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
    "e0_sep14": "#2ca02c",    # Forest Green
    "e0_oct5": "#17becf",     # Cyan
    "e0_oct8": "#9467bd",     # Purple
    "e0_sep14_lam1": "#2ca02c",
    "e0_oct5_lam1": "#17becf",
    "e0_oct8_lam1": "#9467bd",
}

ALGO_LABELS = {
    "mc": "MC (Sampled)",
    "td0": "TD(0) [Sampled]",
    "e0_sep14": "E(0) [Sept 14 / Uncorr]",
    "e0_oct5": "E(0) [Oct 5 / Corr]",
    "e0_oct8": "E(0) [Oct 8 / Start-Anchored]",
    "e0_sep14_lam1": "E(0) [Sept 14] (λ=1)",
    "e0_oct5_lam1": "E(0) [Oct 5] (λ=1)",
    "e0_oct8_lam1": "E(0) [Oct 8] (λ=1)",
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
    parser = argparse.ArgumentParser(description="Generate Sampled PPO Sweep Publication Plots")
    parser.add_argument("--results_dir", type=str, default="results/sampled_ppo_sweep", help="Results directory root")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory (default: <results_dir>/plots)")
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
                print(f"[Warning] Failed to load {csv_path}: {e}")

    combined_df = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
    return env_data, combined_df


def plot_learning_curves_grid(
    env_data: dict,
    metric_key: str,
    output_pdf: str,
    y_label: str,
    y_limits: tuple[float, float] | None = None,
    add_baselines: bool = False,
    algo_subset: list[str] | None = None,
    plot_title: str | None = None,
    use_lam1_for_e: bool = False,
):
    """Generates a 3x3 grid of learning curves across all 9 environments."""
    algos_to_plot = algo_subset if algo_subset is not None else list(ALGO_COLORS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.3, color="#888888")

        if env_name not in env_data:
            ax.text(0.5, 0.5, "No Data", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, color="#888888")
            continue

        data = env_data[env_name]

        for algo_key in algos_to_plot:
            res = None
            if use_lam1_for_e and algo_key.startswith("e0") and "runs_lam1" in data:
                res = data["runs_lam1"].get(algo_key)
            if res is None:
                res = data.get("runs", {}).get(algo_key)

            if res is None:
                continue

            arr = res.get(metric_key)

            if arr is None:
                continue

            # arr shape: (n_seeds, num_updates)
            n_seeds = arr.shape[0]
            mean_curve = np.mean(arr, axis=0)
            sem_curve = np.std(arr, axis=0, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean_curve)
            x_steps = np.arange(len(mean_curve))

            color = ALGO_COLORS.get(algo_key, "#333333")
            label = ALGO_LABELS.get(algo_key, algo_key)
            if use_lam1_for_e and algo_key.startswith("e0") and not label.endswith("(λ=1)"):
                label = f"{label} (λ=1)"

            ax.plot(x_steps, mean_curve, label=label, color=color, linewidth=1.8)
            ax.fill_between(x_steps, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.15)

        if add_baselines:
            ax.axhline(1.0, color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.8, label="Optimal (1.0)")
            ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8, label="Random (0.0)")

        if y_limits is not None:
            ax.set_ylim(y_limits)

        ax.tick_params(axis="both", which="major", labelsize=9)
        if idx % 3 == 0:
            ax.set_ylabel(y_label, fontsize=10)
        if idx >= 6:
            ax.set_xlabel("PPO Rollout Updates", fontsize=10)

    # Global legend from first populated axis
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if len(h) > len(handles):
            handles, labels = h, l

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 7),
            fontsize=10,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cccccc",
        )

    if plot_title:
        fig.suptitle(plot_title, fontsize=15, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95 if plot_title else 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {output_pdf}")


def plot_bar_chart(
    df: pd.DataFrame,
    metric_mean_col: str,
    metric_sem_col: str,
    output_pdf: str,
    y_label: str,
    title: str,
    algo_order: list[str] | None = None,
):
    """Generates a grouped bar chart across environments."""
    if df.empty or metric_mean_col not in df.columns:
        print(f"[Skip] Missing data for bar chart {output_pdf}")
        return

    order = algo_order if algo_order is not None else list(ALGO_COLORS.keys())
    present_envs = [e for e in ALL_ENVS if e in df["env"].unique()]
    if not present_envs:
        return

    n_envs = len(present_envs)
    n_algos = len(order)
    bar_width = 0.8 / n_algos

    fig, ax = plt.subplots(figsize=(14, 6))
    x_indices = np.arange(n_envs)

    for i, algo_key in enumerate(order):
        algo_sub = df[df["algo"] == algo_key]
        means = []
        sems = []
        for env_name in present_envs:
            row = algo_sub[algo_sub["env"] == env_name]
            if not row.empty:
                means.append(row[metric_mean_col].values[0])
                sems.append(row[metric_sem_col].values[0] if metric_sem_col in row.columns else 0.0)
            else:
                means.append(np.nan)
                sems.append(0.0)

        x_pos = x_indices + (i - n_algos / 2.0 + 0.5) * bar_width
        ax.bar(
            x_pos,
            means,
            yerr=sems,
            width=bar_width,
            label=ALGO_LABELS.get(algo_key, algo_key),
            color=ALGO_COLORS.get(algo_key, "#333333"),
            capsize=3,
            alpha=0.9,
            edgecolor="#222222",
            linewidth=0.6,
        )

    ax.set_xticks(x_indices)
    env_labels = [ENV_TITLES.get(e, e).replace(" ", "\n") for e in present_envs]
    ax.set_xticklabels(env_labels, fontsize=9, rotation=0)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {output_pdf}")


def main():
    args = parse_args()
    save_dir = args.save_dir if args.save_dir is not None else os.path.join(args.results_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 75)
    print("Generating Sampled PPO Sweep Publication Plots")
    print(f"Results Directory: {args.results_dir}")
    print(f"Save Directory:    {save_dir}")
    print("=" * 75)

    env_data, summary_df = load_all_env_data(args.results_dir)

    if not env_data and summary_df.empty:
        print("[Error] No data found in results directory. Run the sweep first.")
        return

    # 1. Full 5-Algorithm Standardized Performance Curves (Cross Comparison: E with lambda=1)
    plot_learning_curves_grid(
        env_data=env_data,
        metric_key="standardized_v_start",
        output_pdf=os.path.join(save_dir, "standardized_performance_learning_curves.pdf"),
        y_label="Standardized Policy Return",
        y_limits=(-0.1, 1.15),
        add_baselines=True,
        use_lam1_for_e=True,
        plot_title="Full Sampled PPO: Standardized Policy Return Learning Curves (E: λ=1)",
    )

    # 2. Focused E(0) Sampling Variants Triad Comparison (E Variants: tuned lambda in {0.95, 1.0})
    plot_learning_curves_grid(
        env_data=env_data,
        metric_key="standardized_v_start",
        output_pdf=os.path.join(save_dir, "e0_variants_comparison_curves.pdf"),
        y_label="Standardized Policy Return",
        y_limits=(-0.1, 1.15),
        add_baselines=True,
        algo_subset=["e0_sep14", "e0_oct5", "e0_oct8"],
        use_lam1_for_e=False,
        plot_title="E(0) Formulations Triad: Sept 14 vs Oct 5 vs Oct 8 (Tuned λ ∈ {0.95, 1.0})",
    )

    # 3. Core Baseline Comparison (MC vs TD(0) vs E(0) [Oct 8, lambda=1])
    plot_learning_curves_grid(
        env_data=env_data,
        metric_key="standardized_v_start",
        output_pdf=os.path.join(save_dir, "core_baselines_comparison_curves.pdf"),
        y_label="Standardized Policy Return",
        y_limits=(-0.1, 1.15),
        add_baselines=True,
        algo_subset=["mc", "td0", "e0_oct8"],
        use_lam1_for_e=True,
        plot_title="Core Critic Comparison: MC vs TD(0) vs E(0) [Oct 8, λ=1]",
    )

    # 4. Learned Critic Value Error Learning Curves (Cross Comparison: E with lambda=1)
    plot_learning_curves_grid(
        env_data=env_data,
        metric_key="nn_weighted_VE",
        output_pdf=os.path.join(save_dir, "value_error_learning_curves.pdf"),
        y_label="Critic Value Error (VE)",
        y_limits=None,
        add_baselines=False,
        use_lam1_for_e=True,
        plot_title="Full Sampled PPO: Critic Value Error (nn_weighted_VE) Across Updates (E: λ=1)",
    )

    # 5. Advantage Cosine Similarity Learning Curves (Cross Comparison: E with lambda=1)
    plot_learning_curves_grid(
        env_data=env_data,
        metric_key="nn_advantage_cossim",
        output_pdf=os.path.join(save_dir, "advantage_cossim_learning_curves.pdf"),
        y_label="Advantage CosSim",
        y_limits=(-0.2, 1.05),
        add_baselines=False,
        use_lam1_for_e=True,
        plot_title="Full Sampled PPO: Policy Advantage Cosine Similarity (E: λ=1)",
    )

    # 6. Grouped Bar Charts
    if not summary_df.empty:
        # Cross comparison bars (using lambda=1 for E)
        has_lam1 = "e0_oct8_lam1" in summary_df["algo"].unique()
        core_cross_order = ["mc", "td0", "e0_oct8_lam1" if has_lam1 else "e0_oct8"]

        plot_bar_chart(
            df=summary_df,
            metric_mean_col="std_perf_final_mean",
            metric_sem_col="std_perf_final_sem",
            output_pdf=os.path.join(save_dir, "final_standardized_performance_bars.pdf"),
            y_label="Final Standardized Return",
            title="Core Final Return: MC vs TD(0) vs E(0) [Oct 8, λ=1]",
            algo_order=core_cross_order,
        )

        plot_bar_chart(
            df=summary_df,
            metric_mean_col="std_perf_auc_mean",
            metric_sem_col="std_perf_auc_sem",
            output_pdf=os.path.join(save_dir, "auc_standardized_performance_bars.pdf"),
            y_label="AUC Standardized Return",
            title="Core AUC Return: MC vs TD(0) vs E(0) [Oct 8, λ=1]",
            algo_order=core_cross_order,
        )

        # E(0) Variants Triad bars (tuned lambda in {0.95, 1.0})
        e_order = ["e0_sep14", "e0_oct5", "e0_oct8"]
        plot_bar_chart(
            df=summary_df,
            metric_mean_col="std_perf_final_mean",
            metric_sem_col="std_perf_final_sem",
            output_pdf=os.path.join(save_dir, "e0_variants_final_performance_bars.pdf"),
            y_label="Final Standardized Return",
            title="E(0) Formulations Final Return (Tuned λ ∈ {0.95, 1.0})",
            algo_order=e_order,
        )

        plot_bar_chart(
            df=summary_df,
            metric_mean_col="std_perf_auc_mean",
            metric_sem_col="std_perf_auc_sem",
            output_pdf=os.path.join(save_dir, "e0_variants_auc_performance_bars.pdf"),
            y_label="AUC Standardized Return",
            title="E(0) Formulations AUC Return (Tuned λ ∈ {0.95, 1.0})",
            algo_order=e_order,
        )

    print("All publication charts successfully generated!")


if __name__ == "__main__":
    main()
