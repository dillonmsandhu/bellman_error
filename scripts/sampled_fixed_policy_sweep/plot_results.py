"""
scripts/sampled_fixed_policy_sweep/plot_results.py

Consolidated publication-quality plotting and reporting for Sampled Fixed Policy Evaluation.
Generates:
1. Comparison Set A: MC vs TD(0) vs E(0)
   - Learning curves across all envs for k=8 and k=64 (ve_curves_mc_td0_e0_k{8,64}.pdf)
   - Advantage cosine similarity curves (advantage_cossim_curves_mc_td0_e0_k{8,64}.pdf)
   - Clustered AUC bar charts for Value Error (auc_ve_bars_mc_td0_e0_k{8,64}.pdf)
   - Clustered AUC bar charts for Advantage Cosine Sim (auc_cossim_bars_mc_td0_e0_k{8,64}.pdf)
   - Clustered bar charts for Final Normalized Greedy Return (final_greedy_bars_mc_td0_e0_k{8,64}.pdf)
2. Comparison Set B: Full Spectrum (MC, TD(0), E(0), TD(lambda=0.8), E(lambda=0.8))
   - Learning curves across all envs for k=8 and k=64 (ve_curves_all_algos_k{8,64}.pdf)
   - Advantage cosine similarity curves (advantage_cossim_curves_all_algos_k{8,64}.pdf)
   - Full AUC bar charts for Value Error (auc_ve_bars_all_algos_k{8,64}.pdf)
   - Full AUC bar charts for Advantage Cosine Sim (auc_cossim_bars_all_algos_k{8,64}.pdf)
   - Full bar charts for Final Normalized Greedy Return (final_greedy_bars_all_algos_k{8,64}.pdf)
3. Capacity Scaling Comparison (k=8 vs k=64):
   - Capacity effect plot (k_capacity_effect.pdf)
4. Comprehensive Summary Tables:
   - summary_table_k8.csv
   - summary_table_k64.csv
   - summary_table_all.csv
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
    "mc": "#1f77b4",          # Steel Blue
    "td0": "#ff7f0e",         # Dark Orange
    "e0_sep14": "#2ca02c",    # Forest Green
    "e0_oct5": "#17becf",     # Cyan / Teal
    "e0_oct8": "#9467bd",     # Purple
}

ALGO_LABELS = {
    "mc": "MC (Sampled)",
    "td0": "TD(0) [Sampled]",
    "e0_sep14": "E(0) [Sept 14 / Uncorrected]",
    "e0_oct5": "E(0) [Oct 5 / Corrected]",
    "e0_oct8": "E(0) [Oct 8 / Start-Anchored]",
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
    parser = argparse.ArgumentParser(description="Generate Sampled Fixed Policy Sweep Publication Plots")
    parser.add_argument("--results_dir", type=str, default="results/sampled_fixed_policy_sweep", help="Results directory root")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory (default: same as results_dir)")
    return parser.parse_args()


def load_all_env_data(results_dir: str):
    """Loads out.pkl and summary_auc.csv for all available environments."""
    env_data = {}
    summary_dfs = []

    for env_name in ALL_ENVS:
        env_dir = os.path.join(results_dir, env_name)
        pkl_path = os.path.join(env_dir, "out.pkl")
        csv_path = os.path.join(env_dir, "summary_auc.csv")

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    env_data[env_name] = pickle.load(f)
            except Exception as e:
                print(f"[Warning] Error loading {pkl_path}: {e}")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                summary_dfs.append(df)
            except Exception as e:
                print(f"[Warning] Error loading {csv_path}: {e}")

    combined_df = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
    return env_data, combined_df


def load_env_baselines(project_root: str | None = None) -> dict:
    """Loads precomputed environment baselines (v_star, v_rand, delta) from JSON."""
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    json_path = os.path.join(project_root, "reference", "env_performance_baselines.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load {json_path}: {e}")
    return {}


def plot_learning_curves_grid(
    env_data: dict,
    algo_subset: list[str],
    k_val: int,
    metric_key: str,
    metric_label: str,
    out_pdf_path: str,
    log_scale: bool = True,
):
    """Plots a 3x3 grid of learning curves across all 9 environments for a specific k and metric."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        if env_name not in env_data:
            ax.text(0.5, 0.5, "Data Pending", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        res_by_algo = env_data[env_name].get("results_by_algo", {})

        for algo_key in algo_subset:
            if algo_key not in res_by_algo:
                continue

            k_dict = res_by_algo[algo_key].get(k_val, {})
            if not k_dict:
                continue

            traj = k_dict.get("metrics", {}).get(metric_key, None)
            if traj is not None:
                mean_curve = np.nanmean(traj, axis=0)
                sem_curve = np.nanstd(traj, axis=0) / np.sqrt(max(1, traj.shape[0]))
                steps = np.arange(len(mean_curve))

                color = ALGO_COLORS.get(algo_key, "#333333")
                label = ALGO_LABELS.get(algo_key, algo_key)

                ax.plot(steps, mean_curve, color=color, label=label, linewidth=1.8)
                ax.fill_between(steps, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.18)

        if log_scale and metric_key == "nn_weighted_VE":
            ax.set_yscale("log")

        if idx % 3 == 0:
            ax.set_ylabel(metric_label, fontsize=11)
        if idx >= 6:
            ax.set_xlabel("Rollout Updates", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(algo_subset),
            bbox_to_anchor=(0.5, 0.02),
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=12,
        )

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"Generated learning curve grid: {out_pdf_path}")


def plot_clustered_bar_chart(
    combined_df: pd.DataFrame,
    algo_subset: list[str],
    k_val: int,
    metric_col_mean: str,
    metric_col_sem: str,
    metric_label: str,
    out_pdf_path: str,
    higher_is_better: bool = False,
    log_scale: bool = False,
):
    """Generates a grouped/clustered bar chart across all 9 environments for a given metric."""
    if combined_df.empty:
        return

    sub_df = combined_df[combined_df["k"] == k_val].copy()
    if sub_df.empty:
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    n_envs = len(ALL_ENVS)
    n_algos = len(algo_subset)
    bar_width = 0.8 / n_algos
    indices = np.arange(n_envs)

    for i, algo_key in enumerate(algo_subset):
        algo_data = sub_df[sub_df["algo"] == algo_key]
        means = []
        sems = []

        for env_name in ALL_ENVS:
            row = algo_data[algo_data["env_name"] == env_name]
            if not row.empty and metric_col_mean in row:
                means.append(row[metric_col_mean].values[0])
                sems.append(row[metric_col_sem].values[0] if metric_col_sem in row else 0.0)
            else:
                means.append(np.nan)
                sems.append(0.0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            indices + offset,
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

    ax.set_xticks(indices)
    env_labels = [ENV_TITLES.get(e, e).replace(" ", "\n") for e in ALL_ENVS]
    ax.set_xticklabels(env_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")

    if log_scale:
        ax.set_yscale("log")

    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    title_suffix = f" (k={k_val})"
    ax.set_title(f"{metric_label} across Environments{title_suffix}", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"Generated clustered bar chart: {out_pdf_path}")


def plot_capacity_comparison(
    combined_df: pd.DataFrame,
    out_pdf_path: str,
):
    """Compares bottleneck representation (k=8) vs high-capacity representation (k=64)."""
    if combined_df.empty:
        return

    k_vals = sorted(combined_df["k"].unique())
    if len(k_vals) < 2:
        return

    k_low, k_high = k_vals[0], k_vals[-1]
    df_low = combined_df[combined_df["k"] == k_low]
    df_high = combined_df[combined_df["k"] == k_high]

    algos = ["mc", "td0", "e0_sep14", "e0_oct5", "e0_oct8"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax_idx, (k_val, df_k) in enumerate([(k_low, df_low), (k_high, df_high)]):
        ax = axes[ax_idx]
        ax.set_title(f"Representation Capacity k={k_val}", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35, axis="y")

        n_envs = len(ALL_ENVS)
        n_algos = len(algos)
        bar_width = 0.8 / n_algos
        indices = np.arange(n_envs)

        for i, algo_key in enumerate(algos):
            algo_data = df_k[df_k["algo"] == algo_key]
            means = []
            sems = []
            for env_name in ALL_ENVS:
                row = algo_data[algo_data["env_name"] == env_name]
                if not row.empty:
                    means.append(row["auc_weighted_ve_mean"].values[0])
                    sems.append(row["auc_weighted_ve_sem"].values[0])
                else:
                    means.append(np.nan)
                    sems.append(0.0)

            offset = (i - n_algos / 2 + 0.5) * bar_width
            ax.bar(
                indices + offset,
                means,
                yerr=sems,
                width=bar_width,
                label=ALGO_LABELS.get(algo_key, algo_key),
                color=ALGO_COLORS.get(algo_key, "#333333"),
                capsize=2,
                alpha=0.9,
            )

        ax.set_xticks(indices)
        env_labels = [ENV_TITLES.get(e, e).split(" (")[0].replace(" ", "\n") for e in ALL_ENVS]
        ax.set_xticklabels(env_labels, fontsize=9)
        ax.set_yscale("log")
        if ax_idx == 0:
            ax.set_ylabel("Normalized AUC of Value Error (log scale)", fontsize=11)

    axes[1].legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"Generated capacity scaling comparison: {out_pdf_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir if args.save_dir else results_dir)
    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 80)
    print("SAMPLED FIXED POLICY SWEEP - PLOTTING & REPORTING")
    print(f"Results Directory: {results_dir}")
    print(f"Output Directory:  {figures_dir}")
    print("=" * 80)

    env_data, combined_df = load_all_env_data(results_dir)
    if not env_data and combined_df.empty:
        print("[Error] No environment results found in directory.")
        return

    # Export combined summary CSVs
    if not combined_df.empty:
        combined_df.to_csv(os.path.join(save_dir, "summary_table_all.csv"), index=False)
        for k_val in combined_df["k"].unique():
            df_k = combined_df[combined_df["k"] == k_val]
            df_k.to_csv(os.path.join(save_dir, f"summary_table_k{k_val}.csv"), index=False)
        print("Exported aggregated summary CSV tables.")

    core_algos = ["mc", "td0", "e0_oct8"]
    e0_variants = ["e0_sep14", "e0_oct5", "e0_oct8"]
    full_algos = ["mc", "td0", "e0_sep14", "e0_oct5", "e0_oct8"]

    k_values = [8, 64] if not combined_df.empty else [8]
    if not combined_df.empty:
        k_values = sorted(combined_df["k"].unique())

    for k in k_values:
        print(f"\n--- Generating Plots for k={k} ---")
        
        # 1. Learning Curves - Core Comparison (MC, TD(0), E(0) [Oct 8])
        plot_learning_curves_grid(
            env_data,
            core_algos,
            k,
            "nn_weighted_VE",
            "Normalized Value Error",
            os.path.join(figures_dir, f"ve_curves_mc_td0_e0_oct8_k{k}.pdf"),
            log_scale=True,
        )
        plot_learning_curves_grid(
            env_data,
            core_algos,
            k,
            "nn_advantage_cossim",
            "Advantage Cosine Similarity",
            os.path.join(figures_dir, f"advantage_cossim_curves_mc_td0_e0_oct8_k{k}.pdf"),
            log_scale=False,
        )

        # 2. Learning Curves - E(0) Variants Comparison (Sept 14 vs Oct 5 vs Oct 8)
        plot_learning_curves_grid(
            env_data,
            e0_variants,
            k,
            "nn_weighted_VE",
            "Normalized Value Error",
            os.path.join(figures_dir, f"ve_curves_e0_variants_k{k}.pdf"),
            log_scale=True,
        )
        plot_learning_curves_grid(
            env_data,
            e0_variants,
            k,
            "nn_advantage_cossim",
            "Advantage Cosine Similarity",
            os.path.join(figures_dir, f"advantage_cossim_curves_e0_variants_k{k}.pdf"),
            log_scale=False,
        )

        # 3. Learning Curves - Full 5 Algorithms
        plot_learning_curves_grid(
            env_data,
            full_algos,
            k,
            "nn_weighted_VE",
            "Normalized Value Error",
            os.path.join(figures_dir, f"ve_curves_all_algos_k{k}.pdf"),
            log_scale=True,
        )
        plot_learning_curves_grid(
            env_data,
            full_algos,
            k,
            "nn_advantage_cossim",
            "Advantage Cosine Similarity",
            os.path.join(figures_dir, f"advantage_cossim_curves_all_algos_k{k}.pdf"),
            log_scale=False,
        )

        # 4. Clustered Bar Charts - AUC Value Error
        plot_clustered_bar_chart(
            combined_df,
            core_algos,
            k,
            "auc_weighted_ve_mean",
            "auc_weighted_ve_sem",
            "Normalized AUC Value Error",
            os.path.join(figures_dir, f"auc_ve_bars_mc_td0_e0_oct8_k{k}.pdf"),
            log_scale=True,
        )
        plot_clustered_bar_chart(
            combined_df,
            e0_variants,
            k,
            "auc_weighted_ve_mean",
            "auc_weighted_ve_sem",
            "Normalized AUC Value Error",
            os.path.join(figures_dir, f"auc_ve_bars_e0_variants_k{k}.pdf"),
            log_scale=True,
        )
        plot_clustered_bar_chart(
            combined_df,
            full_algos,
            k,
            "auc_weighted_ve_mean",
            "auc_weighted_ve_sem",
            "Normalized AUC Value Error",
            os.path.join(figures_dir, f"auc_ve_bars_all_algos_k{k}.pdf"),
            log_scale=True,
        )

        # 5. Clustered Bar Charts - Advantage Cosine Sim
        plot_clustered_bar_chart(
            combined_df,
            full_algos,
            k,
            "auc_advantage_cossim_mean",
            "auc_advantage_cossim_sem",
            "Advantage Cosine Similarity (AUC)",
            os.path.join(figures_dir, f"auc_cossim_bars_all_algos_k{k}.pdf"),
            log_scale=False,
        )

        # 6. Clustered Bar Charts - Final Normalized Greedy Return
        plot_clustered_bar_chart(
            combined_df,
            full_algos,
            k,
            "final_normalized_greedy_perf",
            "final_greedy_perf_sem",
            "Final Normalized Greedy Return (0=Rand, 1=Opt)",
            os.path.join(figures_dir, f"final_greedy_bars_all_algos_k{k}.pdf"),
            higher_is_better=True,
            log_scale=False,
        )

    # 6. Capacity comparison
    plot_capacity_comparison(combined_df, os.path.join(figures_dir, "k_capacity_effect.pdf"))

    print(f"\nAll publication plots successfully generated and saved to: {figures_dir}")


if __name__ == "__main__":
    main()
