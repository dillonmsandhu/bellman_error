"""
scripts/sampled_E_aux_random_sweep/plot_results.py

Consolidated publication-quality plotting and reporting for Sampled E-Aux Random Policy Evaluation.
Generates:
1. Learning curves across all envs for each k:
   - Value Error curves (ve_curves_all_algos_k{8,64}.pdf)
   - Advantage Cosine Similarity curves (advantage_cossim_curves_all_algos_k{8,64}.pdf)
2. Clustered bar charts across all envs:
   - AUC of Weighted Value Error (auc_ve_bars_all_algos_k{8,64}.pdf)
   - Final Weighted Value Error (final_ve_bars_all_algos_k{8,64}.pdf)
   - Advantage Cosine Similarity AUC (auc_cossim_bars_all_algos_k{8,64}.pdf)
   - Final Normalized Greedy Return (final_greedy_bars_all_algos_k{8,64}.pdf)
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
    "mc": "#1f77b4",             # Steel Blue
    "oct8_biased": "#d62728",    # Crimson / Red
    "v_aux_full": "#9467bd",     # Purple
    "v_aux_bilinear": "#ff7f0e", # Dark Orange
    "v_aux_surgical": "#2ca02c", # Forest Green
}

ALGO_LABELS = {
    "mc": "MC (Sampled)",
    "oct8_biased": "OG Oct 8 (Biased)",
    "v_aux_full": "Full v_aux (Oct 8)",
    "v_aux_bilinear": "Bilinear v_aux",
    "v_aux_surgical": "Surgical v_aux",
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
    parser = argparse.ArgumentParser(description="Generate Sampled E-Aux Random Policy Sweep Publication Plots")
    parser.add_argument("--results_dir", type=str, default="results/sampled_E_aux_random_sweep", help="Results directory root")
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

    algos = ["mc", "oct8_biased", "v_aux_full", "v_aux_bilinear", "v_aux_surgical"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

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


def export_summary_tables(combined_df: pd.DataFrame, out_dir: str):
    """Exports clean CSV summary tables formatted for paper inclusion."""
    if combined_df.empty:
        return

    cols = [
        "env_name",
        "k",
        "algo",
        "display_name",
        "best_critic_lr",
        "skew_symmetry_ratio",
        "auc_weighted_ve_mean",
        "auc_weighted_ve_sem",
        "final_weighted_ve_mean",
        "final_weighted_ve_sem",
        "auc_advantage_cossim_mean",
        "auc_advantage_cossim_sem",
        "final_normalized_greedy_perf",
    ]
    present_cols = [c for c in cols if c in combined_df.columns]
    df_clean = combined_df[present_cols].copy()

    df_clean.to_csv(os.path.join(out_dir, "summary_table_all.csv"), index=False)

    for k_val in combined_df["k"].unique():
        df_k = df_clean[df_clean["k"] == k_val]
        df_k.to_csv(os.path.join(out_dir, f"summary_table_k{k_val}.csv"), index=False)

    print(f"Exported summary tables to: {out_dir}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else results_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading experiment data from: {results_dir}")
    env_data, combined_df = load_all_env_data(results_dir)

    all_algos = ["mc", "oct8_biased", "v_aux_full", "v_aux_bilinear", "v_aux_surgical"]

    # Detect available k values
    if not combined_df.empty:
        k_values = sorted(combined_df["k"].unique())
    else:
        k_values = [8, 64]

    for k in k_values:
        # 1. Learning curves
        plot_learning_curves_grid(
            env_data=env_data,
            algo_subset=all_algos,
            k_val=k,
            metric_key="nn_weighted_VE",
            metric_label="Weighted Value Error (log scale)",
            out_pdf_path=os.path.join(save_dir, f"ve_curves_all_algos_k{k}.pdf"),
            log_scale=True,
        )
        plot_learning_curves_grid(
            env_data=env_data,
            algo_subset=all_algos,
            k_val=k,
            metric_key="nn_advantage_cossim",
            metric_label="Advantage Cosine Similarity",
            out_pdf_path=os.path.join(save_dir, f"advantage_cossim_curves_all_algos_k{k}.pdf"),
            log_scale=False,
        )

        # 2. Clustered bar charts
        plot_clustered_bar_chart(
            combined_df=combined_df,
            algo_subset=all_algos,
            k_val=k,
            metric_col_mean="auc_weighted_ve_mean",
            metric_col_sem="auc_weighted_ve_sem",
            metric_label="Normalized AUC of Value Error",
            out_pdf_path=os.path.join(save_dir, f"auc_ve_bars_all_algos_k{k}.pdf"),
            log_scale=True,
        )
        plot_clustered_bar_chart(
            combined_df=combined_df,
            algo_subset=all_algos,
            k_val=k,
            metric_col_mean="final_weighted_ve_mean",
            metric_col_sem="final_weighted_ve_sem",
            metric_label="Final Weighted Value Error",
            out_pdf_path=os.path.join(save_dir, f"final_ve_bars_all_algos_k{k}.pdf"),
            log_scale=True,
        )
        plot_clustered_bar_chart(
            combined_df=combined_df,
            algo_subset=all_algos,
            k_val=k,
            metric_col_mean="auc_advantage_cossim_mean",
            metric_col_sem="auc_advantage_cossim_sem",
            metric_label="AUC Advantage Cosine Similarity",
            out_pdf_path=os.path.join(save_dir, f"auc_cossim_bars_all_algos_k{k}.pdf"),
            log_scale=False,
        )
        plot_clustered_bar_chart(
            combined_df=combined_df,
            algo_subset=all_algos,
            k_val=k,
            metric_col_mean="final_normalized_greedy_perf",
            metric_col_sem="final_greedy_perf_sem",
            metric_label="Final Normalized Policy Return",
            out_pdf_path=os.path.join(save_dir, f"final_greedy_bars_all_algos_k{k}.pdf"),
            log_scale=False,
        )

    # 3. Capacity scaling plot
    plot_capacity_comparison(
        combined_df=combined_df,
        out_pdf_path=os.path.join(save_dir, "k_capacity_effect.pdf"),
    )

    # 4. Summary tables
    export_summary_tables(combined_df, save_dir)
    print("All plotting routines finished successfully.")


if __name__ == "__main__":
    main()
