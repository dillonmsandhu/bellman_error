"""
scripts/exact_random_policy_sweep/plot_results.py

Consolidated publication-quality plotting and reporting for Exact Random Policy Evaluation.
Generates:
1. Comparison Set A: MC vs TD(0) vs E(0)
   - Learning curves across all envs for k=8 and k=64 (ve_curves_mc_td0_e0.pdf)
   - Advantage cosine similarity curves (advantage_cossim_curves_mc_td0_e0.pdf)
   - Clustered AUC bar charts comparing algorithms across environments (auc_ve_bars_mc_td0_e0.pdf)
2. Comparison Set B: Full Spectrum (MC, TD(0), E(0), TD(lambda=0.8), E(lambda=0.8))
   - Learning curves across all envs for k=8 and k=64 (ve_curves_all_algos.pdf)
   - Advantage cosine similarity curves (advantage_cossim_curves_all_algos.pdf)
   - Full AUC bar charts (auc_ve_bars_all_algos.pdf)
3. Capacity Comparison (k=8 vs k=64):
   - Capacity effect plot (k_capacity_comparison.pdf)
4. Comprehensive Summary Tables:
   - summary_table_k8.csv
   - summary_table_k64.csv
   - summary_table_all.csv
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import pickle
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

ALGO_COLORS = {
    "mc": "#1f77b4",          # Steel Blue
    "td0": "#ff7f0e",         # Dark Orange
    "e0": "#2ca02c",          # Forest Green
    "td_lambda": "#d62728",   # Crimson Red
    "e_lambda": "#9467bd",    # Purple
}

ALGO_LABELS = {
    "mc": "MC (Exact)",
    "td0": "TD(0) [Exact]",
    "e0": "E(0) [Exact]",
    "td_lambda": "TD(λ=0.8) [Exact]",
    "e_lambda": "E(λ=0.8) [Exact]",
}

ENV_TITLES = {
    "fourrooms-dense": "FourRooms (Dense)",
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
    parser = argparse.ArgumentParser(description="Generate Exact Random Policy Sweep Publication Plots")
    parser.add_argument("--results_dir", type=str, default="results/exact_random_policy_sweep", help="Results directory root")
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

        data_k = env_data[env_name].get("data", {}).get(k_val, {})

        for algo_key in algo_subset:
            if algo_key not in data_k:
                continue

            lrs_dict = data_k[algo_key]
            if not lrs_dict:
                continue

            # Pick best LR based on minimum mean final value or AUC of VE
            best_lr = None
            best_val = float("inf")
            for lr, metrics in lrs_dict.items():
                arr = metrics.get("nn_weighted_VE", None)
                if arr is not None:
                    final_m = np.nanmean(arr[:, -1])
                    if final_m < best_val:
                        best_val = final_m
                        best_lr = lr

            if best_lr is None:
                best_lr = list(lrs_dict.keys())[0]

            traj = lrs_dict[best_lr].get(metric_key, None)
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
            ax.set_xlabel("Updates", fontsize=11)

    # Global legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(algo_subset), frameon=True, fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] {out_pdf_path}")


def plot_auc_grouped_bars(
    summary_df: pd.DataFrame,
    algo_subset: list[str],
    k_val: int,
    out_pdf_path: str,
    title_suffix: str = "",
):
    """Plots clustered bar chart of Normalized AUC of Value Error across environments."""
    if summary_df.empty:
        return

    sub_df = summary_df[(summary_df["k"] == k_val) & (summary_df["algo"].isin(algo_subset))].copy()
    if sub_df.empty:
        return

    envs_present = [e for e in ALL_ENVS if e in sub_df["env"].unique()]
    n_envs = len(envs_present)
    n_algos = len(algo_subset)

    if n_envs == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n_envs * 1.5), 6))

    x = np.arange(n_envs)
    total_width = 0.8
    bar_width = total_width / n_algos

    for i, algo in enumerate(algo_subset):
        algo_data = sub_df[sub_df["algo"] == algo].set_index("env")
        means = [algo_data.loc[e, "auc_ve_mean"] if e in algo_data.index else 0.0 for e in envs_present]
        sems = [algo_data.loc[e, "auc_ve_sem"] if e in algo_data.index else 0.0 for e in envs_present]

        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=sems,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#555555"),
            edgecolor="#222222",
            linewidth=0.8,
            capsize=3,
            alpha=0.9,
        )

    ax.set_ylabel("Normalized AUC of Value Error (Log Scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"Random Policy Evaluation: Normalized AUC of Value Error (k={k_val}) {title_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs_present], rotation=30, ha="right", fontsize=10)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] {out_pdf_path}")


def plot_k_capacity_effect(
    summary_df: pd.DataFrame,
    algo_subset: list[str],
    out_pdf_path: str,
):
    """Plots the effect of bottleneck feature capacity (k=8 vs k=64) on Value Error AUC."""
    if summary_df.empty:
        return

    sub_df = summary_df[summary_df["algo"].isin(algo_subset)].copy()
    if sub_df.empty:
        return

    envs_present = [e for e in ALL_ENVS if e in sub_df["env"].unique()]
    if not envs_present:
        return

    # Create subplots for k=8 vs k=64 side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, k_val in [(ax1, 8), (ax2, 64)]:
        df_k = sub_df[sub_df["k"] == k_val].copy()
        n_envs = len(envs_present)
        n_algos = len(algo_subset)
        x = np.arange(n_envs)
        total_width = 0.8
        bar_width = total_width / n_algos

        for i, algo in enumerate(algo_subset):
            algo_data = df_k[df_k["algo"] == algo].set_index("env")
            means = [algo_data.loc[e, "auc_ve_mean"] if e in algo_data.index else 0.0 for e in envs_present]
            sems = [algo_data.loc[e, "auc_ve_sem"] if e in algo_data.index else 0.0 for e in envs_present]

            offset = (i - n_algos / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                means,
                bar_width,
                yerr=sems,
                label=ALGO_LABELS.get(algo, algo) if k_val == 8 else None,
                color=ALGO_COLORS.get(algo, "#555555"),
                edgecolor="#222222",
                linewidth=0.8,
                capsize=3,
                alpha=0.9,
            )

        ax.set_title(f"Feature Dimension k={k_val} {'(Bottleneck Capacity)' if k_val==8 else '(High Capacity)'}", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs_present], rotation=35, ha="right", fontsize=9)
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

    ax1.set_ylabel("Normalized AUC of Value Error (Log Scale)", fontsize=11)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(algo_subset), frameon=True, fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] {out_pdf_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir or results_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 75)
    print("Generating Consolidated Publication Plots for Exact Random Policy Sweep")
    print(f"Results Directory:  {results_dir}")
    print(f"Save Directory:     {save_dir}")
    print("=" * 75)

    env_data, summary_df = load_all_env_data(results_dir)

    if not env_data and summary_df.empty:
        print(f"[Notice] No experimental results found in {results_dir}. Skipping plot generation until runs finish.")
        return

    # Save aggregated CSV tables
    if not summary_df.empty:
        all_csv = os.path.join(save_dir, "summary_table_all.csv")
        summary_df.to_csv(all_csv, index=False)
        print(f"[Saved] {all_csv}")

        df_k8 = summary_df[summary_df["k"] == 8]
        if not df_k8.empty:
            k8_csv = os.path.join(save_dir, "summary_table_k8.csv")
            df_k8.to_csv(k8_csv, index=False)
            print(f"[Saved] {k8_csv}")

        df_k64 = summary_df[summary_df["k"] == 64]
        if not df_k64.empty:
            k64_csv = os.path.join(save_dir, "summary_table_k64.csv")
            df_k64.to_csv(k64_csv, index=False)
            print(f"[Saved] {k64_csv}")

    # Set A: Core Triad (MC vs TD(0) vs E(0))
    core_algos = ["mc", "td0", "e0"]
    # Set B: Full Spectrum
    all_algos = ["mc", "td0", "e0", "td_lambda", "e_lambda"]

    for k_val in [8, 64]:
        # Set A Plots
        plot_learning_curves_grid(
            env_data,
            algo_subset=core_algos,
            k_val=k_val,
            metric_key="nn_weighted_VE",
            metric_label="Weighted Value Error",
            out_pdf_path=os.path.join(save_dir, f"ve_curves_mc_td0_e0_k{k_val}.pdf"),
            log_scale=True,
        )
        plot_learning_curves_grid(
            env_data,
            algo_subset=core_algos,
            k_val=k_val,
            metric_key="nn_advantage_cossim",
            metric_label="Advantage Cosine Similarity",
            out_pdf_path=os.path.join(save_dir, f"advantage_cossim_curves_mc_td0_e0_k{k_val}.pdf"),
            log_scale=False,
        )
        plot_auc_grouped_bars(
            summary_df,
            algo_subset=core_algos,
            k_val=k_val,
            out_pdf_path=os.path.join(save_dir, f"auc_ve_bars_mc_td0_e0_k{k_val}.pdf"),
            title_suffix="(MC vs TD(0) vs E(0))",
        )

        # Set B Plots (Full Spectrum)
        plot_learning_curves_grid(
            env_data,
            algo_subset=all_algos,
            k_val=k_val,
            metric_key="nn_weighted_VE",
            metric_label="Weighted Value Error",
            out_pdf_path=os.path.join(save_dir, f"ve_curves_all_algos_k{k_val}.pdf"),
            log_scale=True,
        )
        plot_learning_curves_grid(
            env_data,
            algo_subset=all_algos,
            k_val=k_val,
            metric_key="nn_advantage_cossim",
            metric_label="Advantage Cosine Similarity",
            out_pdf_path=os.path.join(save_dir, f"advantage_cossim_curves_all_algos_k{k_val}.pdf"),
            log_scale=False,
        )
        plot_auc_grouped_bars(
            summary_df,
            algo_subset=all_algos,
            k_val=k_val,
            out_pdf_path=os.path.join(save_dir, f"auc_ve_bars_all_algos_k{k_val}.pdf"),
            title_suffix="(All Algorithms)",
        )

    # Set C: Capacity Effect Comparison (k=8 vs k=64)
    plot_k_capacity_effect(
        summary_df,
        algo_subset=core_algos,
        out_pdf_path=os.path.join(save_dir, "k_capacity_effect_mc_td0_e0.pdf"),
    )
    plot_k_capacity_effect(
        summary_df,
        algo_subset=all_algos,
        out_pdf_path=os.path.join(save_dir, "k_capacity_effect_all_algos.pdf"),
    )

    print("\n[Done] All publication-ready PDFs and summary tables exported successfully.")


if __name__ == "__main__":
    main()
