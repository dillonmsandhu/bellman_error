"""
scripts/exact_fixed_policy_sweep/plot_results.py

Consolidated publication-quality plotting and reporting for Exact Fixed Policy Evaluation.
Generates:
1. Comparison Set A: MC vs TD(0) vs E(0)
   - Learning curves across all envs for k=8 and k=64 (ve_curves_mc_td0_e0.pdf)
   - Advantage cosine similarity curves (advantage_cossim_curves_mc_td0_e0.pdf)
   - Clustered AUC bar charts for Value Error (auc_ve_bars_mc_td0_e0_k{8,64}.pdf)
   - Clustered AUC bar charts for Advantage Cosine Sim (auc_cossim_bars_mc_td0_e0_k{8,64}.pdf)
   - Clustered AUC bar charts for Greedy Policy V_start (auc_vstart_bars_mc_td0_e0_k{8,64}.pdf)
2. Comparison Set B: Full Spectrum (MC, TD(0), E(0), TD(lambda=0.8), E(lambda=0.8))
   - Learning curves across all envs for k=8 and k=64 (ve_curves_all_algos.pdf)
   - Advantage cosine similarity curves (advantage_cossim_curves_all_algos.pdf)
   - Full AUC bar charts for Value Error (auc_ve_bars_all_algos_k{8,64}.pdf)
   - Full AUC bar charts for Advantage Cosine Sim (auc_cossim_bars_all_algos_k{8,64}.pdf)
   - Full AUC bar charts for Greedy Policy V_start (auc_vstart_bars_all_algos_k{8,64}.pdf)
3. Capacity Comparison (k=8 vs k=64):
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
    "mountaincar": "Mountain Car (Sparse)",
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
    parser = argparse.ArgumentParser(description="Generate Exact Fixed Policy Sweep Publication Plots")
    parser.add_argument("--results_dir", type=str, default="results/exact_fixed_policy_sweep", help="Results directory root")
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

        data_k = env_data[env_name].get("data", {}).get(k_val, {})

        for algo_key in algo_subset:
            if algo_key not in data_k:
                continue

            lrs_dict = data_k[algo_key]
            if not lrs_dict:
                continue

            # Pick best LR based on minimum mean final value
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
    metric_col: str,
    sem_col: str | None,
    ylabel: str,
    title: str,
    out_pdf_path: str,
    log_scale: bool = False,
    baseline_dict: dict | None = None,
):
    """Plots clustered bar chart of a metric across environments for specified algorithms and k."""
    if summary_df.empty:
        return

    sub_df = summary_df[(summary_df["k"] == k_val) & (summary_df["algo"].isin(algo_subset))].copy()
    if sub_df.empty or metric_col not in sub_df.columns:
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
        means = []
        sems = []
        for e in envs_present:
            val = float(algo_data.loc[e, metric_col]) if (e in algo_data.index and not np.isnan(algo_data.loc[e, metric_col])) else 0.0
            sem = float(algo_data.loc[e, sem_col]) if (sem_col and sem_col in algo_data.columns and e in algo_data.index and not np.isnan(algo_data.loc[e, sem_col])) else 0.0
            if baseline_dict and e in baseline_dict:
                v_star = baseline_dict[e].get("v_star", 1.0)
                v_base = baseline_dict[e].get("v_rand", 0.0)
                denom = v_star - v_base
                if abs(denom) > 1e-8:
                    val = (val - v_base) / denom
                    sem = sem / denom
            means.append(val)
            sems.append(sem)

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

    if baseline_dict:
        ax.axhline(1.0, color="#d9534f", linestyle="--", linewidth=1.2, alpha=0.8, label=r"Optimal Policy ($V^*$)")
        ax.axhline(0.0, color="#777777", linestyle=":", linewidth=1.2, alpha=0.8, label=r"Random Policy ($V^{\pi}$)")
        ax.set_ylim(-0.08, 1.15)

    ax.set_ylabel(ylabel, fontsize=12)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs_present], rotation=30, ha="right", fontsize=10)
    ax.grid(True, which="both" if log_scale else "major", axis="y", linestyle="--", alpha=0.35)
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

ENV_COLORS = {
    "FourRooms-misc": "#1f77b4",
    "fourrooms-dense": "#aec7e8",
    "fourrooms-mines": "#d62728",
    "fourrooms-mines-dense": "#ff9896",
    "eightrooms-misc": "#2ca02c",
    "eightrooms-dense": "#98df8a",
    "whirlpool-misc": "#9467bd",
    "MountainCar-v0": "#ff7f0e",
    "mountaincar": "#ff7f0e",
    "mountaincar-dense": "#ffbb78",
    "SpaceInvadersExactValue": "#8c564b",
}


def plot_skew_symmetry_scatter(
    summary_df: pd.DataFrame,
    out_pdf_path: str,
):
    """
    Plots scatter of Skew-Symmetry (||K|| / ||S||) vs. Ratio of AUC Value Error (E(0) / TD(0)).
    Compares k=8 and k=64 side-by-side.
    """
    if summary_df.empty:
        return

    required = ["env", "k", "algo", "auc_ve_mean"]
    if not all(c in summary_df.columns for c in required):
        return

    e0_df = summary_df[summary_df["algo"] == "e0"].set_index(["env", "k"])
    td0_df = summary_df[summary_df["algo"] == "td0"].set_index(["env", "k"])

    common_indices = e0_df.index.intersection(td0_df.index)
    if len(common_indices) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, k_val in [(ax1, 8), (ax2, 64)]:
        pts_x = []
        pts_y = []
        labels = []
        envs = []

        for env_name in ALL_ENVS:
            idx = (env_name, k_val)
            if idx not in common_indices:
                continue

            e0_auc = float(e0_df.loc[idx, "auc_ve_mean"])
            td0_auc = float(td0_df.loc[idx, "auc_ve_mean"])
            skew = e0_df.loc[idx, "k_over_s_ratio"] if "k_over_s_ratio" in e0_df.columns else np.nan

            if np.isnan(e0_auc) or np.isnan(td0_auc) or td0_auc <= 0 or np.isnan(skew):
                continue

            ratio = e0_auc / td0_auc
            pts_x.append(float(skew))
            pts_y.append(float(ratio))
            labels.append(ENV_TITLES.get(env_name, env_name))
            envs.append(env_name)

        if not pts_x:
            ax.text(0.5, 0.5, "Pending Data", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(f"k = {k_val}", fontsize=12, fontweight="bold")
            continue

        pts_x = np.array(pts_x)
        pts_y = np.array(pts_y)

        # Plot parity reference line
        ax.axhline(1.0, color="#d9534f", linestyle="--", linewidth=1.5, alpha=0.8, label=r"Parity ($E(0) = \mathrm{TD}(0)$)")
        # Shade region where E(0) wins
        y_min_vis = max(0.0, float(np.min(pts_y) * 0.8))
        ax.axhspan(y_min_vis, 1.0, color="#5cb85c", alpha=0.08, label=r"$E(0)$ beats $\mathrm{TD}(0)$ ($y < 1.0$)")

        # Scatter points with custom colors and labels
        for x_val, y_val, lab, env_n in zip(pts_x, pts_y, labels, envs):
            color = ENV_COLORS.get(env_n, "#1f77b4")
            ax.scatter(x_val, y_val, s=90, color=color, edgecolors="#222222", linewidth=1.2, zorder=5)
            ax.annotate(
                lab,
                (x_val, y_val),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
                alpha=0.9,
            )

        # Linear regression trendline if enough points
        if len(pts_x) >= 3:
            slope, intercept = np.polyfit(pts_x, pts_y, 1)
            x_line = np.linspace(float(np.min(pts_x) * 0.9), float(np.max(pts_x) * 1.1), 100)
            ax.plot(x_line, slope * x_line + intercept, color="#333333", linestyle="-.", linewidth=1.4, alpha=0.7, label=f"Trend (slope={slope:+.2f})")

        ax.set_title(f"Feature Dimension k={k_val} {'(Bottleneck)' if k_val==8 else '(High Capacity)'}", fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Degree of Skew Symmetry ($\|K\|_F / \|S\|_F$)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True, fontsize=9, loc="upper right")

    ax1.set_ylabel(r"$\mathrm{AUC\ VE}(E(0))\;/\;\mathrm{AUC\ VE}(\mathrm{TD}(0))$", fontsize=11)
    fig.suptitle(r"Impact of Markov Chain Skew-Symmetry on $E(0)$ vs. $\mathrm{TD}(0)$ Performance", fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] {out_pdf_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir or results_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 75)
    print("Generating Consolidated Publication Plots for Exact Fixed Policy Sweep")
    print(f"Results Directory:  {results_dir}")
    print(f"Save Directory:     {save_dir}")
    print("=" * 75)

    env_data, summary_df = load_all_env_data(results_dir)
    baselines = load_env_baselines()
    if baselines:
        print(f"[Baselines Loaded] Found reference baselines for {len(baselines)} environments.")

    if not env_data and summary_df.empty:
        print(f"[Notice] No experimental results found in {results_dir}. Skipping plot generation until runs finish.")
        return

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

    core_algos = ["mc", "td0", "e0"]
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
        # Set A AUC Bar Charts: Value Error, Advantage Cosine Sim, and V_start Return
        plot_auc_grouped_bars(
            summary_df,
            algo_subset=core_algos,
            k_val=k_val,
            metric_col="auc_ve_mean",
            sem_col="auc_ve_sem",
            ylabel="Normalized AUC of Value Error (Log Scale)",
            title=f"Fixed Policy Evaluation: Normalized AUC of Value Error (k={k_val}) [MC vs TD(0) vs E(0)]",
            out_pdf_path=os.path.join(save_dir, f"auc_ve_bars_mc_td0_e0_k{k_val}.pdf"),
            log_scale=True,
        )
        plot_auc_grouped_bars(
            summary_df,
            algo_subset=core_algos,
            k_val=k_val,
            metric_col="auc_cossim_mean",
            sem_col="auc_cossim_sem",
            ylabel="Normalized AUC of Advantage Cosine Sim",
            title=f"Fixed Policy Evaluation: Normalized AUC of Advantage Cosine Sim (k={k_val}) [MC vs TD(0) vs E(0)]",
            out_pdf_path=os.path.join(save_dir, f"auc_cossim_bars_mc_td0_e0_k{k_val}.pdf"),
            log_scale=False,
        )
        plot_auc_grouped_bars(
            summary_df,
            algo_subset=core_algos,
            k_val=k_val,
            metric_col="auc_greedy_perf",
            sem_col="auc_greedy_perf_sem",
            ylabel=r"Normalized Return Score (0 = Random $V^{\pi}$, 1 = Optimal $V^*$)",
            title=f"Fixed Policy Evaluation: Normalized Return Score (k={k_val}) [MC vs TD(0) vs E(0)]",
            out_pdf_path=os.path.join(save_dir, f"auc_vstart_bars_mc_td0_e0_k{k_val}.pdf"),
            log_scale=False,
            baseline_dict=baselines,
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
        # Set B AUC Bar Charts: Value Error, Advantage Cosine Sim, and V_start Return
        plot_auc_grouped_bars(
            summary_df,
            algo_subset=all_algos,
            k_val=k_val,
            metric_col="auc_ve_mean",
            sem_col="auc_ve_sem",
            ylabel="Normalized AUC of Value Error (Log Scale)",
            title=f"Fixed Policy Evaluation: Normalized AUC of Value Error (k={k_val}) [All Algorithms]",
            out_pdf_path=os.path.join(save_dir, f"auc_ve_bars_all_algos_k{k_val}.pdf"),
            log_scale=True,
        )
        plot_auc_grouped_bars(
            summary_df,
            algo_subset=all_algos,
            k_val=k_val,
            metric_col="auc_cossim_mean",
            sem_col="auc_cossim_sem",
            ylabel="Normalized AUC of Advantage Cosine Sim",
            title=f"Fixed Policy Evaluation: Normalized AUC of Advantage Cosine Sim (k={k_val}) [All Algorithms]",
            out_pdf_path=os.path.join(save_dir, f"auc_cossim_bars_all_algos_k{k_val}.pdf"),
            log_scale=False,
        )
        plot_auc_grouped_bars(
            summary_df,
            algo_subset=all_algos,
            k_val=k_val,
            metric_col="auc_greedy_perf",
            sem_col="auc_greedy_perf_sem",
            ylabel=r"Normalized Return Score (0 = Random $V^{\pi}$, 1 = Optimal $V^*$)",
            title=f"Fixed Policy Evaluation: Normalized Return Score (k={k_val}) [All Algorithms]",
            out_pdf_path=os.path.join(save_dir, f"auc_vstart_bars_all_algos_k{k_val}.pdf"),
            log_scale=False,
            baseline_dict=baselines,
        )

    # Set C: Capacity Effect
    plot_k_capacity_effect(
        summary_df,
        algo_subset=core_algos,
        out_pdf_path=os.path.join(save_dir, "k_capacity_effect_mc_td0_e0.pdf"),
    )
    # Set D: Skew-Symmetry Diagnostic Scatter (||K|| / ||S|| vs E(0) / TD(0))
    plot_skew_symmetry_scatter(
        summary_df,
        out_pdf_path=os.path.join(save_dir, "skew_symmetry_vs_e0_td0_scatter.pdf"),
    )

    print("\n[Done] All publication-ready PDFs and summary tables exported successfully.")


if __name__ == "__main__":
    main()
