"""
scripts/sampled_lambda_sweep/plot_results.py

Generates publication-quality vector PDF charts for Sampled PPO Lambda Sweep (TD(λ) vs E(λ) with v_aux):
1. matching_lambda_comparison_curves.pdf:
   - 3x3 grid across all 9 environments.
   - Compares E(λ) vs TD(λ) for matching λ in {0.5, 0.9, 0.95}.
   - Standardized policy performance (V_start - V_rand) / (V* - V_rand) with ±1 SEM shading.
2. standardized_performance_learning_curves.pdf:
   - Complete 6-condition learning trajectories across all 9 environments.
3. lambda_sensitivity_curves.pdf:
   - Performance (AUC & Final) plotted against λ ∈ [0.5, 0.9, 0.95] for TD vs E.
4. delta_e_minus_td_bars.pdf:
   - Head-to-head performance delta: Δ = Perf(E(λ)) - Perf(TD(λ)) for each λ and environment.
5. auc_standardized_performance_bars.pdf & final_standardized_performance_bars.pdf:
   - Grouped bar charts with ±1 SEM error bars.
6. value_error_learning_curves.pdf:
   - Critic Value Error (nn_weighted_VE) across rollouts.
7. advantage_cossim_learning_curves.pdf:
   - Advantage Cosine Similarity (nn_advantage_cossim) across rollouts.
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

CONDITION_COLORS = {
    "td_0.50": "#fdae6b",   # Light orange
    "td_0.90": "#e6550d",   # Medium orange
    "td_0.95": "#8c2d04",   # Dark rust orange
    "e_0.50":  "#a1d99b",   # Light green
    "e_0.90":  "#31a354",   # Medium green
    "e_0.95":  "#00441b",   # Deep dark green
}

CONDITION_LABELS = {
    "td_0.50": "TD(0.50)",
    "td_0.90": "TD(0.90)",
    "td_0.95": "TD(0.95)",
    "e_0.50":  "E(0.50) [v_aux]",
    "e_0.90":  "E(0.90) [v_aux]",
    "e_0.95":  "E(0.95) [v_aux]",
}

MATCHING_LAMBDAS = [0.5, 0.9, 0.95]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Sampled PPO Lambda Sweep Publication Plots")
    parser.add_argument("--results_dir", type=str, default="results/sampled_lambda_sweep", help="Results directory root")
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
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            env_data[env_name] = data
        elif os.path.exists(csv_path):
            env_data[env_name] = {"summary_only": True}

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            summary_dfs.append(df)

    combined_summary = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
    return env_data, combined_summary


def plot_matching_lambda_curves(env_data: dict, save_dir: str):
    """3x3 grid specifically highlighting matching-lambda head-to-head pairs."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes_flat = axes.flatten()

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes_flat[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axhline(0.0, color="#999999", linestyle=":", linewidth=0.8, alpha=0.7)

        if env_name not in env_data or "runs" not in env_data[env_name]:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color="#888888")
            continue

        data = env_data[env_name]
        runs = data.get("runs", {})
        baselines = data.get("baselines", {})
        v_rand = baselines.get("v_rand", 0.0)
        delta = baselines.get("delta", 1.0)

        for lam in MATCHING_LAMBDAS:
            lam_str = f"{lam:.2f}"
            td_key = f"td_{lam_str}"
            e_key = f"e_{lam_str}"

            # TD curve
            if td_key in runs:
                v_start = runs[td_key].get("V_start")
                if v_start is not None:
                    norm_traj = (v_start - v_rand) / delta
                    mean = np.mean(norm_traj, axis=0)
                    sem = np.std(norm_traj, axis=0) / np.sqrt(norm_traj.shape[0])
                    x = np.arange(len(mean))
                    c = CONDITION_COLORS.get(td_key, "#ff7f0e")
                    lbl = f"TD(λ={lam})"
                    ax.plot(x, mean, color=c, linestyle="--", linewidth=1.8, label=lbl if idx == 0 else "")
                    ax.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.15)

            # E curve
            if e_key in runs:
                v_start = runs[e_key].get("V_start")
                if v_start is not None:
                    norm_traj = (v_start - v_rand) / delta
                    mean = np.mean(norm_traj, axis=0)
                    sem = np.std(norm_traj, axis=0) / np.sqrt(norm_traj.shape[0])
                    x = np.arange(len(mean))
                    c = CONDITION_COLORS.get(e_key, "#2ca02c")
                    lbl = f"E(λ={lam}) [v_aux]"
                    ax.plot(x, mean, color=c, linestyle="-", linewidth=2.0, label=lbl if idx == 0 else "")
                    ax.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.20)

        ax.set_ylim(-0.2, 1.25)
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel("Standardized Return", fontsize=10)
        if idx >= 6:
            ax.set_xlabel("PPO Updates", fontsize=10)

    # Master legend at top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=6,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = os.path.join(save_dir, "matching_lambda_comparison_curves.pdf")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_all_learning_curves(env_data: dict, save_dir: str):
    """3x3 grid showing all 6 conditions across updates."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes_flat = axes.flatten()

    all_cond_keys = ["td_0.50", "td_0.90", "td_0.95", "e_0.50", "e_0.90", "e_0.95"]

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes_flat[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axhline(0.0, color="#999999", linestyle=":", linewidth=0.8, alpha=0.7)

        if env_name not in env_data or "runs" not in env_data[env_name]:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color="#888888")
            continue

        data = env_data[env_name]
        runs = data.get("runs", {})
        baselines = data.get("baselines", {})
        v_rand = baselines.get("v_rand", 0.0)
        delta = baselines.get("delta", 1.0)

        for c_key in all_cond_keys:
            if c_key in runs:
                v_start = runs[c_key].get("V_start")
                if v_start is not None:
                    norm_traj = (v_start - v_rand) / delta
                    mean = np.mean(norm_traj, axis=0)
                    sem = np.std(norm_traj, axis=0) / np.sqrt(norm_traj.shape[0])
                    x = np.arange(len(mean))
                    c = CONDITION_COLORS.get(c_key, "#333333")
                    lbl = CONDITION_LABELS.get(c_key, c_key)
                    ls = "--" if c_key.startswith("td") else "-"
                    ax.plot(x, mean, color=c, linestyle=ls, linewidth=1.8, label=lbl if idx == 0 else "")
                    ax.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.15)

        ax.set_ylim(-0.2, 1.25)
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel("Standardized Return", fontsize=10)
        if idx >= 6:
            ax.set_xlabel("PPO Updates", fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=6,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = os.path.join(save_dir, "standardized_performance_learning_curves.pdf")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_lambda_sensitivity(summary_df: pd.DataFrame, save_dir: str):
    """Plots performance as a function of lambda ∈ [0.5, 0.9, 0.95] for TD vs E."""
    if summary_df.empty or "lambda" not in summary_df.columns:
        return

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey=True)
    axes_flat = axes.flatten()

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes_flat[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axhline(0.0, color="#999999", linestyle=":", linewidth=0.8, alpha=0.7)

        sub_df = summary_df[summary_df["env_name"] == env_name]
        if sub_df.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color="#888888")
            continue

        for algo_key, color, ls, marker, lbl in [
            ("td_lambda", "#ff7f0e", "--", "o", "TD(λ)"),
            ("e_lambda", "#2ca02c", "-", "s", "E(λ) [v_aux]"),
        ]:
            algo_sub = sub_df[sub_df["algo"] == algo_key].sort_values("lambda")
            if not algo_sub.empty:
                lams = algo_sub["lambda"].values
                auc_vals = algo_sub["std_perf_auc"].values
                ax.plot(
                    lams,
                    auc_vals,
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markersize=6,
                    linewidth=2.0,
                    label=lbl if idx == 0 else "",
                )

        ax.set_xticks(MATCHING_LAMBDAS)
        ax.set_xticklabels(["0.50", "0.90", "0.95"])
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel("AUC Standardized Return", fontsize=10)
        if idx >= 6:
            ax.set_xlabel("Trace Parameter λ", fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=11,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = os.path.join(save_dir, "lambda_sensitivity_curves.pdf")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_head_to_head_delta_bars(summary_df: pd.DataFrame, save_dir: str):
    """Bar chart of performance delta: Δ = Perf(E(λ)) - Perf(TD(λ)) across environments."""
    if summary_df.empty or "lambda" not in summary_df.columns:
        return

    envs = [e for e in ALL_ENVS if e in summary_df["env_name"].unique()]
    if not envs:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(envs))
    width = 0.25

    colors = {0.5: "#a1d99b", 0.9: "#41ab5d", 0.95: "#005a32"}

    for idx, lam in enumerate(MATCHING_LAMBDAS):
        deltas = []
        for e in envs:
            td_row = summary_df[(summary_df["env_name"] == e) & (summary_df["algo"] == "td_lambda") & (summary_df["lambda"] == lam)]
            e_row = summary_df[(summary_df["env_name"] == e) & (summary_df["algo"] == "e_lambda") & (summary_df["lambda"] == lam)]

            if not td_row.empty and not e_row.empty:
                td_perf = td_row["std_perf_auc"].values[0]
                e_perf = e_row["std_perf_auc"].values[0]
                deltas.append(e_perf - td_perf)
            else:
                deltas.append(0.0)

        offset = (idx - 1) * width
        rects = ax.bar(x + offset, deltas, width, label=f"λ = {lam:.2f} (E - TD)", color=colors[lam], edgecolor="#222222", linewidth=0.8)

    ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("AUC Performance Advantage: E(λ) - TD(λ)\n(Positive = E(λ) is superior)", fontsize=11)
    ax.set_title("Matching-Lambda Head-to-Head Advantage [E(λ) vs TD(λ)]", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "delta_e_minus_td_bars.pdf")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_grouped_bars(summary_df: pd.DataFrame, metric_col: str, metric_label: str, save_path: str):
    """Grouped bar chart across environments for a specified metric."""
    if summary_df.empty or metric_col not in summary_df.columns:
        return

    envs = [e for e in ALL_ENVS if e in summary_df["env_name"].unique()]
    if not envs:
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    conditions = ["td_0.50", "td_0.90", "td_0.95", "e_0.50", "e_0.90", "e_0.95"]
    x = np.arange(len(envs))
    total_width = 0.8
    bar_width = total_width / len(conditions)

    for i, c_key in enumerate(conditions):
        vals = []
        errs = []
        for e in envs:
            row = summary_df[(summary_df["env_name"] == e) & (summary_df["condition"] == c_key)]
            if not row.empty:
                vals.append(row[metric_col].values[0])
                sem_col = f"{metric_col}_sem"
                errs.append(row[sem_col].values[0] if sem_col in row.columns else 0.0)
            else:
                vals.append(0.0)
                errs.append(0.0)

        offset = (i - (len(conditions) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width,
            yerr=errs,
            capsize=2,
            label=CONDITION_LABELS.get(c_key, c_key),
            color=CONDITION_COLORS.get(c_key, "#555555"),
            edgecolor="#222222",
            linewidth=0.7,
        )

    ax.axhline(0.0, color="#444444", linestyle="-", linewidth=0.8)
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"Comparison across Benchmark Environments: {metric_label}", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_metric_curves(env_data: dict, metric_name: str, y_label: str, filename: str, save_dir: str):
    """3x3 grid plotting an internal metric trajectory (VE or Cossim) across updates."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes_flat = axes.flatten()

    all_cond_keys = ["td_0.50", "td_0.90", "td_0.95", "e_0.50", "e_0.90", "e_0.95"]

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes_flat[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        if env_name not in env_data or "runs" not in env_data[env_name]:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color="#888888")
            continue

        runs = env_data[env_name].get("runs", {})
        for c_key in all_cond_keys:
            if c_key in runs:
                metric_seeds = runs[c_key].get(metric_name)
                if metric_seeds is not None:
                    mean = np.mean(metric_seeds, axis=0)
                    sem = np.std(metric_seeds, axis=0) / np.sqrt(metric_seeds.shape[0])
                    x = np.arange(len(mean))
                    c = CONDITION_COLORS.get(c_key, "#333333")
                    lbl = CONDITION_LABELS.get(c_key, c_key)
                    ls = "--" if c_key.startswith("td") else "-"
                    ax.plot(x, mean, color=c, linestyle=ls, linewidth=1.8, label=lbl if idx == 0 else "")
                    ax.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.15)

        ax.grid(True, linestyle="--", alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel(y_label, fontsize=10)
        if idx >= 6:
            ax.set_xlabel("PPO Updates", fontsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=6,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir or os.path.join(results_dir, "plots"))
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("Generating Publication Plots for Sampled PPO Lambda Sweep")
    print(f"Results Directory: {results_dir}")
    print(f"Save Directory:    {save_dir}")
    print("=" * 80)

    env_data, summary_df = load_all_env_data(results_dir)

    # Save consolidated summary CSV if not already present
    if not summary_df.empty:
        summary_all_path = os.path.join(results_dir, "summary_table_all.csv")
        summary_df.to_csv(summary_all_path, index=False)
        print(f"Consolidated summary table: {summary_all_path}")

    # 1. Matching Lambda Comparison Curves
    plot_matching_lambda_curves(env_data, save_dir)

    # 2. All 6 Conditions Learning Curves
    plot_all_learning_curves(env_data, save_dir)

    # 3. Lambda Sensitivity Curves
    plot_lambda_sensitivity(summary_df, save_dir)

    # 4. Head-to-Head Delta Bars (E(λ) - TD(λ))
    plot_head_to_head_delta_bars(summary_df, save_dir)

    # 5. Grouped Bars: AUC Standardized Return & Final Standardized Return
    plot_grouped_bars(
        summary_df,
        metric_col="std_perf_auc",
        metric_label="AUC Standardized Policy Return",
        save_path=os.path.join(save_dir, "auc_standardized_performance_bars.pdf"),
    )
    plot_grouped_bars(
        summary_df,
        metric_col="std_perf_final",
        metric_label="Final Standardized Policy Return",
        save_path=os.path.join(save_dir, "final_standardized_performance_bars.pdf"),
    )

    # 6. Value Error & Advantage Cosine Similarity Curves
    plot_metric_curves(
        env_data,
        metric_name="nn_weighted_VE",
        y_label="Learned Critic VE",
        filename="value_error_learning_curves.pdf",
        save_dir=save_dir,
    )
    plot_metric_curves(
        env_data,
        metric_name="nn_advantage_cossim",
        y_label="Advantage Cosine Similarity",
        filename="advantage_cossim_learning_curves.pdf",
        save_dir=save_dir,
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
