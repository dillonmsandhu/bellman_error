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
    # TD(lambda) - Orange / Red shades
    "td_0.40": "#fdd0a2",
    "td_0.80": "#fdae6b",
    "td_0.95": "#e6550d",
    "td_0.99": "#8c2d04",
    # Legacy / backward-compat
    "td_0.50": "#fdae6b",
    "td_0.90": "#e6550d",

    # E(lambda) [v_aux] - Green shades
    "e_0.40":  "#c7e9c0",
    "e_0.80":  "#74c476",
    "e_0.95":  "#238b45",
    "e_0.99":  "#00441b",
    # Legacy
    "e_0.50":  "#a1d99b",
    "e_0.90":  "#31a354",

    # E(lambda) [Forward-Only] - Blue shades
    "e_fwd_0.40": "#c6dbef",
    "e_fwd_0.80": "#6baed6",
    "e_fwd_0.95": "#2171b5",
    "e_fwd_0.99": "#08306b",
}

CONDITION_LABELS = {
    "td_0.40": "TD(0.40)",
    "td_0.80": "TD(0.80)",
    "td_0.95": "TD(0.95)",
    "td_0.99": "TD(0.99)",
    "td_0.50": "TD(0.50)",
    "td_0.90": "TD(0.90)",

    "e_0.40":  "E(0.40) [v_aux]",
    "e_0.80":  "E(0.80) [v_aux]",
    "e_0.95":  "E(0.95) [v_aux]",
    "e_0.99":  "E(0.99) [v_aux]",
    "e_0.50":  "E(0.50) [v_aux]",
    "e_0.90":  "E(0.90) [v_aux]",

    "e_fwd_0.40": "E(0.40) [Fwd]",
    "e_fwd_0.80": "E(0.80) [Fwd]",
    "e_fwd_0.95": "E(0.95) [Fwd]",
    "e_fwd_0.99": "E(0.99) [Fwd]",
}

CONDITION_ORDER = [
    "td_0.40", "td_0.80", "td_0.95", "td_0.99",
    "td_0.50", "td_0.90",
    "e_0.40", "e_0.80", "e_0.95", "e_0.99",
    "e_0.50", "e_0.90",
    "e_fwd_0.40", "e_fwd_0.80", "e_fwd_0.95", "e_fwd_0.99",
]

MATCHING_LAMBDAS = [0.4, 0.8, 0.95, 0.99]


def plot_lambda_weight_tradeoff(save_dir: str, gamma: float = 0.99, lambdas: list[float] = [0.4, 0.8, 0.95, 0.99]):
    """
    Plots the analytical loss weights:
      w_dir(lambda) = 0.5 * gamma * (1 - lambda) / (1 - gamma * lambda)
      w_mag(lambda) = 0.5 * (1 - gamma) / (1 - gamma * lambda)
    over lambda in [0, 0.999], highlighting the chosen sweep points.
    """
    os.makedirs(save_dir, exist_ok=True)
    lambda_grid = np.linspace(0.0, 0.999, 1000)
    denom = np.maximum(1.0 - gamma * lambda_grid, 1e-8)
    tilde_gamma = gamma * (1.0 - lambda_grid) / denom
    w_dir = 0.5 * tilde_gamma
    w_mag = 0.5 * (1.0 - gamma) / denom

    share_dir = (w_dir / (w_dir + w_mag)) * 100.0
    share_mag = (w_mag / (w_dir + w_mag)) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Panel 1: Absolute Weights
    ax1 = axes[0]
    ax1.plot(lambda_grid, w_dir, color="#2ca02c", linewidth=2.2, label=r"Dirichlet Weight $w_{\mathrm{dir}}(\lambda) = \frac{1}{2}\tilde{\gamma}$")
    ax1.plot(lambda_grid, w_mag, color="#756bb1", linewidth=2.2, label=r"Magnitude Weight $w_{\mathrm{mag}}(\lambda) = \frac{1-\gamma}{2(1-\gamma\lambda)}$")

    for lam in lambdas:
        d_val = 0.5 * gamma * (1.0 - lam) / max(1.0 - gamma * lam, 1e-8)
        m_val = 0.5 * (1.0 - gamma) / max(1.0 - gamma * lam, 1e-8)
        ax1.axvline(lam, color="#888888", linestyle="--", alpha=0.5, linewidth=0.9)
        ax1.scatter([lam], [d_val], color="#2ca02c", s=35, zorder=5)
        ax1.scatter([lam], [m_val], color="#756bb1", s=35, zorder=5)
        ax1.annotate(
            f"λ={lam:.2f}\n$w_{{dir}}$={d_val:.3f}\n$w_{{mag}}$={m_val:.3f}",
            xy=(lam, max(d_val, m_val)),
            xytext=(lam - 0.12 if lam > 0.85 else lam + 0.02, 0.32 if lam > 0.85 else 0.12),
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#bbbbbb", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7),
        )

    ax1.set_title(r"(a) Absolute Critic Loss Weights ($\gamma=0.99$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Lambda ($\lambda$)", fontsize=10)
    ax1.set_ylabel("Loss Coefficient", fontsize=10)
    ax1.set_xlim(-0.02, 1.01)
    ax1.set_ylim(-0.02, 0.55)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc")

    # Panel 2: Relative Shares (%)
    ax2 = axes[1]
    ax2.plot(lambda_grid, share_dir, color="#2ca02c", linewidth=2.2, label="Dirichlet Share (%)")
    ax2.plot(lambda_grid, share_mag, color="#756bb1", linewidth=2.2, label="Magnitude Share (%)")
    ax2.axhline(50.0, color="#d95f02", linestyle=":", linewidth=1.5, label="50/50 Balance")

    # Intersection where 50/50 occurs: lambda = (2*gamma - 1)/gamma^2
    crossover_lam = (2.0 * gamma - 1.0) / (gamma ** 2) if (2.0 * gamma - 1.0) > 0 else 0.99
    ax2.axvline(crossover_lam, color="#d95f02", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.annotate(
        f"1:1 Equal Balance\nat λ ≈ {crossover_lam:.3f}",
        xy=(crossover_lam, 50.0),
        xytext=(0.55, 55.0),
        fontsize=8.5,
        fontweight="bold",
        color="#d95f02",
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff2e6", edgecolor="#d95f02"),
    )

    for lam in lambdas:
        d_val = 0.5 * gamma * (1.0 - lam) / max(1.0 - gamma * lam, 1e-8)
        m_val = 0.5 * (1.0 - gamma) / max(1.0 - gamma * lam, 1e-8)
        sh_d = (d_val / (d_val + m_val)) * 100.0
        ax2.scatter([lam], [sh_d], color="#2ca02c", s=30, zorder=5)

    ax2.set_title(r"(b) Relative Loss Share (%) vs. $\lambda$", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Lambda ($\lambda$)", fontsize=10)
    ax2.set_ylabel("Share of Total Weight (%)", fontsize=10)
    ax2.set_xlim(-0.02, 1.01)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "lambda_weight_tradeoff.pdf")
    png_path = os.path.join(save_dir, "lambda_weight_tradeoff.png")
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved analytical weight plot: {pdf_path} and {png_path}")


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


def plot_forward_only_comparison_curves(env_data: dict, save_dir: str, lambdas: list[float] = MATCHING_LAMBDAS):
    """3x3 grid specifically highlighting E(lambda) [v_aux] vs E(lambda) [Forward-Only]."""
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

        for lam in lambdas:
            lam_str = f"{lam:.2f}"
            e_key = f"e_{lam_str}"
            fwd_key = f"e_fwd_{lam_str}"

            # E(lambda) [v_aux] curve
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
                    ax.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.18)

            # E(lambda) [Forward-Only] curve
            if fwd_key in runs:
                v_start = runs[fwd_key].get("V_start")
                if v_start is not None:
                    norm_traj = (v_start - v_rand) / delta
                    mean = np.mean(norm_traj, axis=0)
                    sem = np.std(norm_traj, axis=0) / np.sqrt(norm_traj.shape[0])
                    x = np.arange(len(mean))
                    c = CONDITION_COLORS.get(fwd_key, "#1f77b4")
                    lbl = f"E(λ={lam}) [Fwd]"
                    ax.plot(x, mean, color=c, linestyle=":", linewidth=2.0, label=lbl if idx == 0 else "")
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
            ncol=min(len(handles), 8),
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=9.5,
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = os.path.join(save_dir, "forward_only_ablation_curves.pdf")
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
            ("e_lambda_fwd", "#1f77b4", ":", "^", "E(λ) [Forward-Only]"),
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

        present_lambdas = sorted(summary_df["lambda"].dropna().unique())
        if len(present_lambdas) > 0:
            ax.set_xticks(present_lambdas)
            ax.set_xticklabels([f"{l:.2f}" for l in present_lambdas])
        else:
            ax.set_xticks(MATCHING_LAMBDAS)
            ax.set_xticklabels([f"{l:.2f}" for l in MATCHING_LAMBDAS])
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
            ncol=3,
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

    active_lambdas = [l for l in MATCHING_LAMBDAS if l in summary_df["lambda"].unique()]
    if not active_lambdas:
        active_lambdas = sorted(summary_df["lambda"].dropna().unique())
    if not active_lambdas:
        active_lambdas = MATCHING_LAMBDAS

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(envs))
    total_width = 0.8
    width = total_width / max(len(active_lambdas), 1)

    colors = {0.4: "#c7e9c0", 0.8: "#74c476", 0.95: "#31a354", 0.99: "#006d2c", 0.5: "#a1d99b", 0.9: "#41ab5d"}

    for idx, lam in enumerate(active_lambdas):
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

        offset = (idx - (len(active_lambdas) - 1) / 2.0) * width
        rects = ax.bar(x + offset, deltas, width, label=f"λ = {lam:.2f} (E - TD)", color=colors.get(lam, "#31a354"), edgecolor="#222222", linewidth=0.8)

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
    conditions = [c for c in CONDITION_ORDER if c in summary_df["condition"].unique()]
    if not conditions:
        conditions = list(summary_df["condition"].unique())
    x = np.arange(len(envs))
    total_width = 0.8
    bar_width = total_width / max(len(conditions), 1)

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

        offset = (i - (len(conditions) - 1) / 2.0) * bar_width
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

    # 0. Analytical Weight Tradeoff Plot (Dirichlet vs. Magnitude)
    plot_lambda_weight_tradeoff(save_dir, gamma=0.99, lambdas=MATCHING_LAMBDAS)

    # 1. Matching Lambda Comparison Curves: E(lambda) [v_aux] vs. TD(lambda)
    plot_matching_lambda_curves(env_data, save_dir)

    # 2. Forward-Only Ablation Curves: E(lambda) [v_aux] vs. E(lambda) [Forward-Only]
    plot_forward_only_comparison_curves(env_data, save_dir, lambdas=MATCHING_LAMBDAS)

    # 3. All Conditions Learning Curves
    plot_all_learning_curves(env_data, save_dir)

    # 4. Lambda Sensitivity Curves
    plot_lambda_sensitivity(summary_df, save_dir)

    # 5. Head-to-Head Delta Bars (E(λ) - TD(λ))
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
