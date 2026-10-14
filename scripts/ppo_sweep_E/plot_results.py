"""
scripts/ppo_sweep_E/plot_results.py

Consolidates and visualizes PPO Sampled E Critic sweeps across environments:
1. returns_across_envs.pdf: Line charts of evaluation return / V_start during training
2. v_start_across_envs.pdf: Line charts of true policy start value V^pi(s_0)
3. value_error_mse_across_envs.pdf: Line charts of Value Error MSE (log scale)
4. E_across_envs.pdf: Line charts of Exact Matrix E (log scale)
5. final_performance_bars.pdf: Bar charts of final policy performance across environments
6. summary_metrics.csv: Table of numerical metrics (mean, SEM, final, AUC)
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

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"

ALL_ENVS = [
    "FourRooms-misc",
    "fourrooms-dense",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-misc",
    "eightrooms-dense",
    "whirlpool-misc",
    "MountainCar-v0",
    "mountaincar-dense",
    "SpaceInvadersExactValue",
]

ENV_DISPLAY_NAMES = {
    "FourRooms-misc": "Four Rooms",
    "fourrooms-dense": "Four Rooms (Dense)",
    "fourrooms-mines": "Four Rooms Mines",
    "fourrooms-mines-dense": "Four Rooms Mines (Dense)",
    "eightrooms-misc": "Eight Rooms",
    "eightrooms-dense": "Eight Rooms (Dense)",
    "whirlpool-misc": "Whirlpool",
    "MountainCar-v0": "Mountain Car",
    "mountaincar-dense": "Mountain Car (Dense)",
    "SpaceInvadersExactValue": "Space Invaders",
}

VAR_COLORS = {
    "sep14": "#1f77b4",     # Blue
    "oct5": "#d62728",      # Red
    "oct8": "#2ca02c",      # Green
}

VAR_LABELS = {
    "sep14": "Sept 14 (Uncorrected)",
    "oct5": "Oct 5 (Corrected)",
    "oct8": "Oct 8 (Start-Anchored)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PPO Sampled E Comparison Plots")
    parser.add_argument("--results_dir", type=str, default="results/ppo_sweep_E", help="Results directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory (defaults to results_dir)")
    return parser.parse_args()


def load_env_results(results_dir: str, env_name: str):
    """Loads out.pkl for an environment containing dict of variants."""
    candidate = os.path.join(results_dir, env_name, "out.pkl")
    if os.path.exists(candidate):
        try:
            with open(candidate, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {candidate}: {e}")
            return None
    return None


def extract_stats(arr):
    """Extracts mean and SEM over seeds."""
    if arr is None:
        return None, None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr, np.zeros_like(arr)
    elif arr.ndim >= 2:
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(max(arr.shape[0], 1))
        if mean.ndim > 1:
            mean = mean.squeeze()
            sem = sem.squeeze()
        return mean, sem
    return None, None


def plot_metric_grid(env_data: dict, metric_getter, title: str, ylabel: str, filename: str, save_dir: str, log_scale: bool = False):
    """Generates a 10-panel grid (2 rows x 5 columns) for a specific metric."""
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(ALL_ENVS)
    n_cols = 5
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 8), dpi=150, sharex=False)
    axes_flat = axes.flatten()

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes_flat[idx]
        display_name = ENV_DISPLAY_NAMES.get(env_name, env_name)
        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        if env_name not in env_data:
            ax.text(0.5, 0.5, "Pending Run", ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
            continue

        data = env_data[env_name]
        for v_key in ["sep14", "oct5", "oct8"]:
            if v_key not in data:
                continue
            v_metrics = data[v_key]
            raw_metric = metric_getter(v_metrics)
            if raw_metric is None:
                continue

            mean, sem = extract_stats(raw_metric)
            if mean is not None:
                x = np.arange(len(mean))
                color = VAR_COLORS[v_key]
                label = VAR_LABELS[v_key]
                ax.plot(x, mean, color=color, lw=2.0, label=label)
                if sem is not None and np.any(sem > 0):
                    lower = np.maximum(mean - sem, 1e-12) if log_scale else (mean - sem)
                    ax.fill_between(x, lower, mean + sem, color=color, alpha=0.18)

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Updates", fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=10)

    # Common legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if not handles and len(valid_envs) > 0:
        first_valid_idx = ALL_ENVS.index(valid_envs[0])
        handles, labels = axes_flat[first_valid_idx].get_legend_handles_labels()

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=True)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, filename)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


def plot_bar_comparisons(env_data: dict, save_dir: str):
    """Generates bar charts of final return and final value error MSE across environments."""
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_envs * 1.5), 5.5), dpi=150)
    ax_ret, ax_ve = axes[0], axes[1]

    x = np.arange(n_envs)
    width = 0.26
    variants = [
        ("sep14", -width, VAR_COLORS["sep14"], VAR_LABELS["sep14"]),
        ("oct5", 0.0, VAR_COLORS["oct5"], VAR_LABELS["oct5"]),
        ("oct8", width, VAR_COLORS["oct8"], VAR_LABELS["oct8"]),
    ]

    for v_key, offset, color, label in variants:
        final_rets = []
        final_ves = []
        sem_rets = []
        sem_ves = []

        for e in valid_envs:
            data = env_data[e]
            if v_key in data:
                v_metrics = data[v_key]
                # Return
                raw_ret = v_metrics.get("returned_episode_returns")
                if raw_ret is None:
                    raw_ret = v_metrics.get("V_start")
                if raw_ret is not None:
                    arr = np.asarray(raw_ret)
                    final_vals = arr[:, -10:] if arr.ndim >= 2 else arr[-10:]
                    final_rets.append(float(np.nanmean(final_vals)))
                    sem_rets.append(float(np.nanstd(np.nanmean(final_vals, axis=-1 if arr.ndim >= 2 else 0)) / np.sqrt(max(arr.shape[0] if arr.ndim >= 2 else 1, 1))))
                else:
                    final_rets.append(0.0)
                    sem_rets.append(0.0)

                # VE
                raw_ve = v_metrics.get("nn_weighted_VE")
                if raw_ve is not None:
                    arr = np.asarray(raw_ve)
                    final_vals = arr[:, -10:] if arr.ndim >= 2 else arr[-10:]
                    final_ves.append(float(np.nanmean(final_vals)))
                    sem_ves.append(float(np.nanstd(np.nanmean(final_vals, axis=-1 if arr.ndim >= 2 else 0)) / np.sqrt(max(arr.shape[0] if arr.ndim >= 2 else 1, 1))))
                else:
                    final_ves.append(1e-12)
                    sem_ves.append(0.0)
            else:
                final_rets.append(0.0)
                sem_rets.append(0.0)
                final_ves.append(1e-12)
                sem_ves.append(0.0)

        ax_ret.bar(x + offset, final_rets, width, yerr=sem_rets, label=label, color=color, alpha=0.85, edgecolor="black", lw=0.6, capsize=3)
        ax_ve.bar(x + offset, final_ves, width, yerr=sem_ves, label=label, color=color, alpha=0.85, edgecolor="black", lw=0.6, capsize=3)

    display_labels = [ENV_DISPLAY_NAMES.get(e, e) for e in valid_envs]
    ax_ret.set_xticks(x)
    ax_ret.set_xticklabels(display_labels, fontsize=9, fontweight="bold", rotation=25, ha="right")
    ax_ret.set_ylabel("Final Return / $V^\\pi(s_0)$", fontsize=10, fontweight="bold")
    ax_ret.set_title("Final Evaluation Return Across Environments", fontsize=11, fontweight="bold")
    ax_ret.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax_ret.legend(fontsize=9, frameon=True)

    ax_ve.set_xticks(x)
    ax_ve.set_xticklabels(display_labels, fontsize=9, fontweight="bold", rotation=25, ha="right")
    ax_ve.set_yscale("log")
    ax_ve.set_ylabel("Final Value Error MSE (Log Scale)", fontsize=10, fontweight="bold")
    ax_ve.set_title("Final Value Error MSE Across Environments", fontsize=11, fontweight="bold")
    ax_ve.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax_ve.legend(fontsize=9, frameon=True)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "final_performance_bars.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


def generate_summary_csv(env_data: dict, save_dir: str):
    """Generates a summary CSV table with numerical metrics."""
    rows = []
    valid_envs = [e for e in ALL_ENVS if e in env_data]

    for env_name in valid_envs:
        data = env_data[env_name]

        for v_key in ["sep14", "oct5", "oct8"]:
            if v_key not in data:
                continue
            v_metrics = data[v_key]

            # Return
            raw_ret = v_metrics.get("returned_episode_returns")
            if raw_ret is None:
                raw_ret = v_metrics.get("V_start")
            final_ret, sem_ret, auc_ret = None, None, None
            if raw_ret is not None:
                arr = np.asarray(raw_ret)
                m, _ = extract_stats(arr)
                if m is not None:
                    final_ret = float(np.nanmean(m[-10:]))
                    auc_ret = float(np.nanmean(m))
                    if arr.ndim >= 2:
                        sem_ret = float(np.nanstd(np.nanmean(arr[:, -10:], axis=1)) / np.sqrt(arr.shape[0]))

            # V_start
            raw_vs = v_metrics.get("V_start")
            final_vs = None
            if raw_vs is not None:
                m_vs, _ = extract_stats(raw_vs)
                if m_vs is not None:
                    final_vs = float(np.nanmean(m_vs[-10:]))

            # VE
            raw_ve = v_metrics.get("nn_weighted_VE")
            final_ve = None
            if raw_ve is not None:
                m_ve, _ = extract_stats(raw_ve)
                if m_ve is not None:
                    final_ve = float(np.nanmean(m_ve[-10:]))

            # Exact E
            raw_e = v_metrics.get("E")
            if raw_e is None:
                raw_e = v_metrics.get("E_exact")
            final_e = None
            if raw_e is not None:
                m_e, _ = extract_stats(raw_e)
                if m_e is not None:
                    final_e = float(np.nanmean(m_e[-10:]))

            rows.append({
                "Environment": env_name,
                "Variant": v_key,
                "Variant_Name": VAR_LABELS[v_key],
                "Final_Return": final_ret,
                "Final_Return_SEM": sem_ret,
                "AUC_Return": auc_ret,
                "Final_V_Start": final_vs,
                "Final_Value_Error_MSE": final_ve,
                "Final_Exact_E": final_e,
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else results_dir
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("Generating PPO Sampled E Comparison Figures")
    print(f"Results Directory: {results_dir}")
    print(f"Save Directory:    {save_dir}")
    print("=" * 70)

    env_data = {}
    for env_name in ALL_ENVS:
        res = load_env_results(results_dir, env_name)
        if res is not None:
            env_data[env_name] = res

    print(f"Loaded results for {len(env_data)}/{len(ALL_ENVS)} environments.")

    # 1. Returns line chart
    def get_returns(v):
        return v.get("returned_episode_returns") if "returned_episode_returns" in v else v.get("V_start")
    plot_metric_grid(
        env_data,
        metric_getter=get_returns,
        title="PPO Evaluation Returns During Learning Across Environments",
        ylabel="Evaluation Return",
        filename="returns_across_envs.pdf",
        save_dir=save_dir,
        log_scale=False,
    )

    # 2. V_start line chart
    def get_v_start(v):
        return v.get("V_start")
    plot_metric_grid(
        env_data,
        metric_getter=get_v_start,
        title="PPO Policy Value $V^\\pi(s_0)$ During Learning Across Environments",
        ylabel="Start Value $V^\\pi(s_0)$",
        filename="v_start_across_envs.pdf",
        save_dir=save_dir,
        log_scale=False,
    )

    # 3. Value Error MSE line chart
    def get_ve(v):
        return v.get("nn_weighted_VE")
    plot_metric_grid(
        env_data,
        metric_getter=get_ve,
        title="PPO Value Error MSE During Learning Across Environments",
        ylabel="Value Error MSE (Log Scale)",
        filename="value_error_mse_across_envs.pdf",
        save_dir=save_dir,
        log_scale=True,
    )

    # 4. Exact Matrix E line chart
    def get_e(v):
        return v.get("E") if "E" in v else v.get("E_exact")
    plot_metric_grid(
        env_data,
        metric_getter=get_e,
        title="PPO Exact Matrix $E$ During Learning Across Environments",
        ylabel="Exact $E$ (Log Scale)",
        filename="E_across_envs.pdf",
        save_dir=save_dir,
        log_scale=True,
    )

    # 5. Bar chart comparisons
    plot_bar_comparisons(env_data, save_dir)

    # 6. Summary metrics CSV
    generate_summary_csv(env_data, save_dir)

    print("\nAll plots and summary metrics successfully generated!")


if __name__ == "__main__":
    main()
