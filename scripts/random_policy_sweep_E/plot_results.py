"""
scripts/random_policy_sweep_E/plot_results.py

Generates consolidated comparison plots for Random Policy Sampled E Evaluation:
1. Line chart with Value Error MSE during learning across all environments (value_error_mse_across_envs.pdf).
2. Line chart with Exact Matrix E during learning across all environments (E_across_envs.pdf).
3. Bar chart showing average Gradient MSE to Exact E across all environments (gradient_mse_across_envs.pdf).
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
    parser = argparse.ArgumentParser(description="Generate Random Policy Sampled E Plots")
    parser.add_argument("--results_dir", type=str, default="results/random_policy_sweep_E", help="Results directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
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
    """Extracts mean and std over seeds."""
    if arr is None:
        return None, None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr, np.zeros_like(arr)
    elif arr.ndim >= 2:
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        if mean.ndim > 1:
            mean = mean.squeeze()
            std = std.squeeze()
        return mean, std
    return None, None


def plot_value_error_lines(env_data: dict, save_dir: str):
    """
    Plot 1: Line chart with Value Error MSE during learning for all variants across environments.
    """
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    n_cols = 3
    n_rows = int(np.ceil(n_envs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.4), dpi=200, sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, env_name in enumerate(valid_envs):
        ax = axes[idx]
        data = env_data[env_name]

        for v_key in ["sep14", "oct5", "oct8"]:
            if v_key not in data:
                continue
            v_metrics = data[v_key]
            # Key for value error MSE is nn_weighted_VE
            raw_ve = v_metrics.get("nn_weighted_VE")
            if raw_ve is None:
                continue

            mean_ve, std_ve = extract_stats(raw_ve)
            if mean_ve is not None:
                x = np.arange(len(mean_ve))
                ax.plot(x, mean_ve, color=VAR_COLORS[v_key], lw=2.0, label=VAR_LABELS[v_key])
                if std_ve is not None and np.any(std_ve > 0):
                    ax.fill_between(
                        x,
                        np.maximum(mean_ve - std_ve, 1e-12),
                        mean_ve + std_ve,
                        color=VAR_COLORS[v_key],
                        alpha=0.18,
                    )

        ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Value Error MSE (Log Scale)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide unused subplots
    for j in range(n_envs, len(axes)):
        axes[j].set_visible(False)

    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=True)

    plt.suptitle("Random Policy Value Error MSE During Learning Across Environments", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, "value_error_mse_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Value Error Line Chart: {pdf_path}")


def plot_exact_e_lines(env_data: dict, save_dir: str):
    """
    Line chart with Exact Matrix E during learning for all variants across environments.
    """
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    n_cols = 3
    n_rows = int(np.ceil(n_envs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.4), dpi=200, sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, env_name in enumerate(valid_envs):
        ax = axes[idx]
        data = env_data[env_name]

        for v_key in ["sep14", "oct5", "oct8"]:
            if v_key not in data:
                continue
            v_metrics = data[v_key]
            raw_e = v_metrics.get("E")
            if raw_e is None:
                raw_e = v_metrics.get("E_exact")
            if raw_e is None:
                continue

            mean_e, std_e = extract_stats(raw_e)
            if mean_e is not None:
                x = np.arange(len(mean_e))
                ax.plot(x, mean_e, color=VAR_COLORS[v_key], lw=2.0, label=VAR_LABELS[v_key])
                if std_e is not None and np.any(std_e > 0):
                    ax.fill_between(
                        x,
                        np.maximum(mean_e - std_e, 1e-12),
                        mean_e + std_e,
                        color=VAR_COLORS[v_key],
                        alpha=0.18,
                    )

        ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Exact Matrix E (Log Scale)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(n_envs, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=True)

    plt.suptitle("Random Policy Exact Matrix E During Learning Across Environments", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, "E_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Exact E Line Chart: {pdf_path}")


def plot_gradient_mse_bars(env_data: dict, save_dir: str):
    """
    Plot 2: Bar chart showing average Gradient MSE where the error is against Exact E gradient.
    """
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    fig, ax = plt.subplots(figsize=(max(10, n_envs * 1.8), 6), dpi=200)

    x = np.arange(n_envs)
    width = 0.26

    variants = [
        ("sep14", -width, VAR_COLORS["sep14"], VAR_LABELS["sep14"]),
        ("oct5", 0.0, VAR_COLORS["oct5"], VAR_LABELS["oct5"]),
        ("oct8", width, VAR_COLORS["oct8"], VAR_LABELS["oct8"]),
    ]

    for v_key, offset, color, label in variants:
        avg_grad_mses = []
        for e in valid_envs:
            data = env_data[e]
            if v_key in data and "grad_mse_exact_E" in data[v_key]:
                m, _ = extract_stats(data[v_key]["grad_mse_exact_E"])
                val = float(np.nanmean(m)) if m is not None else 1e-12
            else:
                val = 1e-12
            avg_grad_mses.append(max(val, 1e-12))

        ax.bar(
            x + offset,
            avg_grad_mses,
            width,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="black",
            lw=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Average Gradient MSE to Exact E (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Average Gradient MSE to Exact E Gradient (Average during Learning)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(loc="upper right", ncol=3, fontsize=9.5, frameon=True)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "gradient_mse_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Gradient MSE Bar Chart: {pdf_path}")


def generate_summary_csv(env_data: dict, save_dir: str):
    """Generates a summary CSV table with numerical metrics."""
    rows = []
    valid_envs = [e for e in ALL_ENVS if e in env_data]

    for env_name in valid_envs:
        row = {"Environment": env_name}
        data = env_data[env_name]

        for v_key in ["sep14", "oct5", "oct8"]:
            if v_key not in data:
                continue
            v_metrics = data[v_key]

            # Exact Matrix E
            raw_e = v_metrics.get("E")
            if raw_e is None:
                raw_e = v_metrics.get("E_exact")
            if raw_e is not None:
                m_e, _ = extract_stats(raw_e)
                if m_e is not None:
                    row[f"{v_key}_E_mean"] = float(np.nanmean(m_e))
                    row[f"{v_key}_E_final"] = float(m_e[-1])

            # Value Error MSE
            if "nn_weighted_VE" in v_metrics:
                m_ve, _ = extract_stats(v_metrics["nn_weighted_VE"])
                if m_ve is not None:
                    row[f"{v_key}_VE_mean"] = float(np.nanmean(m_ve))
                    row[f"{v_key}_VE_final"] = float(m_ve[-1])

            # Gradient MSE to Exact E
            if "grad_mse_exact_E" in v_metrics:
                m_gmse, _ = extract_stats(v_metrics["grad_mse_exact_E"])
                if m_gmse is not None:
                    row[f"{v_key}_grad_mse_mean"] = float(np.nanmean(m_gmse))
                    row[f"{v_key}_grad_mse_final"] = float(m_gmse[-1])

            # Gradient Cosine
            if "grad_cos_exact_E" in v_metrics:
                m_cos, _ = extract_stats(v_metrics["grad_cos_exact_E"])
                if m_cos is not None:
                    row[f"{v_key}_grad_cos_mean"] = float(np.nanmean(m_cos))
                    row[f"{v_key}_grad_cos_final"] = float(m_cos[-1])

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved Summary Table to: {csv_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir if args.save_dir else results_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("Random Policy Sampled E: Benchmark Plotting")
    print(f"Results Directory: {results_dir}")
    print(f"Save Directory:    {save_dir}")
    print("=" * 70)

    env_data = {}
    for env_name in ALL_ENVS:
        res = load_env_results(results_dir, env_name)
        if res is not None:
            print(f"Loaded: {env_name}")
            env_data[env_name] = res
        else:
            print(f"Missing: {env_name}")

    if not env_data:
        print("No environment data found in", results_dir)
        return

    plot_value_error_lines(env_data, save_dir)
    plot_exact_e_lines(env_data, save_dir)
    plot_gradient_mse_bars(env_data, save_dir)
    generate_summary_csv(env_data, save_dir)


if __name__ == "__main__":
    main()
