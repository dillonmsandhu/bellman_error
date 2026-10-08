"""
scripts/sampling_E_test/plot_comparison.py

Consolidates and plots Sampling E Test results across all benchmark environments.
Generates 5 multi-panel publication figures:
1. returns_across_envs: Episode returns across all 7 environments
2. uncorrected_E_across_envs: Exact E vs Expected Uncorrected E vs Sampled Uncorrected E
3. corrected_E_across_envs: Exact E vs Expected Corrected E vs Sampled Corrected E
4. gradient_similarity_corrected_across_envs: Cosine similarity of corrected gradients vs exact E
5. gradient_similarity_uncorrected_across_envs: Cosine similarity of uncorrected gradients vs exact E

Also outputs summary_metrics.csv.
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import cloudpickle
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate consolidated multi-environment Sampling E comparison plots")
    parser.add_argument("--results_dir", type=str, default="results/sampling_E_test", help="Path to results directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save figures (defaults to results_dir)")
    return parser.parse_args()


def load_env_metrics(results_dir: str, env_name: str):
    """Loads out.pkl and config.json for a given environment."""
    # Find matching directory
    pattern = os.path.join(results_dir, f"*{env_name}*")
    matches = glob.glob(pattern)
    if not matches:
        # Check direct subdirectory
        candidate = os.path.join(results_dir, env_name)
        if os.path.isdir(candidate):
            matches = [candidate]

    if not matches:
        return None, None

    env_dir = matches[0]
    pkl_file = os.path.join(env_dir, "out.pkl")
    if not os.path.exists(pkl_file):
        return None, None

    try:
        with open(pkl_file, "rb") as f:
            data = cloudpickle.load(f)
        if isinstance(data, dict) and "metrics" in data:
            metrics = data["metrics"]
        else:
            metrics = data
        return metrics, env_dir
    except Exception as e:
        print(f"Error loading {pkl_file}: {e}")
        return None, None


def extract_metric(metrics: dict, key: str):
    """Extracts mean and std over seeds for a given metric."""
    if metrics is None or key not in metrics:
        return None, None
    arr = np.asarray(metrics[key])
    if arr.ndim == 1:
        return arr, np.zeros_like(arr)
    elif arr.ndim >= 2:
        # Average over seeds (axis 0)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        if mean.ndim > 1:
            mean = mean.squeeze()
            std = std.squeeze()
        return mean, std
    return None, None


def make_multi_env_grid(
    env_data: dict,
    plot_fn,
    title: str,
    save_path_pdf: str,
    save_path_png: str,
):
    """Creates a 3x3 grid of subplots for the 9 environments."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=False)
    axes_flat = axes.flatten()

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes_flat[idx]
        metrics = env_data.get(env_name)
        if metrics is not None:
            plot_fn(ax, metrics, env_name)
            # Add a local legend to each populated subplot if handles exist
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=9, frameon=True)
        else:
            ax.text(0.5, 0.5, "Data Not Found", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(env_name, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path_pdf, bbox_inches="tight")
    plt.savefig(save_path_png, bbox_inches="tight", dpi=250)
    plt.close()
    print(f"Saved: {save_path_pdf} and {save_path_png}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir if args.save_dir else results_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"Sampling E Test Consolidated Plotting")
    print(f"Results Directory: {results_dir}")
    print(f"Save Directory:    {save_dir}")
    print("=" * 70)

    # 1. Load data for all environments
    env_data = {}
    for env_name in ALL_ENVS:
        metrics, env_dir = load_env_metrics(results_dir, env_name)
        if metrics is not None:
            print(f"Loaded: {env_name} from {env_dir}")
            env_data[env_name] = metrics
        else:
            print(f"Missing: {env_name}")

    if not env_data:
        print("No environment data found in", results_dir)
        return

    # -------------------------------------------------------------------------
    # Plot 1: Returns Across Environments
    # -------------------------------------------------------------------------
    def plot_returns(ax, metrics, env_name):
        mean_ret, std_ret = extract_metric(metrics, "returned_discounted_episode_returns")
        if mean_ret is None:
            mean_ret, std_ret = extract_metric(metrics, "returned_episode_returns")
        if mean_ret is not None:
            x = np.arange(len(mean_ret))
            ax.plot(x, mean_ret, color="tab:blue", lw=2, label="Discounted Return")
            ax.fill_between(x, mean_ret - std_ret, mean_ret + std_ret, color="tab:blue", alpha=0.2)
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=10)
        ax.set_ylabel("Discounted Return", fontsize=10)

    make_multi_env_grid(
        env_data,
        plot_returns,
        title="PPO Policy Learning Returns Across Environments",
        save_path_pdf=os.path.join(save_dir, "returns_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "returns_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Plot 2: Uncorrected E Across Environments
    # -------------------------------------------------------------------------
    def plot_uncorrected_e(ax, metrics, env_name):
        m_exact, s_exact = extract_metric(metrics, "E_exact")
        m_exp_u, s_exp_u = extract_metric(metrics, "E_exp_uncorr")
        m_smp_u, s_smp_u = extract_metric(metrics, "E_samp_uncorr")

        if m_exact is not None:
            x = np.arange(len(m_exact))
            ax.plot(x, m_exact, color="black", lw=2, label="Exact E")
            if m_exp_u is not None:
                ax.plot(x, m_exp_u, color="tab:orange", ls="--", lw=1.8, label="Expected Uncorr E")
            if m_smp_u is not None:
                ax.plot(x, m_smp_u, color="tab:red", ls=":", lw=1.8, alpha=0.85, label="Sampled Uncorr E")
                if s_smp_u is not None and np.any(s_smp_u > 0):
                    ax.fill_between(x, np.maximum(m_smp_u - s_smp_u, 1e-8), m_smp_u + s_smp_u, color="tab:red", alpha=0.15)
            ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=10)
        ax.set_ylabel("E Loss (log scale)", fontsize=10)

    make_multi_env_grid(
        env_data,
        plot_uncorrected_e,
        title="Uncorrected E Loss: Exact vs Expected Uncorrected vs Sampled Uncorrected",
        save_path_pdf=os.path.join(save_dir, "uncorrected_E_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "uncorrected_E_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Plot 3: Corrected E Across Environments
    # -------------------------------------------------------------------------
    def plot_corrected_e(ax, metrics, env_name):
        m_exact, s_exact = extract_metric(metrics, "E_exact")
        m_exp_c, s_exp_c = extract_metric(metrics, "E_exp_corr")
        m_smp_c, s_smp_c = extract_metric(metrics, "E_samp_corr")

        if m_exact is not None:
            x = np.arange(len(m_exact))
            ax.plot(x, m_exact, color="black", lw=2, label="Exact E")
            if m_exp_c is not None:
                ax.plot(x, m_exp_c, color="tab:green", ls="--", lw=1.8, label="Expected Corr E")
            if m_smp_c is not None:
                ax.plot(x, m_smp_c, color="tab:purple", ls=":", lw=1.8, alpha=0.85, label="Sampled Corr E")
                if s_smp_c is not None and np.any(s_smp_c > 0):
                    ax.fill_between(x, np.maximum(m_smp_c - s_smp_c, 1e-8), m_smp_c + s_smp_c, color="tab:purple", alpha=0.15)
            ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=10)
        ax.set_ylabel("E Loss (log scale)", fontsize=10)

    make_multi_env_grid(
        env_data,
        plot_corrected_e,
        title="Corrected E Loss: Exact vs Expected Corrected vs Sampled Corrected",
        save_path_pdf=os.path.join(save_dir, "corrected_E_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "corrected_E_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Plot 4: Corrected Gradient Similarities
    # -------------------------------------------------------------------------
    def plot_grad_sim_corr(ax, metrics, env_name):
        m_exp_c, _ = extract_metric(metrics, "rho_exp_corr_exact")
        m_smp_c, s_smp_c = extract_metric(metrics, "rho_samp_corr_exact")

        if m_smp_c is not None:
            x = np.arange(len(m_smp_c))
            if m_exp_c is not None:
                ax.plot(x, m_exp_c, color="tab:green", ls="--", lw=1.8, label="Expected Corr vs Exact")
            ax.plot(x, m_smp_c, color="tab:purple", lw=2, label="Sampled Corr vs Exact")
            if s_smp_c is not None and np.any(s_smp_c > 0):
                ax.fill_between(x, m_smp_c - s_smp_c, m_smp_c + s_smp_c, color="tab:purple", alpha=0.2)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=10)
        ax.set_ylabel("Cosine Similarity (rho)", fontsize=10)

    make_multi_env_grid(
        env_data,
        plot_grad_sim_corr,
        title="Gradient Alignment (Corrected): Cosine Similarity to Exact E Gradient",
        save_path_pdf=os.path.join(save_dir, "gradient_similarity_corrected_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "gradient_similarity_corrected_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Plot 5: Uncorrected Gradient Similarities
    # -------------------------------------------------------------------------
    def plot_grad_sim_uncorr(ax, metrics, env_name):
        m_exp_u, _ = extract_metric(metrics, "rho_exp_uncorr_exact")
        m_smp_u, s_smp_u = extract_metric(metrics, "rho_samp_uncorr_exact")

        if m_smp_u is not None:
            x = np.arange(len(m_smp_u))
            if m_exp_u is not None:
                ax.plot(x, m_exp_u, color="tab:orange", ls="--", lw=1.8, label="Expected Uncorr vs Exact")
            ax.plot(x, m_smp_u, color="tab:red", lw=2, label="Sampled Uncorr vs Exact")
            if s_smp_u is not None and np.any(s_smp_u > 0):
                ax.fill_between(x, m_smp_u - s_smp_u, m_smp_u + s_smp_u, color="tab:red", alpha=0.2)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=10)
        ax.set_ylabel("Cosine Similarity (rho)", fontsize=10)

    make_multi_env_grid(
        env_data,
        plot_grad_sim_uncorr,
        title="Gradient Alignment (Uncorrected): Cosine Similarity to Exact E Gradient",
        save_path_pdf=os.path.join(save_dir, "gradient_similarity_uncorrected_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "gradient_similarity_uncorrected_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Summary Table CSV
    # -------------------------------------------------------------------------
    summary_rows = []
    for env_name in ALL_ENVS:
        metrics = env_data.get(env_name)
        if metrics is None:
            continue
        row = {"Environment": env_name}
        for k in ["E_exact", "E_exp_corr", "E_exp_uncorr", "E_samp_corr", "E_samp_uncorr", "rho_samp_corr_exact", "rho_samp_uncorr_exact", "E_deficit"]:
            m, s = extract_metric(metrics, k)
            if m is not None:
                row[f"{k}_final_mean"] = float(m[-1])
                row[f"{k}_final_std"] = float(s[-1]) if s is not None else 0.0
        summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(save_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved summary table to {csv_path}")


if __name__ == "__main__":
    main()
