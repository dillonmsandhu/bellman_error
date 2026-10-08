"""
scripts/sampling_E_test/plot_comparison.py

Consolidates and plots Sampling E Test results across all benchmark environments.
Generates comprehensive publication figures comparing the 3 formulations:
1. Sept 14 (Uncorrected):    L = (1-gamma) E[e^2] + (gamma/2) E[(e - e')^2]
2. Oct 5 (Corrected):        L = (1-gamma) E[e^2] + (gamma/2) E[(e - e')^2] + (gamma/2)(E[e^2] - E[e'^2])
3. Oct 8 (Start-Anchored):   L = (1-gamma) E[e^2] + (gamma/2) E[(e - e')^2] + (gamma/2) nu E_{s0}[e(s0)^2]

Figures generated:
1. bias_variance_decomposition_across_envs: Grouped bar chart comparing Variance, Bias^2, and total MSE
2. gradient_mse_across_envs: Time series of Gradient MSE to Exact E across updates
3. gradient_cosine_across_envs: Time series of Cosine Similarity to Exact E across updates
4. loss_E_variants_across_envs: Exact E vs Sept 14 vs Oct 5 vs Oct 8 values
5. returns_across_envs: Episode returns across all benchmark environments
6. summary_metrics.csv: Full statistical summary table across environments
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

VAR_COLORS = {
    "sep14": "#1f77b4",     # Blue
    "corr": "#d62728",      # Red
    "oct8": "#2ca02c",      # Green
    "exact": "#000000",     # Black
}

VAR_LABELS = {
    "sep14": "Sept 14 (Uncorrected)",
    "corr": "Oct 5 (Corrected / High Var)",
    "oct8": "Oct 8 (Start-Anchored / Exact)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate consolidated multi-environment Sampling E comparison plots")
    parser.add_argument("--results_dir", type=str, default="results/sampling_E_test", help="Path to results directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save figures (defaults to results_dir)")
    return parser.parse_args()


def load_env_metrics(results_dir: str, env_name: str):
    """Loads out.pkl for a given environment."""
    candidate = os.path.join(results_dir, env_name)
    if os.path.isdir(candidate):
        matches = [candidate]
    else:
        pattern = os.path.join(results_dir, f"*{env_name}*")
        matches = glob.glob(pattern)

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
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=8, frameon=True)
        else:
            ax.text(0.5, 0.5, "Data Not Found", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path_pdf, bbox_inches="tight")
    plt.savefig(save_path_png, bbox_inches="tight", dpi=250)
    plt.close()
    print(f"Saved: {save_path_pdf} and {save_path_png}")


def plot_bias_variance_decomposition(env_data: dict, save_dir: str):
    """
    Plots the central Bias-Variance decomposition bar chart across all environments:
    Total Bar Height = MSE
    Solid Segment = Sample Variance Var(g)
    Hatched Segment = Squared Bias Bias^2(g)
    Plus a companion panel for Cosine Similarity alignment.
    """
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    fig, (ax_bv, ax_cos) = plt.subplots(2, 1, figsize=(max(12, n_envs * 2.2), 10), dpi=200)

    x = np.arange(n_envs)
    width = 0.26

    # Extract final metrics for each variant
    variants = [
        ("sep14", -width, VAR_COLORS["sep14"], "Sept 14 (Uncorrected)"),
        ("corr", 0.0, VAR_COLORS["corr"], "Oct 5 (Corrected)"),
        ("oct8", width, VAR_COLORS["oct8"], "Oct 8 (Start-Anchored)"),
    ]

    for v_key, offset, color, label in variants:
        variances = []
        biases = []
        mses = []
        cosines = []

        for e in valid_envs:
            m = env_data[e]
            v_traj_m, _ = extract_metric(m, f"var_traj_samp_{v_key}")
            if v_traj_m is None:
                v_traj_m, _ = extract_metric(m, f"var_samp_{v_key}")
            mse_m, _ = extract_metric(m, f"mse_samp_{v_key}")
            cos_m, _ = extract_metric(m, f"rho_samp_{v_key}_exact")

            val_v = float(v_traj_m[-1]) if v_traj_m is not None else 1e-12
            val_mse = float(mse_m[-1]) if mse_m is not None else 1e-12
            val_cos = float(cos_m[-1]) if cos_m is not None else 0.0

            variances.append(max(1e-12, val_v))
            mses.append(max(1e-12, val_mse))
            cosines.append(val_cos)

        # Panel 1: Cosine Alignment of Total Average Gradient
        ax_cos.bar(
            x + offset, cosines, width,
            label=label,
            color=color, alpha=0.85, edgecolor="black", lw=0.6,
        )

        # Panel 2: Trajectory Gradient Variance (Spread across trajectories)
        ax_bv.bar(
            x + offset, variances, width,
            label=label,
            color=color, alpha=0.85, edgecolor="black", lw=0.6,
        )

    ax_cos.set_xticks(x)
    ax_cos.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax_cos.set_ylim(-0.2, 1.05)
    ax_cos.axhline(1.0, color="black", linestyle=":", lw=1.2, alpha=0.7)
    ax_cos.axhline(0.0, color="gray", linestyle="-", lw=0.8, alpha=0.5)
    ax_cos.set_ylabel("Cosine Similarity (rho)", fontsize=11, fontweight="bold")
    ax_cos.set_title("Full-Batch Average Gradient Cosine Alignment to Exact E", fontsize=12, fontweight="bold")
    ax_cos.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax_cos.legend(loc="lower right", ncol=3, fontsize=8.5, frameon=True)

    ax_bv.set_xticks(x)
    ax_bv.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax_bv.set_yscale("log")
    ax_bv.set_ylabel("Trajectory Gradient Variance (Log Scale)\nSpread across rollout trajectories", fontsize=11, fontweight="bold")
    ax_bv.set_title("Trajectory Gradient Variance Across Environments", fontsize=12, fontweight="bold")
    ax_bv.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax_bv.legend(loc="upper right", ncol=3, fontsize=8.5, frameon=True)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "bias_variance_decomposition_across_envs.pdf")
    png_path = os.path.join(save_dir, "bias_variance_decomposition_across_envs.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=250)
    plt.close()
    print(f"Saved Bias-Variance decomposition: {pdf_path} and {png_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir if args.save_dir else results_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("Sampling E Test: Consolidated Benchmark Analysis")
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
    # Central Figure: Bias-Variance Decomposition Across Environments
    # -------------------------------------------------------------------------
    plot_bias_variance_decomposition(env_data, save_dir)

    # -------------------------------------------------------------------------
    # Plot: Gradient MSE Time Series Across Environments
    # -------------------------------------------------------------------------
    def plot_grad_mse(ax, metrics, env_name):
        m_s14, s_s14 = extract_metric(metrics, "mse_samp_uncorr")
        m_corr, s_corr = extract_metric(metrics, "mse_samp_corr")
        m_o8, s_o8 = extract_metric(metrics, "mse_samp_oct8")

        if m_s14 is not None:
            x = np.arange(len(m_s14))
            ax.plot(x, m_s14, color=VAR_COLORS["sep14"], lw=1.8, label="Sept 14 (Uncorr)")
            if s_s14 is not None and np.any(s_s14 > 0):
                ax.fill_between(x, np.maximum(m_s14 - s_s14, 1e-12), m_s14 + s_s14, color=VAR_COLORS["sep14"], alpha=0.15)
        if m_corr is not None:
            x = np.arange(len(m_corr))
            ax.plot(x, m_corr, color=VAR_COLORS["corr"], lw=1.8, label="Oct 5 (Corr)")
            if s_corr is not None and np.any(s_corr > 0):
                ax.fill_between(x, np.maximum(m_corr - s_corr, 1e-12), m_corr + s_corr, color=VAR_COLORS["corr"], alpha=0.15)
        if m_o8 is not None:
            x = np.arange(len(m_o8))
            ax.plot(x, m_o8, color=VAR_COLORS["oct8"], lw=2.2, label="Oct 8 (Start-Anc)")
            if s_o8 is not None and np.any(s_o8 > 0):
                ax.fill_between(x, np.maximum(m_o8 - s_o8, 1e-12), m_o8 + s_o8, color=VAR_COLORS["oct8"], alpha=0.18)

        ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Gradient MSE to Exact E", fontsize=9)

    make_multi_env_grid(
        env_data,
        plot_grad_mse,
        title="Gradient MSE to Exact E Across Training Updates (Lower is Better)",
        save_path_pdf=os.path.join(save_dir, "gradient_mse_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "gradient_mse_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Plot: Gradient Cosine Similarity Time Series Across Environments
    # -------------------------------------------------------------------------
    def plot_grad_cosine(ax, metrics, env_name):
        m_s14, s_s14 = extract_metric(metrics, "rho_samp_uncorr_exact")
        m_corr, s_corr = extract_metric(metrics, "rho_samp_corr_exact")
        m_o8, s_o8 = extract_metric(metrics, "rho_samp_oct8_exact")

        if m_s14 is not None:
            x = np.arange(len(m_s14))
            ax.plot(x, m_s14, color=VAR_COLORS["sep14"], lw=1.8, label="Sept 14 (Uncorr)")
        if m_corr is not None:
            x = np.arange(len(m_corr))
            ax.plot(x, m_corr, color=VAR_COLORS["corr"], lw=1.8, label="Oct 5 (Corr)")
        if m_o8 is not None:
            x = np.arange(len(m_o8))
            ax.plot(x, m_o8, color=VAR_COLORS["oct8"], lw=2.2, label="Oct 8 (Start-Anc)")

        ax.set_ylim(-0.3, 1.05)
        ax.axhline(1.0, color="black", linestyle=":", lw=1.0, alpha=0.6)
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Cosine Similarity (rho)", fontsize=9)

    make_multi_env_grid(
        env_data,
        plot_grad_cosine,
        title="Sample Gradient Alignment to Exact E Across Training Updates",
        save_path_pdf=os.path.join(save_dir, "gradient_cosine_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "gradient_cosine_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Plot: Objective Value Curves Across Environments
    # -------------------------------------------------------------------------
    def plot_loss_e(ax, metrics, env_name):
        m_exact, _ = extract_metric(metrics, "E_exact")
        m_s14, _ = extract_metric(metrics, "E_samp_uncorr")
        m_corr, _ = extract_metric(metrics, "E_samp_corr")
        m_o8, _ = extract_metric(metrics, "E_samp_oct8")

        if m_exact is not None:
            x = np.arange(len(m_exact))
            ax.plot(x, m_exact, color="black", lw=2.2, label="Exact E")
            if m_s14 is not None:
                ax.plot(x, m_s14, color=VAR_COLORS["sep14"], ls="--", lw=1.8, label="Sampled Sept 14")
            if m_corr is not None:
                ax.plot(x, m_corr, color=VAR_COLORS["corr"], ls=":", lw=1.8, label="Sampled Oct 5")
            if m_o8 is not None:
                ax.plot(x, m_o8, color=VAR_COLORS["oct8"], ls="-.", lw=1.8, label="Sampled Oct 8")
            ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("E Objective (Log Scale)", fontsize=9)

    make_multi_env_grid(
        env_data,
        plot_loss_e,
        title="Sampled E Objective vs Exact E Across Training Updates",
        save_path_pdf=os.path.join(save_dir, "loss_E_variants_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "loss_E_variants_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Plot: Returns Across Environments
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
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Return", fontsize=9)

    make_multi_env_grid(
        env_data,
        plot_returns,
        title="PPO Policy Learning Returns Across Environments",
        save_path_pdf=os.path.join(save_dir, "returns_across_envs.pdf"),
        save_path_png=os.path.join(save_dir, "returns_across_envs.png"),
    )

    # -------------------------------------------------------------------------
    # Summary Table CSV
    # -------------------------------------------------------------------------
    summary_rows = []
    metric_keys = [
        "E_exact", "E_deficit",
        "E_samp_uncorr", "E_samp_corr", "E_samp_oct8",
        "var_samp_uncorr", "var_samp_corr", "var_samp_oct8",
        "bias_sq_samp_uncorr", "bias_sq_samp_corr", "bias_sq_samp_oct8",
        "mse_samp_uncorr", "mse_samp_corr", "mse_samp_oct8",
        "rho_samp_uncorr_exact", "rho_samp_corr_exact", "rho_samp_oct8_exact",
    ]

    for env_name in ALL_ENVS:
        metrics = env_data.get(env_name)
        if metrics is None:
            continue
        row = {"Environment": env_name}
        for k in metric_keys:
            m, s = extract_metric(metrics, k)
            if m is not None:
                row[f"{k}_final"] = float(m[-1])
        summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(save_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved comprehensive summary table to {csv_path}")


if __name__ == "__main__":
    main()
