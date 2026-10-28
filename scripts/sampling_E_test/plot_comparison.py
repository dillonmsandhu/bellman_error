"""
scripts/sampling_E_test/plot_comparison.py

Consolidates and plots Sampling E Test results across benchmark environments.
Compares the 3 formulations against Exact Matrix E:
1. Oct 8 (Standard / Full-Branch Start-Anchored)
2. Forward Only
3. V_aux

Figures generated:
1. aggregate_gradient_metrics_across_envs: Grouped bar chart comparing Cosine Alignment, MSE, and Variance
2. gradient_mse_across_envs: Time series of Gradient MSE to Exact E across updates
3. gradient_cosine_across_envs: Time series of Cosine Similarity to Exact E across updates
4. loss_E_variants_across_envs: Exact E vs Oct 8 vs Forward Only vs V_aux values
5. returns_across_envs: Start-state value V^pi(s0) across updates
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

trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

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
    "fourrooms-dense": "FourRooms (Dense)",
}

VAR_COLORS = {
    "oct8": "#2ca02c",      # Green
    "fwd": "#00bcd4",       # Cyan/Teal
    "v_aux": "#9467bd",     # Purple
    "exact": "#000000",     # Black
}

VAR_LABELS = {
    "oct8": "Oct 8",
    "fwd": "Forward Only",
    "v_aux": "V_aux",
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
    plt.close()
    print(f"Saved: {save_path_pdf}")


def plot_aggregate_gradient_metrics(env_data: dict, save_dir: str):
    """
    Consolidated Multi-Environment Summary of Gradient Metrics:
    Panel 1: Cosine Similarity Alignment to Exact E
    Panel 2: Gradient MSE to Exact E (Log Scale)
    Panel 3: Trajectory Gradient Variance (Log Scale)
    """
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    fig, (ax_cos, ax_mse, ax_bv) = plt.subplots(3, 1, figsize=(max(12, n_envs * 2.2), 14), dpi=200)

    x = np.arange(n_envs)
    width = 0.24

    variants = [
        ("oct8", -width, VAR_COLORS["oct8"], "Oct 8"),
        ("fwd", 0.0, VAR_COLORS["fwd"], "Forward Only"),
        ("v_aux", width, VAR_COLORS["v_aux"], "V_aux"),
    ]

    for v_key, offset, color, label in variants:
        variances = []
        mses = []
        cosines = []

        for e in valid_envs:
            m = env_data[e]
            v_traj_m, _ = extract_metric(m, f"var_traj_samp_{v_key}")
            if v_traj_m is None:
                v_traj_m, _ = extract_metric(m, f"var_samp_{v_key}")
            mse_m, _ = extract_metric(m, f"mse_samp_{v_key}")
            cos_m, _ = extract_metric(m, f"rho_samp_{v_key}_exact")

            val_v = float(np.nanmean(v_traj_m)) if v_traj_m is not None else 1e-12
            val_mse = float(np.nanmean(mse_m)) if mse_m is not None else 1e-12
            val_cos = float(np.nanmean(cos_m)) if cos_m is not None else 0.0

            variances.append(max(1e-12, val_v))
            mses.append(max(1e-12, val_mse))
            cosines.append(val_cos)

        # Panel 1: Cosine Alignment
        ax_cos.bar(
            x + offset, cosines, width,
            label=label,
            color=color, alpha=0.85, edgecolor="black", lw=0.6,
        )

        # Panel 2: Gradient MSE
        ax_mse.bar(
            x + offset, mses, width,
            label=label,
            color=color, alpha=0.85, edgecolor="black", lw=0.6,
        )

        # Panel 3: Trajectory Gradient Variance
        ax_bv.bar(
            x + offset, variances, width,
            label=label,
            color=color, alpha=0.85, edgecolor="black", lw=0.6,
        )

    # Cosine Alignment Subplot
    ax_cos.set_xticks(x)
    ax_cos.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax_cos.set_ylim(-0.2, 1.05)
    ax_cos.axhline(1.0, color="black", linestyle=":", lw=1.2, alpha=0.7)
    ax_cos.axhline(0.0, color="gray", linestyle="-", lw=0.8, alpha=0.5)
    ax_cos.set_ylabel("Cosine Similarity (rho)", fontsize=11, fontweight="bold")
    ax_cos.set_title("Sample Gradient Cosine Alignment to Exact E", fontsize=12, fontweight="bold")
    ax_cos.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax_cos.legend(loc="lower right", ncol=3, fontsize=9.0, frameon=True)

    # Gradient MSE Subplot
    ax_mse.set_xticks(x)
    ax_mse.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax_mse.set_yscale("log")
    ax_mse.set_ylabel("Gradient MSE (Log Scale)\nMSE to exact E gradient", fontsize=11, fontweight="bold")
    ax_mse.set_title("Gradient MSE to Exact E Across Environments", fontsize=12, fontweight="bold")
    ax_mse.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax_mse.legend(loc="upper right", ncol=3, fontsize=9.0, frameon=True)

    # Variance Subplot
    ax_bv.set_xticks(x)
    ax_bv.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax_bv.set_yscale("log")
    ax_bv.set_ylabel("Trajectory Gradient Variance (Log Scale)\nSpread across rollout trajectories", fontsize=11, fontweight="bold")
    ax_bv.set_title("Trajectory Gradient Variance Across Environments", fontsize=12, fontweight="bold")
    ax_bv.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax_bv.legend(loc="upper right", ncol=3, fontsize=9.0, frameon=True)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "aggregate_gradient_metrics_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregate gradient metrics: {pdf_path}")


def compute_metric_auc(metrics: dict, key: str) -> tuple[float, float]:
    """
    Computes normalized trapezoidal Area Under the Curve (AUC) across update steps.
    Returns (mean_auc, sem_auc) across seeds.
    """
    if metrics is None or key not in metrics:
        return 0.0, 0.0
    arr = np.asarray(metrics[key])
    if arr.ndim == 1:
        steps = len(arr)
        val = float(trapz_fn(arr) / max(1, steps - 1)) if steps > 1 else float(arr[0])
        return val, 0.0
    elif arr.ndim >= 2:
        aucs = []
        for s in range(arr.shape[0]):
            series = arr[s]
            steps = len(series)
            auc = float(trapz_fn(series) / max(1, steps - 1)) if steps > 1 else float(series[0])
            aucs.append(auc)
        aucs = np.asarray(aucs, dtype=float)
        mean_auc = float(np.nanmean(aucs))
        sem_auc = float(np.nanstd(aucs, ddof=1) / np.sqrt(len(aucs))) if len(aucs) > 1 else 0.0
        return mean_auc, sem_auc
    return 0.0, 0.0


def plot_cosine_auc_bars(env_data: dict, save_dir: str):
    """
    Bar plot of Cosine Similarity AUC per environment to exact matrix E gradient.
    Sized and formatted for ICML columns / margins.
    Generates both single-column (3.25 in) and double-column (6.75 in) versions.
    """
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    variants = [
        ("oct8", VAR_COLORS["oct8"], "Oct 8"),
        ("fwd", VAR_COLORS["fwd"], "Forward Only"),
        ("v_aux", VAR_COLORS["v_aux"], "V_aux"),
    ]
    n_vars = len(variants)

    # Gather AUC and SEM data
    auc_data = {v_key: [] for v_key, _, _ in variants}
    sem_data = {v_key: [] for v_key, _, _ in variants}

    for e in valid_envs:
        m = env_data[e]
        for v_key, _, _ in variants:
            mean_auc, sem_auc = compute_metric_auc(m, f"rho_samp_{v_key}_exact")
            auc_data[v_key].append(mean_auc)
            sem_data[v_key].append(sem_auc)

    env_labels = [ENV_TITLES.get(e, e) for e in valid_envs]

    # Style configs for ICML:
    # 1. Full-width / double-column: 6.75 in width (ideal for 2-column span or large margins)
    # 2. Single-column / column-width: 3.25 in width (ICML column width is 3.25 in)
    configs = [
        {
            "suffix": "",
            "figsize": (6.75, 2.7),
            "fontsize_title": 10.0,
            "fontsize_label": 9.0,
            "fontsize_tick": 8.0,
            "fontsize_legend": 8.0,
            "rotation": 22,
            "capsize": 2.5,
            "total_width": 0.76,
            "legend_y": 1.22,
            "y_max": 1.15,
        },
        {
            "suffix": "_single_col",
            "figsize": (3.25, 2.4),
            "fontsize_title": 7.5,
            "fontsize_label": 7.0,
            "fontsize_tick": 6.0,
            "fontsize_legend": 6.0,
            "rotation": 35,
            "capsize": 1.8,
            "total_width": 0.80,
            "legend_y": 1.25,
            "y_max": 1.15,
        },
    ]

    for cfg in configs:
        fig, ax = plt.subplots(figsize=cfg["figsize"], dpi=300)
        indices = np.arange(n_envs)
        bar_w = cfg["total_width"] / n_vars

        for i, (v_key, color, label) in enumerate(variants):
            offset = (i - (n_vars - 1) / 2) * bar_w
            ax.bar(
                indices + offset,
                auc_data[v_key],
                bar_w,
                yerr=sem_data[v_key],
                label=label,
                color=color,
                alpha=0.90,
                edgecolor="#222222",
                linewidth=0.5,
                capsize=cfg["capsize"],
                error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#222222"},
            )

        ax.axhline(0.0, color="#555555", linestyle="-", lw=0.6, alpha=0.7)
        ax.axhline(1.0, color="#888888", linestyle=":", lw=0.8, alpha=0.8)
        ax.set_xticks(indices)
        ax.set_xticklabels(env_labels, rotation=cfg["rotation"], ha="right", fontsize=cfg["fontsize_tick"])
        ax.set_ylabel(r"Cosine Alignment AUC ($\rho$)", fontsize=cfg["fontsize_label"])
        ax.set_ylim(-0.35, cfg["y_max"])
        ax.set_title("Sample Gradient Cosine Alignment to Exact E (AUC)", fontsize=cfg["fontsize_title"], fontweight="bold", pad=20)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, cfg["legend_y"]),
            ncol=3,
            fontsize=cfg["fontsize_legend"],
            frameon=False,
            handlelength=1.2,
            handletextpad=0.3,
            columnspacing=1.0,
        )

        plt.tight_layout(pad=0.2)
        pdf_out = os.path.join(save_dir, f"cosine_auc_per_env{cfg['suffix']}.pdf")
        png_out = os.path.join(save_dir, f"cosine_auc_per_env{cfg['suffix']}.png")
        plt.savefig(pdf_out, bbox_inches="tight")
        plt.savefig(png_out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {pdf_out} and {png_out}")


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
    # Central Figure: Aggregate Gradient Metrics Across Environments
    # -------------------------------------------------------------------------
    plot_aggregate_gradient_metrics(env_data, save_dir)

    # -------------------------------------------------------------------------
    # Bar Chart: Cosine Similarity AUC per Environment (ICML Margins)
    # -------------------------------------------------------------------------
    plot_cosine_auc_bars(env_data, save_dir)

    # -------------------------------------------------------------------------
    # Plot: Gradient MSE Time Series Across Environments
    # -------------------------------------------------------------------------
    def plot_grad_mse(ax, metrics, env_name):
        m_o8, s_o8 = extract_metric(metrics, "mse_samp_oct8")
        m_fwd, s_fwd = extract_metric(metrics, "mse_samp_fwd")
        m_v, s_v = extract_metric(metrics, "mse_samp_v_aux")

        if m_o8 is not None:
            x = np.arange(len(m_o8))
            ax.plot(x, m_o8, color=VAR_COLORS["oct8"], lw=2.0, label="Oct 8")
            if s_o8 is not None and np.any(s_o8 > 0):
                ax.fill_between(x, np.maximum(m_o8 - s_o8, 1e-12), m_o8 + s_o8, color=VAR_COLORS["oct8"], alpha=0.18)
        if m_fwd is not None:
            x = np.arange(len(m_fwd))
            ax.plot(x, m_fwd, color=VAR_COLORS["fwd"], lw=2.0, label="Forward Only")
            if s_fwd is not None and np.any(s_fwd > 0):
                ax.fill_between(x, np.maximum(m_fwd - s_fwd, 1e-12), m_fwd + s_fwd, color=VAR_COLORS["fwd"], alpha=0.18)
        if m_v is not None:
            x = np.arange(len(m_v))
            ax.plot(x, m_v, color=VAR_COLORS["v_aux"], lw=2.0, label="V_aux")
            if s_v is not None and np.any(s_v > 0):
                ax.fill_between(x, np.maximum(m_v - s_v, 1e-12), m_v + s_v, color=VAR_COLORS["v_aux"], alpha=0.18)

        ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Gradient MSE to Exact E", fontsize=9)

    make_multi_env_grid(
        env_data,
        plot_grad_mse,
        title="Gradient MSE to Exact E Across Training Updates (Lower is Better)",
        save_path_pdf=os.path.join(save_dir, "gradient_mse_across_envs.pdf"),
    )

    # -------------------------------------------------------------------------
    # Plot: Gradient Cosine Similarity Time Series Across Environments
    # -------------------------------------------------------------------------
    def plot_grad_cosine(ax, metrics, env_name):
        m_o8, s_o8 = extract_metric(metrics, "rho_samp_oct8_exact")
        m_fwd, s_fwd = extract_metric(metrics, "rho_samp_fwd_exact")
        m_v, s_v = extract_metric(metrics, "rho_samp_v_aux_exact")

        if m_o8 is not None:
            x = np.arange(len(m_o8))
            ax.plot(x, m_o8, color=VAR_COLORS["oct8"], lw=2.0, label="Oct 8")
        if m_fwd is not None:
            x = np.arange(len(m_fwd))
            ax.plot(x, m_fwd, color=VAR_COLORS["fwd"], lw=2.0, label="Forward Only")
        if m_v is not None:
            x = np.arange(len(m_v))
            ax.plot(x, m_v, color=VAR_COLORS["v_aux"], lw=2.0, label="V_aux")

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
    )

    # -------------------------------------------------------------------------
    # Plot: Objective Value Curves Across Environments
    # -------------------------------------------------------------------------
    def plot_loss_e(ax, metrics, env_name):
        m_exact, _ = extract_metric(metrics, "E_exact")
        m_o8, _ = extract_metric(metrics, "E_samp_oct8")
        m_fwd, _ = extract_metric(metrics, "E_samp_fwd")
        m_v, _ = extract_metric(metrics, "E_samp_v_aux")

        if m_exact is not None:
            x = np.arange(len(m_exact))
            ax.plot(x, m_exact, color="black", lw=2.2, label="Exact E")
            if m_o8 is not None:
                ax.plot(x, m_o8, color=VAR_COLORS["oct8"], ls="-.", lw=1.8, label="Oct 8")
            if m_fwd is not None:
                ax.plot(x, m_fwd, color=VAR_COLORS["fwd"], ls="--", lw=1.8, label="Forward Only")
            if m_v is not None:
                ax.plot(x, m_v, color=VAR_COLORS["v_aux"], ls=":", lw=1.8, label="V_aux")
            ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("E Objective (Log Scale)", fontsize=9)

    make_multi_env_grid(
        env_data,
        plot_loss_e,
        title="Sampled E Objective vs Exact E Across Training Updates",
        save_path_pdf=os.path.join(save_dir, "loss_E_variants_across_envs.pdf"),
    )

    # -------------------------------------------------------------------------
    # Plot: Returns Across Environments (Exact V_start, Trained via exact_E)
    # -------------------------------------------------------------------------
    def plot_returns(ax, metrics, env_name):
        mean_ret, std_ret = extract_metric(metrics, "V_start")
        if mean_ret is not None:
            x = np.arange(len(mean_ret))
            ax.plot(x, mean_ret, color="tab:blue", lw=2, label=r"$V^\pi(s_0)$ (exact_E)")
            if std_ret is not None and np.any(std_ret > 0):
                ax.fill_between(x, mean_ret - std_ret, mean_ret + std_ret, color="tab:blue", alpha=0.2)
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel(r"Policy Value $V^\pi(s_0)$ (exact_E)", fontsize=9)

    make_multi_env_grid(
        env_data,
        plot_returns,
        title=r"Exact Start-State Policy Value $V^\pi(s_0)$ Across Environments (Trained via exact_E)",
        save_path_pdf=os.path.join(save_dir, "returns_across_envs.pdf"),
    )

    # -------------------------------------------------------------------------
    # Summary Table CSV
    # -------------------------------------------------------------------------
    summary_rows = []
    metric_keys = [
        "E_exact",
        "E_samp_oct8", "E_samp_fwd", "E_samp_v_aux",
        "var_samp_oct8", "var_samp_fwd", "var_samp_v_aux",
        "mse_samp_oct8", "mse_samp_fwd", "mse_samp_v_aux",
        "rho_samp_oct8_exact", "rho_samp_fwd_exact", "rho_samp_v_aux_exact",
    ]

    for env_name in ALL_ENVS:
        metrics = env_data.get(env_name)
        if metrics is None:
            continue
        row = {"Environment": env_name}
        for k in metric_keys:
            m, s = extract_metric(metrics, k)
            if m is not None:
                row[f"{k}_mean"] = float(np.nanmean(m))
                row[f"{k}_final"] = float(m[-1])
        # Also record AUC for cosine similarities
        for v_key in ["oct8", "fwd", "v_aux"]:
            auc_m, auc_s = compute_metric_auc(metrics, f"rho_samp_{v_key}_exact")
            row[f"rho_samp_{v_key}_exact_auc"] = auc_m
            row[f"rho_samp_{v_key}_exact_auc_sem"] = auc_s
        summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(save_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved comprehensive summary table to {csv_path}")


if __name__ == "__main__":
    main()
