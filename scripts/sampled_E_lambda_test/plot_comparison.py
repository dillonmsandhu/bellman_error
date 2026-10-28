"""
scripts/sampled_E_lambda_test/plot_comparison.py

Consolidates and plots Sampling E(lambda) Test results across benchmark environments.
Compares the 3 formulations against Exact Matrix E(lambda):
1. Naive Sampling (Oct 8 Analog: differentiates both ends of jump)
2. Aux Version (v_aux debiasing)
3. Forward Only (drops backward branch)

Figures generated:
1. aggregate_gradient_metrics_across_envs.png: Grouped bar chart comparing Cosine Alignment, MSE, and Variance
2. gradient_mse_across_envs.png: Time series of Gradient MSE to Exact E(lambda) across updates
3. gradient_cosine_across_envs.png: Time series of Cosine Similarity to Exact E(lambda) across updates
4. gradient_variance_across_envs.png: Time series of Gradient Variance across updates
5. loss_E_variants_across_envs.png: Exact E(lambda) vs Naive vs Aux vs Forward Only values
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
    "naive": "#2ca02c",     # Green (Oct 8 analog)
    "aux": "#9467bd",       # Purple (v_aux)
    "fwd": "#00bcd4",       # Cyan/Teal (forward only)
    "exact": "#000000",     # Black
}

VAR_LABELS = {
    "naive": "Naive (Oct 8 Analog)",
    "aux": "Aux Version (v_aux)",
    "fwd": "Forward Only",
}


def load_env_metrics(results_dir: str, env_name: str) -> List[Dict[str, np.ndarray]]:
    pattern = os.path.join(results_dir, env_name, "*.pkl")
    files = glob.glob(pattern)
    metrics_list = []
    for fp in sorted(files):
        try:
            with open(fp, "rb") as f:
                data = cloudpickle.load(f)
                if isinstance(data, dict) and "metrics" in data:
                    metrics_list.append(data["metrics"])
                elif isinstance(data, dict):
                    metrics_list.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")
    return metrics_list


def aggregate_seeds(metrics_list: List[Dict[str, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    agg = {}
    for k in keys:
        try:
            arrays = [np.asarray(m[k], dtype=np.float64) for m in metrics_list if k in m]
            if arrays:
                min_len = min(len(a) for a in arrays)
                stacked = np.stack([a[:min_len] for a in arrays], axis=0)
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)
                agg[k] = (mean, std)
        except Exception:
            pass
    return agg


def plot_aggregate_bar_chart(all_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], out_dir: str):
    envs = [e for e in ALL_ENVS if e in all_data]
    if not envs:
        return

    n_envs = len(envs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    x = np.arange(n_envs)
    width = 0.25

    # 1. Cosine Similarity to Exact E(lambda)
    ax = axes[0]
    for idx, (var, label) in enumerate(VAR_LABELS.items()):
        key = f"rho_samp_{var}_exact"
        means = [np.mean(all_data[e][key][0]) if key in all_data[e] else np.nan for e in envs]
        errs = [np.mean(all_data[e][key][1]) if key in all_data[e] else 0.0 for e in envs]
        ax.bar(x + idx * width, means, width, yerr=errs, label=label, color=VAR_COLORS[var], capsize=3, edgecolor="black")

    ax.set_title(r"Cosine Similarity to Exact $\nabla E(\lambda)$", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$\cos(\bar{g}, g^*)$", fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=9)

    # 2. Gradient Variance
    ax = axes[1]
    for idx, (var, label) in enumerate(VAR_LABELS.items()):
        key = f"var_samp_{var}"
        means = [np.mean(all_data[e][key][0]) if key in all_data[e] else np.nan for e in envs]
        ax.bar(x + idx * width, means, width, label=label, color=VAR_COLORS[var], edgecolor="black")

    ax.set_title(r"Gradient Estimator Variance $\mathrm{Var}(\bar{g})$", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variance (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs, rotation=35, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Gradient MSE
    ax = axes[2]
    for idx, (var, label) in enumerate(VAR_LABELS.items()):
        key = f"mse_samp_{var}"
        means = [np.mean(all_data[e][key][0]) if key in all_data[e] else np.nan for e in envs]
        ax.bar(x + idx * width, means, width, label=label, color=VAR_COLORS[var], edgecolor="black")

    ax.set_title(r"Gradient MSE $||\bar{g} - g^*||^2$", fontsize=12, fontweight="bold")
    ax.set_ylabel("MSE (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(x + width)
    ax.set_xticklabels(envs, rotation=35, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Sampled E(lambda) Formulations: Exact Alignment & Variance across Benchmark Environments", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "aggregate_gradient_metrics_across_envs.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def plot_metric_time_series(
    all_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    metric_prefix: str,
    title: str,
    ylabel: str,
    filename: str,
    out_dir: str,
    log_scale: bool = False,
):
    envs = [e for e in ALL_ENVS if e in all_data]
    if not envs:
        return

    n_envs = len(envs)
    cols = min(3, n_envs)
    rows = int(np.ceil(n_envs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for i, env in enumerate(envs):
        ax = axes[i]
        for var, label in VAR_LABELS.items():
            key = f"{metric_prefix}_{var}" if f"{metric_prefix}_{var}" in all_data[env] else f"{metric_prefix}_{var}_exact"
            if key in all_data[env]:
                mean, std = all_data[env][key]
                mean = np.squeeze(np.asarray(mean))
                std = np.squeeze(np.asarray(std))
                if mean.ndim == 0:
                    mean = mean[None]
                    std = std[None]
                steps = np.arange(len(mean))
                ax.plot(steps, mean, label=label, color=VAR_COLORS[var], lw=1.8)
                if std is not None and np.any(std > 0):
                    ax.fill_between(steps, mean - std, mean + std, color=VAR_COLORS[var], alpha=0.15)

        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale("log")
        if i == 0 and len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(frameon=True, fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def export_summary_table(all_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], out_dir: str):
    envs = [e for e in ALL_ENVS if e in all_data]
    if not envs:
        return

    rows = []
    metric_keys = [
        "exact_e_loss",
        "loss_samp_naive", "loss_samp_aux", "loss_samp_fwd",
        "var_samp_naive", "var_samp_aux", "var_samp_fwd",
        "mse_samp_naive", "mse_samp_aux", "mse_samp_fwd",
        "rho_samp_naive_exact", "rho_samp_aux_exact", "rho_samp_fwd_exact",
    ]

    for env in envs:
        row = {"Environment": env}
        for k in metric_keys:
            if k in all_data[env]:
                m, _ = all_data[env][k]
                m_arr = np.asarray(m)
                row[f"{k}_mean"] = float(np.nanmean(m_arr))
                row[f"{k}_final"] = float(np.nanmean(m_arr[-1])) if len(m_arr) > 0 else np.nan
        # Cosine AUC
        for var in ["naive", "aux", "fwd"]:
            k = f"rho_samp_{var}_exact"
            if k in all_data[env]:
                m, _ = all_data[env][k]
                row[f"{k}_auc"] = float(np.nanmean(m))
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/sampled_E_lambda_test")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir if args.out_dir is not None else results_dir
    os.makedirs(out_dir, exist_ok=True)

    all_data = {}
    for env in ALL_ENVS:
        m_list = load_env_metrics(results_dir, env)
        if m_list:
            agg = aggregate_seeds(m_list)
            if agg:
                all_data[env] = agg

    if not all_data:
        subdirs = sorted(glob.glob(os.path.join(results_dir, "lambda_*")))
        if subdirs:
            print(f"Found lambda subdirectories: {subdirs}. Processing each...")
            for sdir in subdirs:
                sub_out = os.path.join(out_dir, os.path.basename(sdir))
                os.makedirs(sub_out, exist_ok=True)
                sub_data = {}
                for env in ALL_ENVS:
                    m_list = load_env_metrics(sdir, env)
                    if m_list:
                        agg = aggregate_seeds(m_list)
                        if agg:
                            sub_data[env] = agg
                if sub_data:
                    plot_aggregate_bar_chart(sub_data, sub_out)
                    plot_metric_time_series(sub_data, "rho_samp", r"Cosine Similarity to Exact $\nabla E(\lambda)$", r"$\cos(\bar{g}, g^*)$", "gradient_cosine_curves.png", sub_out)
                    plot_metric_time_series(sub_data, "var_samp", r"Gradient Variance $\mathrm{Var}(\bar{g})$", "Variance", "gradient_variance_curves.png", sub_out, log_scale=True)
                    plot_metric_time_series(sub_data, "mse_samp", r"Gradient MSE $||\bar{g} - g^*||^2$", "MSE", "gradient_mse_curves.png", sub_out, log_scale=True)
                    plot_metric_time_series(sub_data, "loss_samp", r"Sampled $E(\lambda)$ Loss", "Loss", "loss_curves.png", sub_out, log_scale=True)
                    export_summary_table(sub_data, sub_out)
            return

        print(f"No metric files found in {results_dir}.")
        return

    print(f"Loaded results for {len(all_data)} environments: {list(all_data.keys())}")
    plot_aggregate_bar_chart(all_data, out_dir)
    plot_metric_time_series(all_data, "rho_samp", r"Cosine Similarity to Exact $\nabla E(\lambda)$", r"$\cos(\bar{g}, g^*)$", "gradient_cosine_curves.png", out_dir)
    plot_metric_time_series(all_data, "var_samp", r"Gradient Variance $\mathrm{Var}(\bar{g})$", "Variance", "gradient_variance_curves.png", out_dir, log_scale=True)
    plot_metric_time_series(all_data, "mse_samp", r"Gradient MSE $||\bar{g} - g^*||^2$", "MSE", "gradient_mse_curves.png", out_dir, log_scale=True)
    plot_metric_time_series(all_data, "loss_samp", r"Sampled $E(\lambda)$ Loss", "Loss", "loss_curves.png", out_dir, log_scale=True)
    export_summary_table(all_data, out_dir)


if __name__ == "__main__":
    main()
