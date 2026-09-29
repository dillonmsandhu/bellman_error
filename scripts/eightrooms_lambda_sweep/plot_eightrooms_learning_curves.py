#!/usr/bin/env python3
"""
Publication-Quality Learning Curves Plotter for EightRooms Benchmark (Vector PDF Only).

Compares Ground Truth PPO and Exact E-Lambda PPO across lambda in {0.0, 0.8, 0.95, 0.99, 1.0}
on the full 2x2 factorial ablation of EightRooms:
  1. Discrete Sparse: EightRooms-misc
  2. Discrete Dense: EightRooms-Dense
  3. Continuous Sparse: continuous-eightrooms
  4. Continuous Dense: continuous-eightrooms-dense

Generates:
  - 2x2 or 1x4 panel vector PDF with TrueType fonts (Mean ± SEM across 8 seeds)
  - Detailed summary statistics CSV (Final Window Mean, SEM, AUC)
"""

import os
import sys
import glob
import argparse
import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ENV_DISPLAY_NAMES = {
    "EightRooms-misc": "Discrete Sparse (EightRooms)",
    "EightRooms": "Discrete Sparse (EightRooms)",
    "EightRooms-Dense": "Discrete Dense (EightRooms-Dense)",
    "eightrooms-dense": "Discrete Dense (EightRooms-Dense)",
    "continuous-eightrooms": "Continuous Sparse (continuous-eightrooms)",
    "continuous-eightrooms-dense": "Continuous Dense (continuous-eightrooms-dense)",
}

RUN_CONFIGS = [
    {
        "id": "ground_truth",
        "label": "Ground Truth PPO",
        "algo": "ppo/ground_truth",
        "subpath": "ground_truth",
        "color": "#111827",  # Deep charcoal / black
        "linestyle": "--",
        "linewidth": 2.2,
    },
    {
        "id": "lambda_0.0",
        "label": "Exact E(λ = 0.0) PPO",
        "algo": "ppo/exact_E_lambda",
        "subpath": "lambda_0.0",
        "color": "#2563EB",  # Royal Blue
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "id": "lambda_0.8",
        "label": "Exact E(λ = 0.8) PPO",
        "algo": "ppo/exact_E_lambda",
        "subpath": "lambda_0.8",
        "color": "#059669",  # Emerald Green
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "id": "lambda_0.95",
        "label": "Exact E(λ = 0.95) PPO",
        "algo": "ppo/exact_E_lambda",
        "subpath": "lambda_0.95",
        "color": "#D97706",  # Amber / Warm Orange
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "id": "lambda_0.99",
        "label": "Exact E(λ = 0.99) PPO",
        "algo": "ppo/exact_E_lambda",
        "subpath": "lambda_0.99",
        "color": "#7C3AED",  # Vibrant Purple
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "id": "lambda_1.0",
        "label": "Exact E(λ = 1.0) PPO",
        "algo": "ppo/exact_E_lambda",
        "subpath": "lambda_1.0",
        "color": "#DC2626",  # Crimson Red
        "linestyle": "-",
        "linewidth": 2.0,
    },
]


def find_latest_sweep_id(results_root="results"):
    """Find the most recent sweep ID under results/eightrooms_sweep/"""
    candidates = []
    patterns = [
        os.path.join(results_root, "eightrooms_sweep", "*"),
        os.path.join(results_root, "ppo", "ground_truth", "eightrooms_sweep", "*"),
        os.path.join(results_root, "ppo", "exact_E_lambda", "eightrooms_sweep", "*"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                candidates.append((os.path.getmtime(path), os.path.basename(path)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def locate_metrics_file(results_root, sweep_id, run_info, env_name):
    """Search for out.pkl across possible directory layouts."""
    possible_paths = [
        os.path.join(results_root, "eightrooms_sweep", sweep_id, run_info["subpath"], env_name, "out.pkl"),
        os.path.join(results_root, run_info["algo"], "eightrooms_sweep", sweep_id, run_info["subpath"], env_name, "out.pkl"),
        os.path.join(results_root, run_info["algo"], sweep_id, run_info["subpath"], env_name, "out.pkl"),
        os.path.join(results_root, run_info["algo"], sweep_id, env_name, "out.pkl"),
    ]

    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None


def load_run_metric(pkl_path, metric_key="V_start"):
    """Load metrics dictionary from out.pkl and extract specified series."""
    with open(pkl_path, "rb") as f:
        data = cloudpickle.load(f)

    if isinstance(data, dict):
        if "metrics" in data:
            metrics = data["metrics"]
        else:
            metrics = data
    else:
        return None

    if metric_key not in metrics:
        return None

    arr = np.asarray(metrics[metric_key])
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def plot_learning_curves(
    sweep_id,
    results_root="results",
    metric_key="V_start",
    output_dir=None,
    envs=None,
    confidence="sem",
    layout="2x2",
):
    if envs is None:
        envs = [
            "EightRooms-misc",
            "continuous-eightrooms",
            "EightRooms-Dense",
            "continuous-eightrooms-dense",
        ]

    if output_dir is None:
        output_dir = os.path.join(results_root, "eightrooms_sweep", sweep_id, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # High-quality publication styling for vector PDF
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if layout == "2x2":
        fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharey=False)
        flat_axes = axes.flatten()
    else:
        num_envs = len(envs)
        fig, axes = plt.subplots(1, num_envs, figsize=(5.5 * num_envs, 5.0), sharey=False)
        flat_axes = axes if num_envs > 1 else [axes]

    summary_records = []

    for idx, env_name in enumerate(envs):
        if idx >= len(flat_axes):
            break
        ax = flat_axes[idx]
        display_name = ENV_DISPLAY_NAMES.get(env_name, env_name)
        ax.set_title(display_name, pad=12, fontweight="bold")

        env_has_data = False

        for run_info in RUN_CONFIGS:
            pkl_path = locate_metrics_file(results_root, sweep_id, run_info, env_name)
            if not pkl_path:
                continue

            data = load_run_metric(pkl_path, metric_key)
            if data is None or data.size == 0:
                continue

            env_has_data = True
            n_seeds, n_steps = data.shape
            x_steps = np.arange(1, n_steps + 1)
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            sem = std / np.sqrt(max(n_seeds, 1))

            err = sem if confidence == "sem" else std

            # Line plot
            ax.plot(
                x_steps,
                mean,
                label=f"{run_info['label']} (N={n_seeds})",
                color=run_info["color"],
                linestyle=run_info["linestyle"],
                linewidth=run_info["linewidth"],
            )

            # Confidence band
            ax.fill_between(
                x_steps,
                mean - err,
                mean + err,
                color=run_info["color"],
                alpha=0.15,
            )

            # Summary metrics
            final_window = slice(-max(1, int(n_steps * 0.1)), None)
            final_mean = float(np.nanmean(mean[final_window]))
            final_sem = float(np.nanmean(sem[final_window]))
            auc = float(np.trapz(mean, x_steps))

            summary_records.append({
                "Environment": env_name,
                "Algorithm": run_info["label"],
                "Seeds": n_seeds,
                "Final_Window_Mean": final_mean,
                "Final_Window_SEM": final_sem,
                "AUC": auc,
            })

        ax.set_xlabel("PPO Update Iteration", labelpad=8)
        ylabel = "Start-State True Value $V(s_0)$" if metric_key == "V_start" else metric_key
        ax.set_ylabel(ylabel, labelpad=8)

        if not env_has_data:
            ax.text(
                0.5,
                0.5,
                "No Data Available Yet",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="gray",
                fontsize=12,
            )

    # Shared Legend
    handles, labels = flat_axes[0].get_legend_handles_labels()
    for ax in flat_axes[1:]:
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

    if handles:
        legend_y = 0.99 if layout == "2x2" else 1.08
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=6,
            frameon=True,
            facecolor="white",
            edgecolor="#e5e7eb",
            framealpha=0.95,
        )

    metric_title = "Start-State Value $V(s_0)$" if metric_key == "V_start" else metric_key
    title_y = 1.03 if layout == "2x2" else 1.16
    fig.suptitle(
        f"EightRooms Benchmark: [Discrete vs Continuous] x [Sparse vs Dense]\nMetric: {metric_title} (Mean ± {confidence.upper()})",
        y=title_y,
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    out_pdf = os.path.join(output_dir, f"learning_curves_{metric_key}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Learning curves saved to:")
    print(f"  -> Vector PDF: {out_pdf}")

    # Summary table CSV
    if summary_records:
        df = pd.DataFrame(summary_records)
        csv_path = os.path.join(output_dir, f"summary_{metric_key}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  -> Summary Table: {csv_path}")
        print("\nSummary Statistics:")
        print(df.to_string(index=False))

    return out_pdf


def main():
    parser = argparse.ArgumentParser(description="Plot EightRooms Learning Curves (PDF Only)")
    parser.add_argument("--sweep-id", type=str, default=None, help="Sweep ID identifier")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory root")
    parser.add_argument("--metric", type=str, default="V_start", help="Metric to plot (default: V_start)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for PDF")
    parser.add_argument("--confidence", choices=["sem", "std"], default="sem", help="Error band type")
    parser.add_argument("--layout", choices=["2x2", "1x4"], default="2x2", help="Subplot layout")

    args = parser.parse_args()

    sweep_id = args.sweep_id
    if not sweep_id:
        sweep_id = find_latest_sweep_id(args.results_dir)
        if not sweep_id:
            print(f"Error: No sweep directory found under {args.results_dir}. Provide --sweep-id.")
            sys.exit(1)
        print(f"Auto-detected latest sweep ID: {sweep_id}")

    plot_learning_curves(
        sweep_id=sweep_id,
        results_root=args.results_dir,
        metric_key=args.metric,
        output_dir=args.output_dir,
        confidence=args.confidence,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
