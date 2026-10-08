"""
scripts/sampled_algo_sweep/plot_comparison.py

Aggregates results across all 10 environments from the sampled algorithm sweep:
  - MC (lambda=1.0)
  - TD(0)
  - TD(lambda=0.8)
  - E(0) [Corrected boundary loss]

Generates:
  1. 10-panel cross-environment comparison figure (PDF + PNG)
  2. summary_metrics.csv with final values, SEM, and AUC per algorithm
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

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

ALGORITHMS = [
    {"name": "MC", "display_name": "MC ($\\lambda=1.0$)", "color": "#1f77b4"},
    {"name": "TD0", "display_name": "TD(0)", "color": "#ff7f0e"},
    {"name": "TD_lambda", "display_name": "TD($\\lambda=0.8$)", "color": "#2ca02c"},
    {"name": "E0", "display_name": "E(0) [Corrected]", "color": "#d62728"},
]

LR_MULTIPLIERS = ["0.5x", "1.0x", "2.0x"]


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidated plot of sampled algorithm sweeps across all environments")
    parser.add_argument("--results_dir", type=str, default="results/sampled_algo_sweep", help="Directory containing env folders")
    parser.add_argument("--out_dir", type=str, default=None, help="Output plot directory (defaults to results_dir)")
    return parser.parse_args()


def load_env_results(env_dir: str):
    npz_path = os.path.join(env_dir, "results.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def find_best_condition(env_data, algo_name):
    best_mult = None
    best_score = -float("inf")
    best_arr = None
    metric_used = None

    for mult in LR_MULTIPLIERS:
        for metric in ["v_true_start", "v_pred_start", "mean_rew"]:
            key = f"{algo_name}_{mult}_{metric}"
            if key in env_data:
                arr = env_data[key]
                score = np.mean(arr[:, -5:])
                if score > best_score or best_mult is None:
                    best_score = score
                    best_mult = mult
                    best_arr = arr
                    metric_used = metric
                break

    return best_mult, best_score, best_arr, metric_used


def main():
    args = parse_args()
    results_dir = args.results_dir
    out_dir = args.out_dir if args.out_dir is not None else results_dir
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    # Setup 10-panel figure (2 rows x 5 columns)
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), dpi=150)
    axes_flat = axes.flatten()

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes_flat[idx]
        env_dir = os.path.join(results_dir, env_name)
        env_data = load_env_results(env_dir)

        display_name = ENV_DISPLAY_NAMES.get(env_name, env_name)
        ax.set_title(display_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if env_data is None:
            ax.text(0.5, 0.5, "Pending Run", ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
            continue

        for algo in ALGORITHMS:
            algo_name = algo["name"]
            best_mult, score, arr, metric = find_best_condition(env_data, algo_name)

            if arr is not None:
                x = np.arange(arr.shape[1])
                m = np.mean(arr, axis=0)
                sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
                color = algo["color"]
                label = f"{algo['display_name']} ({best_mult})"

                ax.plot(x, m, label=label, color=color, lw=2.0)
                ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.18)

                # Summary statistics
                final_mean = np.mean(arr[:, -5:])
                final_sem = np.std(np.mean(arr[:, -5:], axis=1)) / np.sqrt(arr.shape[0])
                auc = np.mean(m)
                summary_rows.append({
                    "environment": env_name,
                    "algorithm": algo_name,
                    "best_critic_lr": best_mult,
                    "final_score": final_mean,
                    "final_sem": final_sem,
                    "auc": auc,
                    "metric": metric,
                })

        ax.set_xlabel("PPO Updates", fontsize=10)
        if idx % 5 == 0:
            ax.set_ylabel("Value / Return", fontsize=10)
        ax.legend(frameon=True, fontsize=8, loc="best")

    plt.suptitle("Sampled Algorithm Comparison Across Environments (Best Critic LR per Algorithm)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    pdf_path = os.path.join(out_dir, "sampled_algo_cross_env_comparison.pdf")
    png_path = os.path.join(out_dir, "sampled_algo_cross_env_comparison.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"Saved cross-environment plot to:\n  - {pdf_path}\n  - {png_path}")

    # Save summary metrics table
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(out_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved summary metrics table to: {csv_path}")


if __name__ == "__main__":
    main()
