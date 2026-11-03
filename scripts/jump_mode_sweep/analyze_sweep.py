"""
scripts/jump_mode_sweep/analyze_sweep.py

Consolidates and visualizes PPO performance across the 3 algorithms:
1. e_lambda_drop: Geometric jump E(lambda=0.9) with drop_invalid_jump
2. e_lambda_mc:   Geometric jump E(lambda=0.9) with mc_invalid_jump
3. bsmc:          1-Step Bellman-Smoothed Monte Carlo with equivalent weight (lambda=0.9)

Across 9 benchmark environments and 5 random seeds.
Primary metric: Ground truth value of the start state V^pi(s_0) [V_start].

Outputs:
- learning_curves_V_start.png: 9-panel subplots showing V^pi(s_0) convergence across updates
- auc_performance_comparison.png: Grouped bar chart comparing AUC of V^pi(s_0)
- final_performance_comparison.png: Grouped bar chart comparing final achieved V^pi(s_0)
- summary_table.csv / summary_table.md: Detailed numerical mean +/- SEM tables
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import pickle
from typing import Dict, Any, List, Tuple

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

ALGORITHMS = {
    "e_lambda_drop": {
        "label": "E(λ=0.95) [Drop Invalid Jump]",
        "color": "#9467bd",  # Purple
    },
    "e_lambda_mc": {
        "label": "E(λ=0.95) [MC Invalid Jump]",
        "color": "#d62728",  # Red / Coral
    },
    "e_lambda_full_mc": {
        "label": "E(λ=0.95) [Full MC Invalid Jump]",
        "color": "#ff7f0e",  # Orange
    },
    "bsmc": {
        "label": "BSMC [1-step, Calibrated λ=0.95]",
        "color": "#1f77b4",  # Blue
    },
}


def load_sweep_data(results_dir: str, value_lambda: float = 0.95) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    lambda_dir = os.path.join(results_dir, f"lambda_{value_lambda}")
    all_data = {}

    for env in ALL_ENVS:
        env_dir = os.path.join(lambda_dir, env)
        if not os.path.isdir(env_dir):
            continue

        all_data[env] = {}
        for algo_key in ALGORITHMS.keys():
            pkl_path = os.path.join(env_dir, f"{algo_key}.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                    all_data[env][algo_key] = data
                except Exception as e:
                    print(f"Warning: Failed to load {pkl_path}: {e}")

    return all_data


def plot_learning_curves(all_data: Dict[str, Dict[str, Any]], out_dir: str, value_lambda: float = 0.95):
    envs = [e for e in ALL_ENVS if e in all_data and len(all_data[e]) > 0]
    if not envs:
        print("No environment data found for learning curves.")
        return

    n_envs = len(envs)
    cols = 3
    rows = int(np.ceil(n_envs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.8 * rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, env in enumerate(envs):
        ax = axes[idx]
        env_data = all_data[env]

        for algo_key, meta in ALGORITHMS.items():
            if algo_key not in env_data:
                continue

            v_arr = np.asarray(env_data[algo_key]["V_start"], dtype=np.float64)  # (n_seeds, num_updates)
            if v_arr.ndim == 1:
                v_arr = v_arr[None, :]

            n_seeds, n_updates = v_arr.shape
            x = np.arange(n_updates)
            mean_v = np.mean(v_arr, axis=0)
            sem_v = np.std(v_arr, axis=0) / np.sqrt(max(1, n_seeds))

            ax.plot(x, mean_v, label=meta["label"], color=meta["color"], lw=2.0)
            ax.fill_between(x, mean_v - sem_v, mean_v + sem_v, color=meta["color"], alpha=0.18)

        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_xlabel("PPO Updates", fontsize=9)
        ax.set_ylabel(r"Ground Truth $V^\pi(s_0)$", fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(frameon=True, fontsize=8, loc="lower right")

    for j in range(len(envs), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f"Policy Performance (Ground Truth Value $V^\\pi(s_0)$): Jump Modes vs BSMC ($\\lambda={value_lambda}$, 5 Seeds)",
        fontsize=13,
        y=1.01,
        fontweight="bold",
    )
    plt.tight_layout()
    out_file = os.path.join(out_dir, "learning_curves_V_start.png")
    pdf_file = os.path.join(out_dir, "learning_curves_V_start.pdf")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_bar_comparisons(all_data: Dict[str, Dict[str, Any]], out_dir: str, value_lambda: float = 0.95):
    envs = [e for e in ALL_ENVS if e in all_data and len(all_data[e]) > 0]
    if not envs:
        return

    n_envs = len(envs)
    algos = [a for a in ALGORITHMS.keys() if any(a in all_data[e] for e in envs)]
    n_algos = len(algos)
    x = np.arange(n_envs)
    width = min(0.20, 0.8 / max(1, n_algos))

    # 1. AUC Comparison
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    for a_idx, algo_key in enumerate(algos):
        meta = ALGORITHMS[algo_key]
        means = []
        sems = []
        for env in envs:
            if algo_key in all_data[env]:
                v_arr = np.asarray(all_data[env][algo_key]["V_start"], dtype=np.float64)
                if v_arr.ndim == 1:
                    v_arr = v_arr[None, :]
                # AUC per seed as mean over updates
                aucs = np.mean(v_arr, axis=1)
                means.append(float(np.mean(aucs)))
                sems.append(float(np.std(aucs) / np.sqrt(max(1, len(aucs)))))
            else:
                means.append(np.nan)
                sems.append(0.0)

        offset = (a_idx - (n_algos - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            label=meta["label"],
            color=meta["color"],
            capsize=3,
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_title(f"Area Under the Learning Curve (AUC of $V^\\pi(s_0)$): Jump Modes vs BSMC ($\\lambda={value_lambda}$, 5 Seeds)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Mean $V^\pi(s_0)$ Across Training", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=30, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    auc_file = os.path.join(out_dir, "auc_performance_comparison.png")
    plt.savefig(auc_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {auc_file}")

    # 2. Final Achieved Performance Comparison
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    for a_idx, algo_key in enumerate(algos):
        meta = ALGORITHMS[algo_key]
        means = []
        sems = []
        for env in envs:
            if algo_key in all_data[env]:
                v_arr = np.asarray(all_data[env][algo_key]["V_start"], dtype=np.float64)
                if v_arr.ndim == 1:
                    v_arr = v_arr[None, :]
                finals = v_arr[:, -1]
                means.append(float(np.mean(finals)))
                sems.append(float(np.std(finals) / np.sqrt(max(1, len(finals)))))
            else:
                means.append(np.nan)
                sems.append(0.0)

        offset = (a_idx - (n_algos - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            label=meta["label"],
            color=meta["color"],
            capsize=3,
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_title(f"Final Convergence Performance (Final $V^\\pi(s_0)$): Jump Modes vs BSMC ($\\lambda={value_lambda}$, 5 Seeds)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Final $V^\pi(s_0)$", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=30, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    final_file = os.path.join(out_dir, "final_performance_comparison.png")
    plt.savefig(final_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {final_file}")


def generate_summary_tables(all_data: Dict[str, Dict[str, Any]], out_dir: str):
    envs = [e for e in ALL_ENVS if e in all_data and len(all_data[e]) > 0]
    records = []

    for env in envs:
        row = {"Environment": env}
        for algo_key, meta in ALGORITHMS.items():
            if algo_key in all_data[env]:
                v_arr = np.asarray(all_data[env][algo_key]["V_start"], dtype=np.float64)
                if v_arr.ndim == 1:
                    v_arr = v_arr[None, :]
                # Final
                finals = v_arr[:, -1]
                m_fin = np.mean(finals)
                s_fin = np.std(finals) / np.sqrt(max(1, len(finals)))
                # AUC
                aucs = np.mean(v_arr, axis=1)
                m_auc = np.mean(aucs)
                s_auc = np.std(aucs) / np.sqrt(max(1, len(aucs)))

                row[f"{meta['label']} Final"] = f"{m_fin:.4f} +/- {s_fin:.4f}"
                row[f"{meta['label']} AUC"] = f"{m_auc:.4f} +/- {s_auc:.4f}"
            else:
                row[f"{meta['label']} Final"] = "N/A"
                row[f"{meta['label']} AUC"] = "N/A"
        records.append(row)

    df = pd.DataFrame(records)
    csv_file = os.path.join(out_dir, "summary_table.csv")
    md_file = os.path.join(out_dir, "summary_table.md")
    df.to_csv(csv_file, index=False)
    with open(md_file, "w") as f:
        f.write(df.to_markdown(index=False))

    print(f"Saved: {csv_file}")
    print(f"Saved: {md_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Jump Mode & BSMC Sweep Results")
    parser.add_argument("--results_dir", type=str, default="results/jump_mode_sweep", help="Root results directory")
    parser.add_argument("--value_lambda", type=float, default=0.95, help="Value lambda to analyze (default: 0.95)")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, f"lambda_{args.value_lambda}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"ANALYZING JUMP MODES VS BSMC SWEEP (lambda={args.value_lambda})")
    print(f"Results Directory: {out_dir}")
    print("=" * 80)

    all_data = load_sweep_data(args.results_dir, args.value_lambda)
    print(f"Found data for {len(all_data)} environments.")

    if not all_data:
        print("No results found to analyze.")
        return

    plot_learning_curves(all_data, out_dir, args.value_lambda)
    plot_bar_comparisons(all_data, out_dir, args.value_lambda)
    generate_summary_tables(all_data, out_dir)
    print("\n[DONE] Analysis and visualization complete.")


if __name__ == "__main__":
    main()
