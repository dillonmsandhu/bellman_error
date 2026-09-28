"""
sampled_e0_diagnostics.py
Specialized diagnostics and visualization for comparing the 4 sampled E variants at lambda = 0.0:
  1. sampled_E (baseline formulation)
  2. E_lambda_fixed (fixed / stop-gradient)
  3. E_lambda_geometric (geometric jumps)
  4. E_lambda_differentiable (differentiable scan)

Generates:
  - Policy Eval Tasks (random & fixed):
      * Value Learning Curves (nn_weighted_VE, log scale)
      * Greedy Accuracy Curves (nn_greedy_correct, [0, 1] scale)
  - PPO Tasks (ppo):
      * Performance Curves (V_start)
  - Master summary grid PDF & PNG across all 12 tasks
  - Individual high-res comparison plots per task
  - Quantitative metrics CSV with rankings
"""

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from notebooks.analyze_sweeps import (
    load_sweep_data,
    extract_best_configuration,
)

ALGO_STYLES = {
    "sampled_E": {
        "name": "Sampled E (Original)",
        "color": "#111111",      # Solid Black
        "linestyle": "-",
        "linewidth": 2.2,
    },
    "E_lambda_fixed": {
        "name": "E(λ) Fixed / Stop-Grad",
        "color": "#e74c3c",      # Crimson Red
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "E_lambda_geometric": {
        "name": "E(λ) Geometric Jumps",
        "color": "#2980b9",      # Blue
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    "E_lambda_differentiable": {
        "name": "E(λ) Differentiable Scan",
        "color": "#27ae60",      # Green
        "linestyle": ":",
        "linewidth": 2.0,
    },
}


def _find_algo_key(runs, algo_base):
    """Matches algo name with or without _0.0 suffix."""
    candidates = [
        f"{algo_base}_0.0",
        f"{algo_base}_0",
        algo_base,
    ]
    for c in candidates:
        if c in runs:
            return c
    # Fallback substring match
    for k in runs:
        if k.startswith(algo_base):
            return k
    return None


def _resolve_metric_key(metrics_dict, requested_metric):
    """Finds exact or fallback metric key in run metrics."""
    if requested_metric in metrics_dict:
        return requested_metric

    lower_map = {k.lower(): k for k in metrics_dict.keys()}
    if requested_metric.lower() in lower_map:
        return lower_map[requested_metric.lower()]

    # Metric fallbacks
    fallbacks = {
        "nn_weighted_VE": ["Weighted Value Errors", "weighted_VE", "E", "value_loss"],
        "nn_greedy_correct": ["Value Learning Greedy Accuracy", "greedy_correct", "greedy_acc"],
        "V_start": ["returned_discounted_episode_returns"],
    }
    for fb in fallbacks.get(requested_metric, []):
        if fb in metrics_dict:
            return fb
        if fb.lower() in lower_map:
            return lower_map[fb.lower()]

    return None


def generate_task_plots(sweep_dir, all_tasks_data, win_size=40, plots_dir=None):
    """
    Generates standalone high-res comparison figures for each task.
    """
    if plots_dir is None:
        plots_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    records = []

    for task_name, t_info in all_tasks_data.items():
        env_name = t_info["env_name"]
        pol = t_info["policy_type"].lower()
        runs = t_info["runs"]

        # Determine target metrics
        if pol in ["random", "fixed"]:
            metrics_to_plot = [
                ("nn_weighted_VE", "Value Learning Curve (Value Error)", "lower", True),
                ("nn_greedy_correct", "Greedy Action Accuracy", "higher", False),
            ]
        else:  # ppo
            metrics_to_plot = [
                ("V_start", "PPO Policy Performance (Return)", "higher", False),
            ]

        for metric_key, label, rank_order, log_scale in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(8, 5))
            has_data = False

            for algo_base, style in ALGO_STYLES.items():
                match_key = _find_algo_key(runs, algo_base)
                if not match_key:
                    continue

                s_data = runs[match_key]
                actual_metric = _resolve_metric_key(s_data["metrics"], metric_key)
                if not actual_metric:
                    continue

                try:
                    trajs, best_label, _, _ = extract_best_configuration(
                        s_data,
                        metric_key=actual_metric,
                        rank_by="final_window",
                        rank_order=rank_order,
                        window_size=win_size,
                    )
                    n_seeds, time_steps = trajs.shape
                    x = np.arange(time_steps)
                    mean = trajs.mean(axis=0)
                    std = trajs.std(axis=0)

                    c = style["color"]
                    ls = style["linestyle"]
                    lw = style["linewidth"]
                    lbl = style["name"]

                    ax.plot(x, mean, label=lbl, color=c, linestyle=ls, linewidth=lw)
                    if n_seeds > 1:
                        if log_scale:
                            lower = np.maximum(mean - std, mean * 0.05)
                        else:
                            lower = mean - std
                        upper = mean + std
                        ax.fill_between(x, lower, upper, color=c, alpha=0.15)

                    has_data = True

                    # Record quantitative score
                    auc_val = float(np.mean(mean))
                    final_val = float(mean[-max(1, win_size):].mean())
                    records.append({
                        "task": task_name,
                        "env": env_name,
                        "policy": pol,
                        "algo": algo_base,
                        "metric": actual_metric,
                        "final_window_mean": final_val,
                        "auc": auc_val,
                    })

                except Exception as e:
                    print(f"Error plotting {algo_base} for {task_name} ({metric_key}): {e}")
                    continue

            if has_data:
                ax.set_title(f"{env_name} ({pol.upper()}) - {label} [λ=0.0]", fontsize=12, fontweight="bold")
                ax.set_xlabel("Updates", fontsize=11)
                ax.set_ylabel(label, fontsize=11)
                if log_scale:
                    ax.set_yscale("log")
                ax.grid(True, which="both", linestyle="--", alpha=0.35)
                ax.legend(loc="best", fontsize=9, framealpha=0.9)
                fig.tight_layout()

                clean_metric = metric_key.replace("nn_", "").replace("returned_", "")
                out_path = os.path.join(plots_dir, f"{env_name}_{pol}_{clean_metric}.png")
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.close(fig)

    return records


def generate_master_summary_grid(sweep_dir, all_tasks_data, win_size=40):
    """
    Creates a master multi-panel figure consolidating:
      Columns:
        1. Random Policy - Value Error (log scale)
        2. Random Policy - Greedy Accuracy ([0, 1])
        3. Fixed Policy  - Value Error (log scale)
        4. Fixed Policy  - Greedy Accuracy ([0, 1])
        5. PPO Policy    - Performance Return
      Rows: Environments (EightRooms, FourRooms-misc, Whirlpool, MountainCar-v0)
    """
    envs = ["EightRooms", "FourRooms-misc", "Whirlpool", "MountainCar-v0"]
    # Filter to existing envs
    available_envs = [e for e in envs if any(t.startswith(e) for t in all_tasks_data.keys())]
    if not available_envs:
        available_envs = sorted(list(set(t["env_name"] for t in all_tasks_data.values())))

    num_rows = len(available_envs)
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=5,
        figsize=(22, 3.5 * num_rows),
        squeeze=False,
    )

    column_defs = [
        ("random", "nn_weighted_VE", "Value Error", "lower", True),
        ("random", "nn_greedy_correct", "Greedy Acc.", "higher", False),
        ("fixed", "nn_weighted_VE", "Value Error", "lower", True),
        ("fixed", "nn_greedy_correct", "Greedy Acc.", "higher", False),
        ("ppo", "V_start", "PPO Return", "higher", False),
    ]

    for col_idx, (pol, metric_key, col_title, _, _) in enumerate(column_defs):
        axes[0, col_idx].set_title(f"{pol.upper()}: {col_title}", fontsize=11, fontweight="bold", pad=10)

    for row_idx, env_name in enumerate(available_envs):
        axes[row_idx, 0].set_ylabel(f"{env_name}", fontsize=12, fontweight="bold")

        for col_idx, (pol, metric_key, col_title, rank_order, log_scale) in enumerate(column_defs):
            ax = axes[row_idx, col_idx]
            task_name = f"{env_name}_{pol}"
            if task_name not in all_tasks_data:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, color="gray")
                continue

            runs = all_tasks_data[task_name]["runs"]
            plotted = False

            for algo_base, style in ALGO_STYLES.items():
                match_key = _find_algo_key(runs, algo_base)
                if not match_key:
                    continue

                s_data = runs[match_key]
                actual_metric = _resolve_metric_key(s_data["metrics"], metric_key)
                if not actual_metric:
                    continue

                try:
                    trajs, _, _, _ = extract_best_configuration(
                        s_data,
                        metric_key=actual_metric,
                        rank_by="final_window",
                        rank_order=rank_order,
                        window_size=win_size,
                    )
                    n_seeds, time_steps = trajs.shape
                    x = np.arange(time_steps)
                    mean = trajs.mean(axis=0)
                    std = trajs.std(axis=0)

                    c = style["color"]
                    ls = style["linestyle"]
                    lw = style["linewidth"]
                    lbl = style["name"]

                    ax.plot(x, mean, label=lbl, color=c, linestyle=ls, linewidth=lw)
                    if n_seeds > 1:
                        if log_scale:
                            lower = np.maximum(mean - std, mean * 0.05)
                        else:
                            lower = mean - std
                        upper = mean + std
                        ax.fill_between(x, lower, upper, color=c, alpha=0.15)
                    plotted = True
                except Exception:
                    continue

            if plotted:
                if log_scale:
                    ax.set_yscale("log")
                if "greedy" in metric_key.lower():
                    ax.set_ylim(-0.02, 1.05)
                ax.grid(True, which="both", linestyle="--", alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color="gray")

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

            if row_idx == num_rows - 1:
                ax.set_xlabel("Updates", fontsize=10)

    fig.tight_layout()

    pdf_path = os.path.join(sweep_dir, "sampled_e0_master_summary.pdf")
    png_path = os.path.join(sweep_dir, "sampled_e0_master_summary.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved master PDF: {pdf_path}")
    print(f"Saved master PNG preview: {png_path}")
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for Sampled E(lambda=0.0) Comparison")
    parser.add_argument("--sweep-id", type=str, required=True, help="Sweep batch ID")
    parser.add_argument("--base-dir", type=str, default="results/lambda_sweep", help="Base sweep results directory")
    parser.add_argument("--win-size", type=int, default=40, help="Smoothing window size")
    args = parser.parse_args()

    sweep_dir = os.path.join(args.base_dir, args.sweep_id)
    if not os.path.exists(sweep_dir):
        print(f"Error: Sweep directory {sweep_dir} does not exist.")
        sys.exit(1)

    all_tasks_data = {}
    for env_name in os.listdir(sweep_dir):
        env_dir = os.path.join(sweep_dir, env_name)
        if not os.path.isdir(env_dir) or env_name.startswith("."):
            continue

        for policy_type in os.listdir(env_dir):
            pol_dir = os.path.join(env_dir, policy_type)
            if not os.path.isdir(pol_dir) or policy_type.startswith("."):
                continue

            task_name = f"{env_name}_{policy_type}"
            runs = {}
            for algo_dir in os.listdir(pol_dir):
                full_algo_dir = os.path.join(pol_dir, algo_dir)
                if not os.path.isdir(full_algo_dir) or algo_dir.startswith("."):
                    continue

                tuning_dir = os.path.join(full_algo_dir, "tuning")
                search_dir = tuning_dir if os.path.exists(tuning_dir) else full_algo_dir
                try:
                    s_data = load_sweep_data(search_dir)
                    if s_data and "metrics" in s_data and s_data["metrics"]:
                        runs[algo_dir] = s_data
                except Exception as e:
                    print(f"Warning: Failed to load sweep data for {algo_dir} in {task_name}: {e}")
                    continue

            if runs:
                all_tasks_data[task_name] = {
                    "env_name": env_name,
                    "policy_type": policy_type,
                    "runs": runs,
                }

    print(f"Loaded data for {len(all_tasks_data)} tasks.")
    if not all_tasks_data:
        print("No task data found.")
        sys.exit(1)

    plots_dir = os.path.join(sweep_dir, "plots")
    records = generate_task_plots(sweep_dir, all_tasks_data, win_size=args.win_size, plots_dir=plots_dir)
    generate_master_summary_grid(sweep_dir, all_tasks_data, win_size=args.win_size)

    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(sweep_dir, "sampled_e0_comparison_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics CSV: {csv_path}")

        print("\n" + "=" * 80)
        print("AVERAGE PERFORMANCE ACROSS TASKS [λ=0.0]")
        print("=" * 80)
        summary = df.groupby(["metric", "algo"])[["final_window_mean", "auc"]].mean().reset_index()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
