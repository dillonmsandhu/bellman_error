"""
sampled_lambda_diagnostics.py
Consolidates sweep results for the 4 sampled E(lambda) variants across all 12 tasks.
Generates:
  1. A 2-column master summary plot:
     - Left Column:  E(lambda=0.0) across all 4 sampling implementations:
                     [sampled_E, E_lambda_fixed, E_lambda_geometric, E_lambda_differentiable]
     - Right Column: E(lambda=0.9) across the 3 parameter-dependent implementations:
                     [E_lambda_fixed, E_lambda_geometric, E_lambda_differentiable]
  2. Quantitative evaluation tables ranking the sampling implementations to identify the best variant.
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


def generate_sampled_summary_grid(
    base_dir,
    all_tasks_data,
    win_size=40,
    pdf_filename="sampled_lambda_comparison_summary.pdf",
):
    """
    Generates a unified 2-column grid PDF & PNG across all 12 tasks:
      - Left column:  E(0) across all 4 sampling implementations
      - Right column: E(0.9) across the 3 parameter-dependent implementations
      - Shared Y-axis per row for direct comparison
    """
    if not all_tasks_data:
        print("No task data found to generate master summary grid.")
        return None, None

    policy_priority = {"random": 0, "fixed": 1, "ppo": 2, "hybrid": 3}
    def _task_sort_key(name):
        _info = all_tasks_data[name]
        _env = _info["env_name"]
        _pol = _info["policy_type"].lower()
        return (_env, policy_priority.get(_pol, 99))

    task_keys = sorted(list(all_tasks_data.keys()), key=_task_sort_key)
    num_tasks = len(task_keys)

    print(f"\n{'='*80}")
    print(f"Generating Master Sampled Comparison Summary ({num_tasks} Tasks) -> {pdf_filename}")
    print(f"{'='*80}")

    # Algorithm Color & Style Palette
    ALGO_STYLES = {
        "sampled_E": {
            "name": "Sampled E (Original)",
            "color": "#111111",      # Solid Black
            "linestyle": "-",
            "marker": None,
        },
        "E_lambda_fixed": {
            "name": "Fixed / Stop-Grad (Method 1)",
            "color": "#e74c3c",      # Crimson Red
            "linestyle": "--",
            "marker": None,
        },
        "E_lambda_geometric": {
            "name": "Geometric Jumps (Method 3)",
            "color": "#2980b9",      # Blue
            "linestyle": "-.",
            "marker": None,
        },
        "E_lambda_differentiable": {
            "name": "Differentiable Scan (Method 2)",
            "color": "#27ae60",      # Green
            "linestyle": ":",
            "marker": None,
        },
    }

    fig, axes = plt.subplots(
        nrows=num_tasks,
        ncols=2,
        figsize=(16, 3.4 * num_tasks),
        sharex=False,
        squeeze=False,
    )

    # Column Super-Headers
    axes[0, 0].set_title(r"$\mathbf{E(\lambda=0.0)}$: All 4 Sampling Implementations", fontsize=13, fontweight="bold", pad=12)
    axes[0, 1].set_title(r"$\mathbf{E(\lambda=0.9)}$: Parameter-Dependent Variants", fontsize=13, fontweight="bold", pad=12)

    all_table_records = []

    for row_idx, task_name in enumerate(task_keys):
        t_info = all_tasks_data[task_name]
        runs = t_info["runs"]
        pol = t_info["policy_type"].lower()

        if pol in ["ppo", "hybrid"]:
            metric_key = "V_start"
            rank_order = "higher"
            log_scale = False
            metric_label = r"$V_{\mathrm{start}}$"
        else:
            metric_key = "nn_weighted_VE"
            rank_order = "lower"
            log_scale = True
            metric_label = "Value Error (VE)"

        ax_0 = axes[row_idx, 0]
        ax_9 = axes[row_idx, 1]

        row_min_vals = []
        row_max_vals = []

        def _plot_col(ax, target_lambda):
            col_plotted = False
            for algo_base, meta in ALGO_STYLES.items():
                if target_lambda == "0.9" and algo_base == "sampled_E":
                    continue  # sampled_E only implements lambda=0

                # Match pseudo-algo names: e.g. E_lambda_fixed_0.0 or E_lambda_fixed_0.9 or sampled_E_0.0
                cand_keys = [
                    f"{algo_base}_{target_lambda}",
                    f"{algo_base}_{float(target_lambda):.1f}",
                    algo_base if target_lambda == "0.0" and algo_base == "sampled_E" else None,
                ]
                match_key = next((k for k in cand_keys if k and k in runs), None)
                if not match_key:
                    continue

                s_data = runs[match_key]
                try:
                    actual_metric = metric_key
                    if actual_metric not in s_data["metrics"]:
                        avail = {k.lower(): k for k in s_data["metrics"].keys()}
                        if actual_metric.lower() in avail:
                            actual_metric = avail[actual_metric.lower()]
                        elif actual_metric == "V_start" and "returned_discounted_episode_returns" in s_data["metrics"]:
                            actual_metric = "returned_discounted_episode_returns"

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

                    c = meta["color"]
                    ls = meta["linestyle"]
                    lbl = meta["name"]

                    ax.plot(x, mean, label=lbl, color=c, linestyle=ls, linewidth=2.0)
                    if n_seeds > 1:
                        if log_scale:
                            lower = np.maximum(mean - std, mean * 0.05)
                        else:
                            lower = mean - std
                        upper = mean + std
                        ax.fill_between(x, lower, upper, color=c, alpha=0.15)
                        row_min_vals.append(lower)
                        row_max_vals.append(upper)
                    else:
                        row_min_vals.append(mean)
                        row_max_vals.append(mean)

                    col_plotted = True

                    # Record stats for summary table
                    auc_val = float(np.mean(mean))
                    final_val = float(mean[-max(1, win_size):].mean())
                    all_table_records.append({
                        "task": task_name,
                        "env": t_info["env_name"],
                        "policy": pol,
                        "lambda": target_lambda,
                        "algo": algo_base,
                        "metric": actual_metric,
                        "final_perf": final_val,
                        "auc": auc_val,
                    })

                except Exception as e:
                    print(f"Error extracting {match_key} for task {task_name}: {e}")
                    continue

            return col_plotted

        _plot_col(ax_0, "0.0")
        _plot_col(ax_9, "0.9")

        # Synchronize Y-Axis bounds across the two subplots
        if row_min_vals and row_max_vals:
            all_low = np.concatenate([arr[arr > 0] if log_scale else arr for arr in row_min_vals] or [np.array([1e-3])])
            all_high = np.concatenate(row_max_vals)
            if len(all_low) > 0 and len(all_high) > 0:
                if log_scale:
                    ymin = max(np.percentile(all_low, 1) * 0.7, np.min(all_low) * 0.3)
                    ymax = np.percentile(all_high, 99) * 1.3
                else:
                    span = np.max(all_high) - np.min(all_low)
                    ymin = np.min(all_low) - 0.05 * span
                    ymax = np.max(all_high) + 0.05 * span

                if ymax > ymin and (ymin > 0 or not log_scale):
                    ax_0.set_ylim(bottom=ymin, top=ymax)
                    ax_9.set_ylim(bottom=ymin, top=ymax)

        if log_scale:
            ax_0.set_yscale("log")
            ax_9.set_yscale("log")

        ax_0.set_ylabel(f"{task_name}\n{metric_label}", fontsize=10, fontweight="bold")
        ax_0.grid(True, which="both", linestyle="--", alpha=0.35)
        ax_9.grid(True, which="both", linestyle="--", alpha=0.35)

        if row_idx == 0:
            ax_0.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
            ax_9.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

        if row_idx == num_tasks - 1:
            ax_0.set_xlabel("Update Steps", fontsize=11)
            ax_9.set_xlabel("Update Steps", fontsize=11)

    fig.tight_layout()

    pdf_path = os.path.join(base_dir, pdf_filename)
    png_path = os.path.join(base_dir, pdf_filename.replace(".pdf", ".png"))
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved master PDF: {pdf_path}")
    print(f"Saved master PNG preview: {png_path}")

    # Generate summary CSV and ranking report
    if all_table_records:
        df = pd.DataFrame(all_table_records)
        csv_path = os.path.join(base_dir, "sampled_lambda_comparison_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics CSV: {csv_path}")

        print(f"\n{'='*80}")
        print("SUMMARY RANKING: AVERAGE FINAL PERFORMANCE BY ALGORITHM & LAMBDA")
        print(f"{'='*80}")
        summary_table = df.groupby(["lambda", "algo"])[["final_perf", "auc"]].mean().reset_index()
        print(summary_table.to_string(index=False))

    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Diagnostics & Cross-Algorithm Comparison for Sampled E(lambda)")
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
    generate_sampled_summary_grid(sweep_dir, all_tasks_data, win_size=args.win_size)


if __name__ == "__main__":
    main()
