"""
sampled_best_lambda_diagnostics.py
Consolidates lambda sweep results across all 12 tasks for the 4 sampled E variants.
Automatically tunes over all swept lambdas to find the best-performing lambda for each algorithm,
then generates:
  1. Policy Eval Tasks (random & fixed):
      * Value Learning Curves (nn_weighted_VE, log scale) for the best lambda
      * Greedy Accuracy Curves (nn_greedy_correct, [0, 1] scale) for the best lambda
  2. PPO Tasks (ppo):
      * Performance Curves (V_start) for the best lambda
  3. Master summary dashboard (PDF & PNG) with winning lambdas displayed in legends
  4. Individual high-res task comparison plots
  5. Quantitative metrics CSV explicitly reporting best lambda and performance metrics
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
        "name": "E(λ) Fixed",
        "color": "#e74c3c",      # Crimson Red
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "E_lambda_geometric": {
        "name": "E(λ) Geometric",
        "color": "#2980b9",      # Blue
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    "E_lambda_differentiable": {
        "name": "E(λ) Differentiable",
        "color": "#27ae60",      # Green
        "linestyle": ":",
        "linewidth": 2.0,
    },
}


def _parse_algo_and_lambda(dir_name):
    """
    Parses directory name into (algo_base, lambda_val).
    e.g. 'E_lambda_fixed_0.9' -> ('E_lambda_fixed', 0.9)
         'sampled_E'          -> ('sampled_E', 0.0)
         'sampled_E_0.0'      -> ('sampled_E', 0.0)
    """
    for algo_base in ["E_lambda_differentiable", "E_lambda_geometric", "E_lambda_fixed", "sampled_E"]:
        if dir_name.startswith(algo_base):
            suffix = dir_name[len(algo_base):].lstrip("_")
            try:
                lmbda = float(suffix) if suffix else 0.0
            except ValueError:
                lmbda = 0.0
            return algo_base, lmbda
    return dir_name, 0.0


def _resolve_metric_key(metrics_dict, requested_metric):
    """Finds exact or fallback metric key in run metrics."""
    if requested_metric in metrics_dict:
        return requested_metric

    lower_map = {k.lower(): k for k in metrics_dict.keys()}
    if requested_metric.lower() in lower_map:
        return lower_map[requested_metric.lower()]

    fallbacks = {
        "nn_weighted_VE": ["Weighted Value Errors", "weighted_VE", "E", "value_loss"],
        "nn_greedy_correct": ["Value Learning Greedy Accuracy", "greedy_correct", "greedy_acc"],
        "V_start": ["returned_discounted_episode_returns", "returned_episode_returns", "mean_rew"],
    }
    for fb in fallbacks.get(requested_metric, []):
        if fb in metrics_dict:
            return fb
        if fb.lower() in lower_map:
            return lower_map[fb.lower()]

    return None


def select_best_lambdas(all_tasks_data, win_size=40):
    """
    For each task and algorithm, evaluates all available lambdas and selects the best lambda.
    Returns:
        best_data: { task_name: { algo_base: { 'best_lambda': float, 'best_run_data': ..., 'score': float } } }
        all_eval_records: list of dicts with quantitative metrics across all lambdas
    """
    best_data = {}
    all_eval_records = []

    for task_name, t_info in all_tasks_data.items():
        env_name = t_info["env_name"]
        pol = t_info["policy_type"].lower()
        runs = t_info["runs"]

        # Selection criterion metric
        if pol in ["random", "fixed"]:
            primary_metric = "nn_weighted_VE"
            rank_order = "lower"
        else:  # ppo
            primary_metric = "V_start"
            rank_order = "higher"

        # Group runs by algo_base
        algo_groups = {}
        for run_name, s_data in runs.items():
            algo_base, lmbda = _parse_algo_and_lambda(run_name)
            if algo_base not in algo_groups:
                algo_groups[algo_base] = []
            algo_groups[algo_base].append((lmbda, run_name, s_data))

        task_best = {}

        for algo_base, lmbda_runs in algo_groups.items():
            best_candidate = None
            best_score = float("inf") if rank_order == "lower" else float("-inf")

            for lmbda, run_name, s_data in lmbda_runs:
                actual_metric = _resolve_metric_key(s_data["metrics"], primary_metric)
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
                    mean_traj = trajs.mean(axis=0)
                    final_score = float(mean_traj[-max(1, win_size):].mean())
                    auc_score = float(np.mean(mean_traj))

                    record = {
                        "task": task_name,
                        "env": env_name,
                        "policy": pol,
                        "algo": algo_base,
                        "lambda": lmbda,
                        "metric": actual_metric,
                        "final_perf": final_score,
                        "auc": auc_score,
                    }
                    all_eval_records.append(record)

                    # Check if this lambda is the best so far
                    is_better = (final_score < best_score) if rank_order == "lower" else (final_score > best_score)
                    if is_better or best_candidate is None:
                        best_score = final_score
                        best_candidate = {
                            "algo_base": algo_base,
                            "best_lambda": lmbda,
                            "run_name": run_name,
                            "s_data": s_data,
                            "score": final_score,
                            "auc": auc_score,
                            "record": record,
                        }

                except Exception as e:
                    print(f"Error evaluating {algo_base} (λ={lmbda}) on {task_name}: {e}")
                    continue

            if best_candidate:
                task_best[algo_base] = best_candidate
                best_candidate["record"]["is_best_lambda"] = True

        best_data[task_name] = task_best

    return best_data, all_eval_records


def generate_best_lambda_plots(sweep_dir, all_tasks_data, best_data, win_size=40, plots_dir=None):
    """
    Generates standalone comparison plots per task showing all 4 algorithms at their best lambda.
    """
    if plots_dir is None:
        plots_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary_records = []

    for task_name, t_info in all_tasks_data.items():
        env_name = t_info["env_name"]
        pol = t_info["policy_type"].lower()
        task_bests = best_data.get(task_name, {})

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
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            has_data = False

            for algo_base, style in ALGO_STYLES.items():
                if algo_base not in task_bests:
                    continue

                best_info = task_bests[algo_base]
                s_data = best_info["s_data"]
                best_lmbda = best_info["best_lambda"]
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
                    lbl = f"{style['name']} (best λ={best_lmbda})"

                    ax.plot(x, mean, label=lbl, color=c, linestyle=ls, linewidth=lw)
                    if n_seeds > 1:
                        if log_scale:
                            lower = np.maximum(mean - std, mean * 0.05)
                        else:
                            lower = mean - std
                        upper = mean + std
                        ax.fill_between(x, lower, upper, color=c, alpha=0.15)

                    has_data = True

                    if metric_key in ["nn_weighted_VE", "V_start"]:
                        final_val = float(mean[-max(1, win_size):].mean())
                        auc_val = float(np.mean(mean))
                        summary_records.append({
                            "task": task_name,
                            "env": env_name,
                            "policy": pol,
                            "algo": algo_base,
                            "best_lambda": best_lmbda,
                            "metric": actual_metric,
                            "final_window_mean": final_val,
                            "auc": auc_val,
                        })

                except Exception as e:
                    print(f"Error plotting best {algo_base} for {task_name}: {e}")
                    continue

            if has_data:
                ax.set_title(f"{env_name} ({pol.upper()}) - {label}\n[Best Performing λ per Algorithm]", fontsize=12, fontweight="bold")
                ax.set_xlabel("Updates", fontsize=11)
                ax.set_ylabel(label, fontsize=11)
                if log_scale:
                    ax.set_yscale("log")
                if "greedy" in metric_key.lower():
                    ax.set_ylim(-0.02, 1.05)
                ax.grid(True, which="both", linestyle="--", alpha=0.35)
                ax.legend(loc="best", fontsize=9, framealpha=0.9)
                fig.tight_layout()

                clean_metric = metric_key.replace("nn_", "").replace("returned_", "")
                out_path = os.path.join(plots_dir, f"{env_name}_{pol}_{clean_metric}_best_lambda.png")
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.close(fig)

    return summary_records


def generate_master_summary_grid(sweep_dir, all_tasks_data, best_data, win_size=40):
    """
    Creates master 5-column dashboard across all environments displaying algorithms at best lambda.
    """
    envs = ["EightRooms", "FourRooms-misc", "Whirlpool", "MountainCar-v0"]
    available_envs = [e for e in envs if any(t.startswith(e) for t in all_tasks_data.keys())]
    if not available_envs:
        available_envs = sorted(list(set(t["env_name"] for t in all_tasks_data.values())))

    num_rows = len(available_envs)
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=5,
        figsize=(23, 3.6 * num_rows),
        squeeze=False,
    )

    column_defs = [
        ("random", "nn_weighted_VE", "Value Error", "lower", True),
        ("random", "nn_greedy_correct", "Greedy Acc.", "higher", False),
        ("fixed", "nn_weighted_VE", "Value Error", "lower", True),
        ("fixed", "nn_greedy_correct", "Greedy Acc.", "higher", False),
        ("ppo", "V_start", "PPO Return (V_start)", "higher", False),
    ]

    for col_idx, (pol, metric_key, col_title, _, _) in enumerate(column_defs):
        axes[0, col_idx].set_title(f"{pol.upper()}: {col_title} [Best λ]", fontsize=11, fontweight="bold", pad=10)

    for row_idx, env_name in enumerate(available_envs):
        axes[row_idx, 0].set_ylabel(f"{env_name}", fontsize=12, fontweight="bold")

        for col_idx, (pol, metric_key, col_title, rank_order, log_scale) in enumerate(column_defs):
            ax = axes[row_idx, col_idx]
            task_name = f"{env_name}_{pol}"
            task_bests = best_data.get(task_name, {})

            if not task_bests:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, color="gray")
                continue

            plotted = False
            for algo_base, style in ALGO_STYLES.items():
                if algo_base not in task_bests:
                    continue

                best_info = task_bests[algo_base]
                s_data = best_info["s_data"]
                best_lmbda = best_info["best_lambda"]
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
                    lbl = f"{style['name']} (λ={best_lmbda})"

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

    pdf_path = os.path.join(sweep_dir, "sampled_best_lambda_master_summary.pdf")
    png_path = os.path.join(sweep_dir, "sampled_best_lambda_master_summary.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved master PDF: {pdf_path}")
    print(f"Saved master PNG preview: {png_path}")
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for Sampled E Best Lambda Tuning Comparison")
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

    best_data, all_eval_records = select_best_lambdas(all_tasks_data, win_size=args.win_size)

    plots_dir = os.path.join(sweep_dir, "plots")
    best_summary_records = generate_best_lambda_plots(sweep_dir, all_tasks_data, best_data, win_size=args.win_size, plots_dir=plots_dir)
    generate_master_summary_grid(sweep_dir, all_tasks_data, best_data, win_size=args.win_size)

    # 1. Save all evaluation trials CSV (all lambdas tested)
    if all_eval_records:
        df_all = pd.DataFrame(all_eval_records)
        all_csv_path = os.path.join(sweep_dir, "all_lambdas_metrics.csv")
        df_all.to_csv(all_csv_path, index=False)
        print(f"Saved full trials CSV: {all_csv_path}")

    # 2. Save best lambda CSV (clearly indicating best lambda and scores)
    if best_summary_records:
        df_best = pd.DataFrame(best_summary_records)
        best_csv_path = os.path.join(sweep_dir, "best_lambda_comparison_metrics.csv")
        df_best.to_csv(best_csv_path, index=False)
        print(f"Saved best lambda metrics CSV: {best_csv_path}")

        print("\n" + "=" * 80)
        print("SUMMARY: BEST LAMBDA IDENTIFIED PER TASK & ALGORITHM")
        print("=" * 80)
        print(df_best[["env", "policy", "algo", "best_lambda", "final_window_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
