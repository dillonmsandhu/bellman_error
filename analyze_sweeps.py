"""
analyze_sweeps.py
Modular analysis, extraction, and visualization tools for hyperparameter sweeps
and cross-algorithm comparisons.
"""

import os
# Force JAX to CPU to prevent GPU VRAM allocation/OOM during analysis
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import cloudpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.utils import load_run_data, load_run_data_from_path


def find_latest_run_dir(base_dir):
    """
    Finds the latest timestamp directory and environment subfolder under a tuning/results base directory.
    
    Returns:
        (timestamp, env_name, full_path) or (None, None, None) if not found.
    """
    if not os.path.exists(base_dir):
        return None, None, None
    
    # Check if base_dir itself directly contains config.json / out.pkl
    if os.path.exists(os.path.join(base_dir, "config.json")) and (
        os.path.exists(os.path.join(base_dir, "out.pkl")) or os.path.exists(os.path.join(base_dir, "best_config.json"))
    ):
        env_name = os.path.basename(base_dir)
        parent_name = os.path.basename(os.path.dirname(base_dir))
        return parent_name, env_name, base_dir

    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")])
    if not subdirs:
        return None, None, None

    # Try latest timestamp directory
    latest_sub = subdirs[-1]
    sub_path = os.path.join(base_dir, latest_sub)

    # Check if sub_path contains environment folders
    envs = [e for e in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, e)) and not e.startswith(".")]
    if envs:
        env_name = envs[0]
        return latest_sub, env_name, os.path.join(sub_path, env_name)
    
    # Alternatively sub_path might be the run directory itself
    if os.path.exists(os.path.join(sub_path, "config.json")):
        return latest_sub, "default", sub_path

def discover_algorithm_sweeps(policy="fixed", env_name="FourRooms-misc", base_results_dir="results"):
    """
    Finds the latest sweep runs for each algorithm under policy sweeps or standalone tuning dirs.
    """
    found = {}
    
    # 1. Check all sweep batches under results/{policy}/sweeps/ in reverse chronological order
    sweeps_dir = os.path.join(base_results_dir, policy, "sweeps")
    if os.path.exists(sweeps_dir):
        batches = sorted([d for d in os.listdir(sweeps_dir) if os.path.isdir(os.path.join(sweeps_dir, d)) and not d.startswith(".")], reverse=True)
        for batch in batches:
            batch_path = os.path.join(sweeps_dir, batch)
            if not os.path.isdir(batch_path):
                continue
            for algo in os.listdir(batch_path):
                if algo in found:
                    continue
                algo_dir = os.path.join(batch_path, algo, "tuning")
                if os.path.exists(algo_dir):
                    ts, env, run_path = find_latest_run_dir(algo_dir)
                    if run_path and (env_name is None or env == env_name):
                        found[algo] = run_path
                        
    # 2. Also check standalone algorithm tuning dirs if not already found
    standalone_dirs = {
        "exact_td": f"{base_results_dir}/{policy}/td_exact/tuning",
        "exact_mc": f"{base_results_dir}/{policy}/mc_exact/tuning",
        "exact_E_gd": f"{base_results_dir}/{policy}/E_gd_exact/tuning",
        "exact_E": f"{base_results_dir}/{policy}/exact_E/tuning",
        "exact_td_lambda": f"{base_results_dir}/{policy}/td_lambda_exact/tuning",
        "exact_td_symmetric": f"{base_results_dir}/{policy}/td_exact_symmetric/tuning",
        "sampled_E": f"{base_results_dir}/{policy}/sampled_E/tuning",
        "td": f"{base_results_dir}/{policy}/td/tuning",
        "mc": f"{base_results_dir}/{policy}/mc/tuning",
    }
    for algo, tuning_dir in standalone_dirs.items():
        if algo not in found and os.path.exists(tuning_dir):
            ts, env, run_path = find_latest_run_dir(tuning_dir)
            if run_path and (env_name is None or env == env_name):
                found[algo] = run_path

    return found


def load_sweep_data(path_or_base_dir, env_name=None):
    """
    Loads all sweep artifacts (config, metrics, summary_df, best_config) from a run folder or base directory.
    
    Returns:
        dict with keys: 'config', 'metrics', 'summary_df', 'best_config', 'run_dir', 'env_name', 'timestamp'
    """
    if os.path.exists(os.path.join(path_or_base_dir, "config.json")):
        run_dir = path_or_base_dir
        timestamp = os.path.basename(os.path.dirname(run_dir))
        env = os.path.basename(run_dir)
    else:
        ts, env, run_dir = find_latest_run_dir(path_or_base_dir)
        if run_dir is None:
            raise FileNotFoundError(f"No valid sweep run found in {path_or_base_dir}")
        timestamp = ts

    # Load config and out.pkl
    config, metrics = load_run_data_from_path(run_dir)

    # Load tuning_summary.csv if present
    summary_path = os.path.join(run_dir, "tuning_summary.csv")
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    # Load best_config.json if present
    best_config_path = os.path.join(run_dir, "best_config.json")
    best_config = None
    if os.path.exists(best_config_path):
        with open(best_config_path, "r") as f:
            best_config = json.load(f)

    return {
        "config": config,
        "metrics": metrics,
        "summary_df": summary_df,
        "best_config": best_config,
        "run_dir": run_dir,
        "env_name": env,
        "timestamp": timestamp,
    }


def extract_best_configuration(
    sweep_data,
    metric_key="nn_weighted_VE",
    rank_by="auc",
    rank_order="lower",
    config_idx=None,
    window_size=20,
):
    """
    Extracts the 2D seed trajectories (n_seeds, time_steps) and hyperparameter label
    for the best performing configuration in a sweep.
    
    Args:
        sweep_data: Dict loaded via load_sweep_data
        metric_key: Name of metric (e.g. "nn_weighted_VE")
        rank_by: Ranking criterion ("auc", "final_window", "final_step", "min", "max")
        rank_order: "lower" (lower is better) or "higher" (higher is better)
        config_idx: Optional explicit integer config index to force-select
        window_size: Window size in steps for "final_window"
        
    Returns:
        (seed_trajectories, best_label, best_idx, best_hparams_dict)
    """
    metrics = sweep_data["metrics"]
    summary_df = sweep_data.get("summary_df")
    best_config_meta = sweep_data.get("best_config")

    if metric_key not in metrics:
        raise KeyError(f"Metric '{metric_key}' not found in metrics. Available: {list(metrics.keys())}")

    arr = np.asarray(metrics[metric_key])

    if arr.ndim == 2:
        # Single configuration evaluated across seeds
        return arr, "Default", 0, {}
    elif arr.ndim == 3:
        # (n_combos, n_seeds, time_steps)
        n_combos, n_seeds, time_steps = arr.shape

        if config_idx is not None:
            best_idx = int(config_idx)
        else:
            rank_by_lower = rank_by.lower()
            if rank_by_lower in ["auc", "mean", "time_mean", "auc_mean"]:
                scores = arr.mean(axis=(1, 2))
            elif rank_by_lower in ["final_window", "window", "final_window_mean"]:
                win = max(1, min(time_steps, window_size))
                scores = arr[:, :, -win:].mean(axis=(1, 2))
            elif rank_by_lower in ["final", "final_step", "final_mean", "last"]:
                scores = arr[:, :, -1].mean(axis=1)
            elif rank_by_lower in ["min", "minimum"]:
                scores = arr.min(axis=-1).mean(axis=1)
            elif rank_by_lower in ["max", "maximum"]:
                scores = arr.max(axis=-1).mean(axis=1)
            else:
                scores = arr.mean(axis=(1, 2))

            is_lower = rank_order.lower() in ["lower", "min", "asc", "ascending"]
            best_idx = int(np.argmin(scores)) if is_lower else int(np.argmax(scores))

        # Identify hyperparameter columns
        best_hparams = {}
        if summary_df is not None and not summary_df.empty:
            match = summary_df[summary_df["config_idx"] == best_idx]
            if not match.empty:
                row = match.iloc[0]
                def is_metric_col(col):
                    c = col.lower()
                    if c in ["rank", "config_idx", "timestamp", "env_name", "env"]:
                        return True
                    for prefix in ["auc", "final", "final_window", "mean", "min", "max", "std"]:
                        if c.startswith(prefix + "_") or c == prefix:
                            return True
                    if c.endswith("_std") or c.endswith("_mean"):
                        return True
                    return False

                hparam_cols = [c for c in summary_df.columns if not is_metric_col(c)]
                best_hparams = {c: row[c] for c in hparam_cols}

        if not best_hparams and best_config_meta is not None and best_config_meta.get("best_config_idx") == best_idx:
            best_hparams = best_config_meta.get("best_hyperparameters", {})

        if best_hparams:
            best_label = ", ".join([f"{k}={v}" for k, v in best_hparams.items()])
        else:
            best_label = f"Config #{best_idx}"

        seed_trajectories = arr[best_idx]
        return seed_trajectories, best_label, best_idx, best_hparams
    else:
        raise ValueError(f"Unexpected shape for metric array: {arr.shape}")


def plot_algorithm_comparison(
    algorithms_dict,
    metric_key="nn_weighted_VE",
    ylabel=None,
    title=None,
    env_name="FourRooms-misc",
    log_scale=True,
    use_geom_mean=False,
    rank_by="auc",
    rank_order="lower",
    window_size=20,
    config_idx=None,
    x_axis="update_steps",
    steps_per_pi=None,
    save_path=None,
):
    """
    Plots a comparison of the best hyperparameter configurations across multiple algorithms.
    
    Args:
        algorithms_dict: Dict mapping Algorithm Name (e.g. "Exact TD") to:
                         - path to tuning directory (str) OR
                         - loaded sweep_data dict
        metric_key: Name of metric (e.g. "nn_weighted_VE")
        ylabel: Label for y-axis
        title: Plot title
        env_name: Name of environment for title
        log_scale: Whether to plot on log scale
        use_geom_mean: If True, uses geometric mean and log-std band; else standard mean ± std.
        rank_by: Metric ranking criterion ("auc", "final_window", "final_step", "min", "max")
        rank_order: "lower" (default) or "higher"
        window_size: Window size in steps for final_window ranking (default: 20)
        config_idx: Optional explicit integer config index to force-select
        x_axis: "update_steps" (default) or "env_steps".
        steps_per_pi: Optional override for environment steps per update.
        save_path: Optional file path to save plot PNG.
        
    Returns:
        fig: matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    plotted_count = 0

    for idx, (algo_name, source) in enumerate(algorithms_dict.items()):
        try:
            if isinstance(source, str):
                sweep_data = load_sweep_data(source)
            elif isinstance(source, dict) and "metrics" in source:
                sweep_data = source
            else:
                print(f"Skipping {algo_name}: invalid source type {type(source)}")
                continue

            seed_trajectories, best_label, best_idx, best_hparams = extract_best_configuration(
                sweep_data,
                metric_key=metric_key,
                rank_by=rank_by,
                rank_order=rank_order,
                config_idx=config_idx,
                window_size=window_size,
            )
        except Exception as e:
            print(f"Could not load/extract best configuration for {algo_name}: {e}")
            continue

        n_seeds, time_steps = seed_trajectories.shape
        cfg = sweep_data.get("config", {})
        env_steps_per_update = cfg.get("NUM_ENVS", 1) * cfg.get("NUM_STEPS", 1) if steps_per_pi is None else steps_per_pi

        if x_axis == "env_steps":
            x = [i * env_steps_per_update for i in range(time_steps)]
        else:
            x = list(range(time_steps))

        color = colors[idx % len(colors)]
        
        # Build clean label with hyperparams and env steps/update for sampled methods
        label_parts = []
        if best_label:
            label_parts.append(best_label)
        if env_steps_per_update > 1 and x_axis == "update_steps":
            label_parts.append(f"{env_steps_per_update} env steps/update")
        
        extra_str = f" ({', '.join(label_parts)})" if label_parts else ""
        label_with_hparam = f"{algo_name}{extra_str}"

        if use_geom_mean:
            # Geometric mean & multiplicative std band
            safe_arr = np.maximum(seed_trajectories, 1e-18)
            log_arr = np.log(safe_arr)
            log_mean = np.mean(log_arr, axis=0)
            log_std = np.std(log_arr, axis=0)
            geom_mean = np.exp(log_mean)
            lower = np.exp(log_mean - log_std)
            upper = np.exp(log_mean + log_std)

            line, = ax.plot(x, geom_mean, label=label_with_hparam, color=color, linewidth=2.2)
            if n_seeds > 1:
                ax.fill_between(x, lower, upper, color=color, alpha=0.18)
        else:
            # Arithmetic mean & std band
            mean_curve = seed_trajectories.mean(axis=0)
            std_curve = seed_trajectories.std(axis=0)

            line, = ax.plot(x, mean_curve, label=label_with_hparam, color=color, linewidth=2.2)
            if n_seeds > 1:
                ax.fill_between(x, np.maximum(mean_curve - std_curve, 1e-18), mean_curve + std_curve, color=color, alpha=0.18)

        plotted_count += 1

    if plotted_count == 0:
        print("No algorithms could be plotted.")
        plt.close(fig)
        return None

    if log_scale:
        ax.set_yscale("log")
    
    if x_axis == "env_steps":
        ax.set_xlabel("Environment Steps", fontsize=12)
    else:
        ax.set_xlabel("Update Steps", fontsize=12)

    ax.set_ylabel(ylabel or metric_key, fontsize=12)
    plot_title = title or f"Algorithm Comparison (Best Configs) - {env_name}"
    ax.set_title(plot_title, fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Comparison plot saved to {save_path}")

    return fig


def summarize_algorithm_comparison(
    algorithms_dict,
    metric_key="nn_weighted_VE",
    rank_by="auc",
    rank_order="lower",
    window_size=20,
    save_path=None,
):
    """
    Creates a unified comparison table of the best hyperparameter settings and performance
    across multiple algorithms.
    
    Returns:
        summary_df: pd.DataFrame
    """
    rows = []
    for algo_name, source in algorithms_dict.items():
        try:
            if isinstance(source, str):
                sweep_data = load_sweep_data(source)
            else:
                sweep_data = source

            seed_trajectories, best_label, best_idx, best_hparams = extract_best_configuration(
                sweep_data,
                metric_key=metric_key,
                rank_by=rank_by,
                rank_order=rank_order,
                window_size=window_size,
            )
            n_seeds, time_steps = seed_trajectories.shape
            win = max(1, min(time_steps, window_size))

            auc_val = float(seed_trajectories.mean())
            window_val = float(seed_trajectories[:, -win:].mean())
            final_vals = seed_trajectories[:, -1]
            min_vals = seed_trajectories.min(axis=-1)
            max_vals = seed_trajectories.max(axis=-1)

            rows.append({
                "Algorithm": algo_name,
                "Best Hyperparameters": best_label,
                f"AUC {metric_key}": auc_val,
                f"Final Window ({win} steps)": window_val,
                f"Final {metric_key} (Mean)": float(final_vals.mean()),
                f"Final {metric_key} (Std)": float(final_vals.std()) if len(final_vals) > 1 else 0.0,
                f"Min {metric_key}": float(min_vals.mean()),
                f"Max {metric_key}": float(max_vals.mean()),
                "Seeds": int(n_seeds),
                "Run Dir": sweep_data.get("run_dir", "N/A"),
            })
        except Exception as e:
            print(f"Error summarizing {algo_name}: {e}")

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        rank_by_lower = rank_by.lower()
        if rank_by_lower in ["auc", "mean", "time_mean", "auc_mean"]:
            sort_col = f"AUC {metric_key}"
        elif rank_by_lower in ["final_window", "window", "final_window_mean"]:
            matching_cols = [c for c in summary_df.columns if c.startswith("Final Window")]
            sort_col = matching_cols[0] if matching_cols else f"Final {metric_key} (Mean)"
        elif rank_by_lower in ["final", "final_step", "final_mean"]:
            sort_col = f"Final {metric_key} (Mean)"
        elif rank_by_lower in ["min", "minimum"]:
            sort_col = f"Min {metric_key}"
        elif rank_by_lower in ["max", "maximum"]:
            sort_col = f"Max {metric_key}"
        else:
            sort_col = f"AUC {metric_key}" if f"AUC {metric_key}" in summary_df.columns else f"Final {metric_key} (Mean)"

        is_ascending = rank_order.lower() in ["lower", "min", "asc", "ascending"]
        summary_df = summary_df.sort_values(by=sort_col, ascending=is_ascending).reset_index(drop=True)
        summary_df.insert(0, "Rank", summary_df.index + 1)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if save_path.endswith(".csv"):
            summary_df.to_csv(save_path, index=False)
        elif save_path.endswith(".json"):
            summary_df.to_json(save_path, orient="records", indent=4)
        print(f"Comparison summary saved to {save_path}")

    return summary_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze and compare hyperparameter sweeps.")
    parser.add_argument("--policy", type=str, default="fixed", choices=["fixed", "random", "ppo"], help="Policy category to search")
    parser.add_argument("--env-name", type=str, default="FourRooms-misc", help="Environment name")
    parser.add_argument("--metric", type=str, default="nn_weighted_VE", help="Primary metric key")
    parser.add_argument("--use-geom-mean", action="store_true", help="Plot geometric mean instead of arithmetic mean")
    parser.add_argument("--rank-by", type=str, default="auc", choices=["auc", "final_window", "final_step", "min", "max"],
                        help="Criterion to rank and select best config (default: auc)")
    parser.add_argument("--rank-order", type=str, default="lower", choices=["lower", "higher"],
                        help="Optimization goal: lower (default) or higher")
    parser.add_argument("--higher-is-better", action="store_true", help="Shortcut for --rank-order higher")
    parser.add_argument("--window-size", type=int, default=20, help="Window size (in steps) for final_window ranking (default: 20)")
    args = parser.parse_args()

    rank_order = "higher" if args.higher_is_better else args.rank_order

    default_algos = {
        "fixed": {
            "Exact TD": "results/fixed/td_exact/tuning",
            "Exact MC": "results/fixed/mc_exact/tuning",
            "Exact E": "results/fixed/E_gd_exact/tuning",
            "Exact TD(λ)": "results/fixed/td_lambda_exact/tuning",
        },
        "random": {
            "Exact TD": "results/random/td_exact/tuning",
            "Exact MC": "results/random/mc_exact/tuning",
            "Exact E": "results/random/exact_E/tuning",
            "Exact TD(λ)": "results/random/td_lambda_exact/tuning",
        },
        "ppo": {
            "Exact TD": "results/ppo/exact_td/tuning",
            "Exact MC": "results/ppo/exact_mc/tuning",
            "Exact E": "results/ppo/exact_E/tuning",
            "Exact TD(λ)": "results/ppo/exact_td_lambda/tuning",
        }
    }

    target_dict = default_algos.get(args.policy, default_algos["fixed"])
    print(f"\nAnalyzing latest sweeps for policy='{args.policy}', env='{args.env_name}'...")
    
    summary = summarize_algorithm_comparison(
        target_dict,
        metric_key=args.metric,
        rank_by=args.rank_by,
        rank_order=rank_order,
        window_size=args.window_size,
    )
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON SUMMARY:")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70)

    save_dir = f"results/{args.policy}/sweeps/comparison"
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "algorithm_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)

    plot_path = os.path.join(save_dir, "algorithm_comparison_best_configs.png")
    plot_algorithm_comparison(
        target_dict,
        metric_key=args.metric,
        env_name=args.env_name,
        use_geom_mean=args.use_geom_mean,
        rank_by=args.rank_by,
        rank_order=rank_order,
        window_size=args.window_size,
        save_path=plot_path,
    )
