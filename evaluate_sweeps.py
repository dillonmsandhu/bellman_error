"""
evaluate_sweeps.py
Clean, modular script to load latest sweep runs, plot seed trajectories for the best configurations,
and generate cross-algorithm comparisons.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from analyze_sweeps import (
    load_sweep_data,
    extract_best_configuration,
    plot_algorithm_comparison,
    summarize_algorithm_comparison,
    discover_algorithm_sweeps,
)


def plot_all_latest_sweeps(metric_key="nn_weighted_VE", policy="fixed", env_name="FourRooms-misc", use_geom_mean=False):
    """
    Finds the latest run for each algorithm, plots individual seed trajectories for the best config,
    and produces a cross-algorithm comparison plot.
    """
    discovered = discover_algorithm_sweeps(policy=policy, env_name=env_name)
    if not discovered:
        print(f"No sweep runs found for policy='{policy}', env='{env_name}'.")
        return

    valid_runs = {}

    for algo_name, run_path in discovered.items():
        try:
            sweep_data = load_sweep_data(run_path)
            valid_runs[algo_name] = sweep_data
            run_dir = sweep_data["run_dir"]
            env = sweep_data["env_name"]
            print(f"Loaded {algo_name} from {run_dir} (env: {env})")
        except Exception as e:
            print(f"Could not load {algo_name} from {run_path}: {e}")
            continue

        # Plot seed trajectories for the winning configuration
        try:
            seed_trajectories, best_label, best_idx, best_hparams = extract_best_configuration(
                sweep_data, metric_key=metric_key
            )
            cfg = sweep_data.get("config", {})
            steps_per_pi = cfg.get("NUM_ENVS", 1) * cfg.get("NUM_STEPS", 1)
            n_seeds, time_steps = seed_trajectories.shape
            x = list(range(time_steps))

            fig, ax = plt.subplots(figsize=(10, 6))
            for seed_idx in range(n_seeds):
                ax.plot(
                    x, seed_trajectories[seed_idx],
                    label=f"Seed {seed_idx} (final={seed_trajectories[seed_idx, -1]:.2e})",
                    linewidth=1.5,
                    alpha=0.75
                )

            mean_y = seed_trajectories.mean(axis=0)
            ax.plot(x, mean_y, label=f"Mean across {n_seeds} seeds", color="black", linewidth=2.2, linestyle="--")

            ax.set_yscale("log")
            xlabel_str = f"Update Steps ({steps_per_pi} env steps/update)" if steps_per_pi > 1 else "Update Steps"
            ax.set_xlabel(xlabel_str, fontsize=12)
            ax.set_ylabel(metric_key, fontsize=12)
            ax.set_title(f"{algo_name} ({env}) - Best Config: {best_label}", fontsize=13, fontweight="bold")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize=9, frameon=True)

            plot_path = os.path.join(run_dir, f"{metric_key}_best_config_seeds.png")
            fig.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  -> Saved seed trajectory plot to {plot_path}")

        except Exception as e:
            print(f"  -> Error plotting seeds for {algo_name}: {e}")

    if not valid_runs:
        print("No valid sweep runs found.")
        return

    # Generate Cross-Algorithm Comparison
    print("\n" + "=" * 70)
    print("CROSS-ALGORITHM COMPARISON")
    print("=" * 70)
    summary_df = summarize_algorithm_comparison(valid_runs, metric_key=metric_key)
    print(summary_df.to_string(index=False))

    save_dir = f"results/{policy}/sweeps/comparison"
    os.makedirs(save_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    plot_path = os.path.join(save_dir, f"algorithm_comparison_{metric_key}.png")
    plot_algorithm_comparison(
        valid_runs,
        metric_key=metric_key,
        env_name=env_name,
        log_scale=True,
        use_geom_mean=use_geom_mean,
        save_path=plot_path,
        title=f"{policy.capitalize()} Policy Evaluation ({env_name}) - Algorithm Comparison",
    )
    print(f"\nSummary & Comparison plot saved in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latest tuning sweeps and comparisons")
    parser.add_argument("--policy", type=str, default="fixed", choices=["fixed", "random", "ppo"],
                        help="Filter by policy type (fixed, random, ppo)")
    parser.add_argument("--env-name", type=str, default="FourRooms-misc", help="Environment name")
    parser.add_argument("--metric", type=str, default="nn_weighted_VE", help="Metric to plot")
    parser.add_argument("--use-geom-mean", action="store_true", help="Use geometric mean for error bands")
    args = parser.parse_args()

    plot_all_latest_sweeps(metric_key=args.metric, policy=args.policy, env_name=args.env_name, use_geom_mean=args.use_geom_mean)
