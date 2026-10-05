"""
run_all_environments_suite.py

Orchestrates the 2D parameter sweeps over (num_steps, num_envs) across all 6 target environments:
1. FourRooms-misc
2. fourrooms-dense
3. eightrooms-misc
4. eightrooms-dense
5. whirlpool-misc
6. MountainCar-v0

Evaluates:
- Directional Cosine Similarity rho(g_bar_update, g_exact)
- Relative Error ||g_bar_update - g_exact|| / ||g_exact||
- Update Gradient Variance Var(g_bar_update) as a function of Total Batch Size N = B * T

Generates:
- Individual 3-panel heatmaps per environment
- Individual Iso-Batch scaling plots per environment
- Unified 6-environment comparative summary figure
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.variance_investigation.run_sampled_vs_exact_e import run_sweep


TARGET_ENVS = [
    "FourRooms-misc",
    "fourrooms-dense",
    "eightrooms-misc",
    "eightrooms-dense",
    "whirlpool-misc",
    "MountainCar-v0",
]


def plot_unified_cross_env_summary(all_sweep_data: dict, out_path: str):
    """
    Creates a unified 2x3 grid comparison plot across all 6 environments,
    displaying the Update Gradient Variance Var(g_bar_update) vs. Total Batch Size N = B * T.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    axes = axes.flatten()

    for idx, env_name in enumerate(TARGET_ENVS):
        ax = axes[idx]
        if env_name not in all_sweep_data:
            ax.set_title(f"{env_name} (No Data)", fontsize=11, fontweight="bold")
            continue

        data = all_sweep_data[env_name]
        steps_list = data["num_steps_list"]
        envs_list = data["num_envs_list"]
        grid_var = data["grid_var_update"]
        grid_rho = data["grid_rho"]

        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(steps_list)))
        for j, t in enumerate(steps_list):
            batch_sizes = [b * t for b in envs_list]
            vars_t = grid_var[:, j]
            ax.plot(batch_sizes, vars_t, marker="o", lw=1.8, color=colors[j], label=f"T = {t}")

        # Add 1/N reference slope
        all_batches = np.sort(np.unique([[b * t for b in envs_list] for t in steps_list]))
        valid_vars = grid_var[grid_var > 0]
        if len(valid_vars) > 0:
            median_var = np.median(valid_vars)
            mid_batch = all_batches[len(all_batches)//2]
            ref_x = np.array([all_batches[0], all_batches[-1]])
            ref_y = median_var * (mid_batch / ref_x)
            ax.plot(ref_x, ref_y, "k--", lw=1.5, alpha=0.6, label=r"$O(1/N)$ Scaling")

        mean_overall_rho = float(np.mean(grid_rho[grid_rho > 0])) if np.any(grid_rho > 0) else 0.0
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total Batch Size $N = B \\times T$", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"$\widehat{\mathrm{Var}}(\bar{g}_{\mathrm{update}})$", fontsize=10, fontweight="bold")
        ax.set_title(f"{env_name} (Mean $\\rho = {mean_overall_rho:.2f}$)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Update Gradient Variance Scaling vs. Total Batch Size $N = B \\times T$\n"
        r"(Averaged Minibatch Gradients $\bar{g}_{\mathrm{update}}$ across All Tabular Envs and Mountain Car)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved unified cross-environment summary plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Sampled vs Exact E Suite Across All Environments")
    parser.add_argument("--num_updates", type=int, default=30, help="Updates per sweep setting")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--out_dir", type=str, default="results/sampled_vs_exact_e_suite", help="Output directory")
    parser.add_argument("--envs", nargs="+", default=TARGET_ENVS, help="Subset of environments to run")
    args = parser.parse_args()

    suite_out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(suite_out_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING SAMPLED VS. EXACT E FULL SUITE")
    print(f"Environments: {args.envs}")
    print(f"Updates/Setting: {args.num_updates} | Seed: {args.seed}")
    print(f"Output Directory: {suite_out_dir}")
    print("=" * 80)

    all_sweep_data = {}
    suite_start = time.time()

    for env_name in args.envs:
        print(f"\n[{env_name}] Starting 2D sweep...")
        env_out_dir = os.path.join(suite_out_dir, env_name)
        os.makedirs(env_out_dir, exist_ok=True)

        if env_name == "MountainCar-v0":
            steps_list = [16, 32, 64, 100]
        else:
            steps_list = [16, 32, 64, 128]
        envs_list = [4, 8, 16, 32, 64]

        env_start = time.time()
        sweep_res = run_sweep(
            env_name=env_name,
            num_steps_list=steps_list,
            num_envs_list=envs_list,
            num_updates=args.num_updates,
            seed=args.seed,
            out_dir=env_out_dir,
        )
        all_sweep_data[env_name] = sweep_res
        env_elapsed = time.time() - env_start
        print(f"[{env_name}] Sweep finished in {env_elapsed:.1f}s")

    # Save complete suite dictionary
    np.save(os.path.join(suite_out_dir, "all_sweep_data.npy"), all_sweep_data)

    # Plot unified 6-environment comparison figure
    summary_plot_path = os.path.join(suite_out_dir, "all_environments_variance_summary.png")
    plot_unified_cross_env_summary(all_sweep_data, summary_plot_path)

    total_elapsed = time.time() - suite_start
    print("=" * 80)
    print(f"ALL ENVIRONMENTS COMPLETED IN {total_elapsed:.1f}s ({total_elapsed/60:.2f} mins)")
    print(f"Results stored at: {suite_out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
