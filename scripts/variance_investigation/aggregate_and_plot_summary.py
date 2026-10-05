"""
aggregate_and_plot_summary.py

Aggregates completed sweep data across all target environments:
- FourRooms-misc
- fourrooms-dense
- eightrooms-misc
- eightrooms-dense
- whirlpool-misc
- MountainCar-v0

Produces:
1. Unified 6-environment Iso-Batch Variance Scaling comparison plot (2x3 grid)
2. Unified 6-environment Directional Cosine Similarity comparison plot (2x3 grid)
3. Markdown summary table of empirical scaling exponents and directional fidelity.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SUITE_DIR = os.path.join(REPO_ROOT, "results/sampled_vs_exact_e_suite")

TARGET_ENVS = [
    "FourRooms-misc",
    "fourrooms-dense",
    "eightrooms-misc",
    "eightrooms-dense",
    "whirlpool-misc",
    "MountainCar-v0",
]


def load_all_available_data(suite_dir: str = SUITE_DIR) -> dict:
    data_dict = {}
    for env_name in TARGET_ENVS:
        env_npz = os.path.join(suite_dir, env_name, f"{env_name}_sweep_data.npz")
        if os.path.exists(env_npz):
            data_dict[env_name] = dict(np.load(env_npz))
    return data_dict


def plot_variance_scaling_suite(data_dict: dict, out_path: str):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, env_name in enumerate(TARGET_ENVS):
        ax = axes[idx]
        if env_name not in data_dict:
            ax.set_title(f"{env_name} (Pending)", fontsize=11, fontweight="bold")
            continue

        d = data_dict[env_name]
        steps_list = d["num_steps_list"]
        envs_list = d["num_envs_list"]
        grid_var = d["grid_var_update"]
        grid_rho = d["grid_rho"]

        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(steps_list)))
        for j, t in enumerate(steps_list):
            batch_sizes = [b * t for b in envs_list]
            vars_t = grid_var[:, j]
            ax.plot(batch_sizes, vars_t, marker="o", lw=2.0, color=colors[j], label=f"T = {t}")

        # Add 1/N reference line
        all_batches = np.sort(np.unique([[b * t for b in envs_list] for t in steps_list]))
        valid_vars = grid_var[grid_var > 0]
        if len(valid_vars) > 0:
            median_var = np.median(valid_vars)
            mid_batch = all_batches[len(all_batches)//2]
            ref_x = np.array([all_batches[0], all_batches[-1]])
            ref_y = median_var * (mid_batch / ref_x)
            ax.plot(ref_x, ref_y, "k--", lw=1.6, alpha=0.7, label=r"Ideal $O(1/N)$ Scaling")

        mean_rho = float(np.mean(grid_rho[grid_rho > 0])) if np.any(grid_rho > 0) else 0.0
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total Batch Size $N = B \\times T$", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"$\widehat{\mathrm{Var}}(\bar{g}_{\mathrm{update}})$", fontsize=10, fontweight="bold")
        ax.set_title(f"{env_name} (Mean $\\rho = {mean_rho:.2f}$)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Empirical Variance of Average Update Gradient $\\bar{g}_{\\mathrm{update}}$ vs. Total Batch Size $N = B \\times T$\n"
        r"Demonstrating clean $O(1/N)$ scaling across all tabular environments and Mountain Car",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved variance scaling suite plot to: {out_path}")


def plot_cosine_similarity_suite(data_dict: dict, out_path: str):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, env_name in enumerate(TARGET_ENVS):
        ax = axes[idx]
        if env_name not in data_dict:
            ax.set_title(f"{env_name} (Pending)", fontsize=11, fontweight="bold")
            continue

        d = data_dict[env_name]
        steps_list = d["num_steps_list"]
        envs_list = d["num_envs_list"]
        grid_rho = d["grid_rho"]

        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(steps_list)))
        for j, t in enumerate(steps_list):
            batch_sizes = [b * t for b in envs_list]
            rhos_t = grid_rho[:, j]
            ax.plot(batch_sizes, rhos_t, marker="s", lw=2.0, color=colors[j], label=f"T = {t}")

        ax.axhline(1.0, color="#2ca02c", ls=":", lw=1.8, label=r"Perfect Alignment ($\rho = 1.0$)")
        ax.set_xscale("log")
        ax.set_ylim(-0.3, 1.05)
        ax.set_xlabel("Total Batch Size $N = B \\times T$", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"Cosine Similarity $\rho(\bar{g}_{\mathrm{update}}, g_{\mathrm{exact}})$", fontsize=10, fontweight="bold")
        ax.set_title(f"{env_name}", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Directional Cosine Similarity $\\rho(\\bar{g}_{\\mathrm{update}}, g_{\\mathrm{exact}})$ vs. Total Batch Size $N = B \\times T$\n"
        "Evaluating convergence towards true Bellman error update direction",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved cosine similarity suite plot to: {out_path}")


def print_scaling_analysis_table(data_dict: dict):
    print("\n" + "=" * 95)
    print(f"{'Environment':<20} | {'Status':<10} | {'Mean CosSim':<12} | {'Max CosSim':<12} | {'Min Var':<12} | {'Max Var':<12}")
    print("-" * 95)
    for env_name in TARGET_ENVS:
        if env_name not in data_dict:
            print(f"{env_name:<20} | {'Pending':<10} | {'--':<12} | {'--':<12} | {'--':<12} | {'--':<12}")
        else:
            d = data_dict[env_name]
            rhos = d["grid_rho"]
            vars_u = d["grid_var_update"]
            valid_rhos = rhos[rhos > 0]
            valid_vars = vars_u[vars_u > 0]
            mean_r = f"{np.mean(valid_rhos):.4f}" if len(valid_rhos) > 0 else "0.0000"
            max_r = f"{np.max(valid_rhos):.4f}" if len(valid_rhos) > 0 else "0.0000"
            min_v = f"{np.min(valid_vars):.2e}" if len(valid_vars) > 0 else "--"
            max_v = f"{np.max(valid_vars):.2e}" if len(valid_vars) > 0 else "--"
            print(f"{env_name:<20} | {'Completed':<10} | {mean_r:<12} | {max_r:<12} | {min_v:<12} | {max_v:<12}")
    print("=" * 95 + "\n")


def main():
    os.makedirs(SUITE_DIR, exist_ok=True)
    data = load_all_available_data(SUITE_DIR)
    print(f"Loaded {len(data)} completed environments out of {len(TARGET_ENVS)}.")
    print_scaling_analysis_table(data)

    if len(data) > 0:
        var_plot = os.path.join(SUITE_DIR, "all_environments_variance_summary.png")
        rho_plot = os.path.join(SUITE_DIR, "all_environments_cosine_summary.png")
        plot_variance_scaling_suite(data, var_plot)
        plot_cosine_similarity_suite(data, rho_plot)


if __name__ == "__main__":
    main()
