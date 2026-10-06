"""
scripts/sweeps/plot_consolidated.py

Consolidates all completed environment results (.npz) from results/sweeps/{sweep_id}/
and produces comprehensive cross-environment comparison figures:

1. cross_env_sampled_vs_exact_E_trajectories (multi-panel trajectories across T)
2. cross_env_avg_cosine_vs_T (average directional alignment vs horizon T)
3. cross_env_exact_E_vs_TD_all_tasks (theoretical rho(g_exact_E, g_exact_TD) for each task)
4. cross_env_three_way_alignment (rho(E, TD), rho(TD, MC), rho(E, MC) showing E as dot product)
5. cross_env_squared_error_trajectories (squared error ||g_samp - g_exact||^2 on log scale)
6. cross_env_avg_squared_error_vs_T (average squared error vs horizon T)
7. cross_env_relative_spectral_norm (norm(K) / norm(S) vs final alignment across tasks)
8. cross_env_bias_variance_summary (bias^2 vs sample variance across horizons)
"""

import os
import sys
import glob
import argparse
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Plot consolidated cross-environment sweep metrics")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to plot")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing .npz files")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save consolidated plots")
    return parser.parse_args()


def load_all_env_results(search_dir: str) -> Dict[str, Dict[str, Any]]:
    npz_files = sorted(glob.glob(os.path.join(search_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz result files found in {search_dir}")

    all_data = {}
    for fpath in npz_files:
        fname = os.path.basename(fpath)
        env_name = fname.replace(".npz", "")
        data = np.load(fpath)
        horizons = list(data["horizons"])

        env_dict = {"horizons": horizons, "total_batch_size": int(data["total_batch_size"])}
        for t in horizons:
            t_data = {}
            prefix = f"T_{t}_"
            for k in data.files:
                if k.startswith(prefix):
                    metric_name = k[len(prefix):]
                    t_data[metric_name] = data[k]
            env_dict[t] = t_data
        all_data[env_name] = env_dict

    return all_data


def save_fig(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, name)}.png / .pdf")


def plot_all(all_data: Dict[str, Dict[str, Any]], out_dir: str):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    env_names = sorted(all_data.keys())
    n_envs = len(env_names)

    if n_envs == 6:
        rows, cols = 2, 3
    elif n_envs <= 3:
        rows, cols = 1, n_envs
    elif n_envs == 4:
        rows, cols = 2, 2
    else:
        cols = min(4, n_envs)
        rows = (n_envs + cols - 1) // cols

    # Common horizon colors
    first_env = env_names[0]
    sample_horizons = sorted(all_data[first_env]["horizons"])
    h_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(sample_horizons)))

    # =========================================================================
    # 0. Policy Return Trajectories Across Updates for All Horizons T (MAIN FIGURE)
    # =========================================================================
    fig0, axes0 = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes0[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])

        for h_idx, t in enumerate(horizons):
            res = env_data[t]
            u = res["updates"]
            ret_m = res["policy_return_mean"]
            ret_s = res.get("policy_return_std", np.zeros_like(ret_m)) / np.sqrt(4)
            color = h_colors[h_idx % len(h_colors)]
            ax.plot(u, ret_m, color=color, lw=2.0, label=f"T={t}")
            ax.fill_between(u, ret_m - 1.96 * ret_s, ret_m + 1.96 * ret_s, color=color, alpha=0.15)

        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        if c == 0:
            ax.set_ylabel(r"Policy Return $V^\pi(s_0)$", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right", frameon=True)
        ax.grid(True, alpha=0.3)

    for idx in range(n_envs, rows * cols):
        axes0[idx // cols][idx % cols].axis("off")

    fig0.suptitle("Policy Return Learning Curves Across Rollout Horizons T (Fixed Batch N = 16384)", fontsize=14, fontweight="bold")
    fig0.tight_layout()
    save_fig(fig0, out_dir, "cross_env_policy_returns_all_T")

    # =========================================================================
    # 0B. Final / Asymptotic Policy Return vs Horizon T
    # =========================================================================
    fig0b, axes0b = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes0b[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])

        final_returns = [float(np.mean(env_data[t]["policy_return_mean"][-5:])) for t in horizons]
        ax.plot(horizons, final_returns, marker="o", lw=2.4, color="#2ecc71", markersize=7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(horizons)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon T", fontsize=9)
        if c == 0:
            ax.set_ylabel(r"Final Return $V^\pi(s_0)$", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    for idx in range(n_envs, rows * cols):
        axes0b[idx // cols][idx % cols].axis("off")

    fig0b.suptitle("Asymptotic Policy Performance vs. Rollout Horizon T (Fixed Batch N = 16384)", fontsize=14, fontweight="bold")
    fig0b.tight_layout()
    save_fig(fig0b, out_dir, "cross_env_final_return_vs_T")

    # =========================================================================
    # 1. Sampled vs Exact E Trajectories
    # =========================================================================
    fig1, axes1 = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False, sharey=True)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes1[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])

        for h_idx, t in enumerate(horizons):
            res = env_data[t]
            u = res["updates"]
            ax.plot(u, res["rho_SE_EE_mean"], color=h_colors[h_idx % len(h_colors)], lw=1.8, label=f"T={t}")

        ax.axhline(1.0, color="#27ae60", ls=":", lw=1.2)
        ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Update Step", fontsize=9)
        if c == 0:
            ax.set_ylabel(r"$\rho(g_{\mathrm{samp\_E}}, g_{\mathrm{exact\_E}})$", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right", frameon=True)

    for idx in range(n_envs, rows * cols):
        axes1[idx // cols][idx % cols].axis("off")

    fig1.suptitle("Sampled E vs. Exact Expected Bellman Error Across Optimization", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    save_fig(fig1, out_dir, "cross_env_sampled_vs_exact_E_trajectories")

    # =========================================================================
    # 2. Average Cosine Similarity vs Horizon T
    # =========================================================================
    fig2, axes2 = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False, sharey=True)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes2[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])

        avg_rho_EE = [float(np.mean(env_data[t]["rho_SE_EE_mean"])) for t in horizons]
        avg_rho_ETD = [float(np.mean(env_data[t]["rho_SE_ETD_mean"])) for t in horizons]

        ax.plot(horizons, avg_rho_EE, marker="o", lw=2.2, color="#d63031", label=r"$\rho(\bar{g}_E, g_E^*)$")
        ax.plot(horizons, avg_rho_ETD, marker="s", lw=2.2, color="#0984e3", ls="--", label=r"$\rho(\bar{g}_E, g_{\mathrm{TD}}^*)$")
        ax.axhline(1.0, color="#27ae60", ls=":", lw=1.2)
        ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(horizons)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Horizon T", fontsize=9)
        if c == 0:
            ax.set_ylabel(r"Average Cosine $\bar{\rho}$", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right", frameon=True)

    for idx in range(n_envs, rows * cols):
        axes2[idx // cols][idx % cols].axis("off")

    fig2.suptitle("Directional Rotation vs. Horizon T (Fixed Batch N = 16384)", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    save_fig(fig2, out_dir, "cross_env_avg_cosine_vs_T")

    # =========================================================================
    # 3. Exact E vs TD for Each Task
    # =========================================================================
    fig3, axes3 = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False, sharey=True)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes3[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])
        rep_t = horizons[len(horizons) // 2]
        u = env_data[rep_t]["updates"]

        ax.plot(u, env_data[rep_t]["rho_exactE_exactTD_mean"], color="#0984e3", lw=2.2, label=r"$\rho(g_E^*, g_{\mathrm{TD}}^*)$")
        ax.axhline(1.0, color="#27ae60", ls=":", lw=1.2)
        ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Update Step", fontsize=9)
        if c == 0:
            ax.set_ylabel(r"$\rho(g_E^*, g_{\mathrm{TD}}^*)$", fontsize=10)

    for idx in range(n_envs, rows * cols):
        axes3[idx // cols][idx % cols].axis("off")

    fig3.suptitle("Theoretical Alignment Between Expected Bellman Error & TD Semi-Gradient", fontsize=14, fontweight="bold")
    fig3.tight_layout()
    save_fig(fig3, out_dir, "cross_env_exact_E_vs_TD_all_tasks")

    # =========================================================================
    # 4. Three-Way Alignment: E, TD, and MC (Demonstrating E = dot product)
    # =========================================================================
    fig4, axes4 = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False, sharey=True)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes4[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])
        rep_t = horizons[len(horizons) // 2]
        u = env_data[rep_t]["updates"]

        ax.plot(u, env_data[rep_t]["rho_exactE_exactTD_mean"], color="#0984e3", lw=2.0, label=r"$\rho(E^*, \mathrm{TD}^*)$")
        ax.plot(u, env_data[rep_t]["rho_exactE_exactMC_mean"], color="#6c5ce7", lw=2.0, label=r"$\rho(E^*, \mathrm{MC}^*)$")
        ax.plot(u, env_data[rep_t]["rho_exactTD_exactMC_mean"], color="#00b894", lw=2.0, ls="--", label=r"$\rho(\mathrm{TD}^*, \mathrm{MC}^*)$")
        ax.axhline(1.0, color="#27ae60", ls=":", lw=1.2)
        ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Update Step", fontsize=9)
        if c == 0:
            ax.set_ylabel("Cosine Similarity", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right", frameon=True)

    for idx in range(n_envs, rows * cols):
        axes4[idx // cols][idx % cols].axis("off")

    fig4.suptitle("Three-Way Exact Alignment: Bellman Error Symmetrization vs. TD & MC", fontsize=14, fontweight="bold")
    fig4.tight_layout()
    save_fig(fig4, out_dir, "cross_env_three_way_alignment")

    # =========================================================================
    # 5. Gradient Squared Error Trajectories
    # =========================================================================
    fig5, axes5 = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes5[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])

        for h_idx, t in enumerate(horizons):
            res = env_data[t]
            u = res["updates"]
            ax.plot(u, res["sq_err_mean"], color=h_colors[h_idx % len(h_colors)], lw=1.8, label=f"T={t}")

        ax.set_yscale("log")
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        if c == 0:
            ax.set_ylabel(r"$\|g_{\mathrm{samp\_E}} - g_E^*\|^2$", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right", frameon=True)

    for idx in range(n_envs, rows * cols):
        axes5[idx // cols][idx % cols].axis("off")

    fig5.suptitle("Gradient Squared Error Across Learning (Log Scale)", fontsize=14, fontweight="bold")
    fig5.tight_layout()
    save_fig(fig5, out_dir, "cross_env_squared_error_trajectories")

    # =========================================================================
    # 6. Average Squared Error vs Horizon T
    # =========================================================================
    fig6, axes6 = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes6[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])

        avg_sq_err = [float(np.mean(env_data[t]["sq_err_mean"])) for t in horizons]
        ax.plot(horizons, avg_sq_err, marker="d", lw=2.2, color="#6c5ce7", label=r"Mean $\|g_{\mathrm{samp\_E}} - g_E^*\|^2$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(horizons)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon T", fontsize=9)
        if c == 0:
            ax.set_ylabel("Mean Squared Error", fontsize=10)

    for idx in range(n_envs, rows * cols):
        axes6[idx // cols][idx % cols].axis("off")

    fig6.suptitle("Truncation Squared Error vs. Rollout Horizon T (Log Scale)", fontsize=14, fontweight="bold")
    fig6.tight_layout()
    save_fig(fig6, out_dir, "cross_env_avg_squared_error_vs_T")

    # =========================================================================
    # 7. Relative Spectral Norm ||K||_2 / ||S||_2 vs Final Alignment
    # =========================================================================
    fig7, ax7 = plt.subplots(figsize=(9, 6))
    env_labels = []
    spec_norms = []
    mean_rhos = []

    for env in env_names:
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])
        rep_t = horizons[len(horizons) // 2]
        spec_val = float(np.mean(env_data[rep_t]["relative_spectral_norm_mean"]))
        rho_val = float(np.mean(env_data[rep_t]["rho_exactE_exactTD_mean"]))
        env_labels.append(env)
        spec_norms.append(spec_val)
        mean_rhos.append(rho_val)

    scatter = ax7.scatter(spec_norms, mean_rhos, c=np.arange(len(env_names)), cmap="tab10", s=140, edgecolors="black", zorder=5)
    for i, label in enumerate(env_labels):
        ax7.annotate(label, (spec_norms[i], mean_rhos[i]), textcoords="offset points", xytext=(8, 4), fontsize=9, fontweight="bold")

    ax7.set_xlabel(r"Relative Skew Spectral Norm $\|K\|_2 / \|S\|_2$", fontsize=11, fontweight="bold")
    ax7.set_ylabel(r"Mean Alignment Throughout Learning $\bar{\rho}(g_E^*, g_{\mathrm{TD}}^*)$", fontsize=11, fontweight="bold")
    ax7.set_title(r"Geometric Impact of Non-Reversibility ($\|K\|_2 / \|S\|_2$) on Alignment", fontsize=13, fontweight="bold")
    ax7.axhline(1.0, color="#27ae60", ls=":", lw=1.2)
    ax7.grid(True, alpha=0.3)
    fig7.tight_layout()
    save_fig(fig7, out_dir, "cross_env_alignment_relative_spectral_norm")

    # =========================================================================
    # 8. Bias-Variance Decomposition Across Environments
    # =========================================================================
    fig8, axes8 = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)
    for idx, env in enumerate(env_names):
        r, c = idx // cols, idx % cols
        ax = axes8[r][c]
        env_data = all_data[env]
        horizons = sorted(env_data["horizons"])

        biases = [float(np.mean(env_data[t]["truncation_bias_sq"])) for t in horizons]
        variances = [float(np.mean(env_data[t]["sample_variance"])) for t in horizons]

        x_inds = np.arange(len(horizons))
        w = 0.35
        ax.bar(x_inds - w / 2, biases, w, label=r"$\mathrm{Bias}^2$", color="#e74c3c", alpha=0.85)
        ax.bar(x_inds + w / 2, variances, w, label=r"$\mathrm{Variance}$", color="#3498db", alpha=0.85)

        ax.set_yscale("log")
        ax.set_xticks(x_inds)
        ax.set_xticklabels([f"T={t}" for t in horizons], fontsize=8)
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon T", fontsize=9)
        if c == 0:
            ax.set_ylabel("Error Component (Log)", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right", frameon=True)

    for idx in range(n_envs, rows * cols):
        axes8[idx // cols][idx % cols].axis("off")

    fig8.suptitle("Bias-Variance Decomposition Across Rollout Horizons", fontsize=14, fontweight="bold")
    fig8.tight_layout()
    save_fig(fig8, out_dir, "cross_env_bias_variance_summary")


def main():
    args = parse_args()
    if args.results_dir:
        search_dir = args.results_dir
    elif args.sweep_id:
        search_dir = os.path.join(REPO_ROOT, "results/sweeps", args.sweep_id)
    else:
        # Default to latest sweep directory
        all_dirs = sorted(glob.glob(os.path.join(REPO_ROOT, "results/sweeps/*")))
        if not all_dirs:
            raise FileNotFoundError("No sweep directories found in results/sweeps/")
        search_dir = all_dirs[-1]

    out_dir = args.out_dir or os.path.join(search_dir, "consolidated_plots")
    print(f"Loading results from: {search_dir}")
    print(f"Saving plots to:     {out_dir}")

    all_data = load_all_env_results(search_dir)
    print(f"Loaded {len(all_data)} environments: {list(all_data.keys())}")
    plot_all(all_data, out_dir)
    print("\nAll consolidated plots generated successfully!")


if __name__ == "__main__":
    main()
