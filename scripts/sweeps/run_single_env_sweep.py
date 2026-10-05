"""
scripts/sweeps/run_single_env_sweep.py

Executes a rollout horizon sweep T in [8, 64, 128, 512, 1024] under a strictly fixed
total batch size N = B * T = 16384 for a single environment, vmapping over seeds using
ppo/sampled_E.py and core/sweep.py.

Logs:
- Cosine similarities: rho(g_samp_E, g_exact_E), rho(g_samp_E, g_exact_TD), rho(g_samp_TD, g_exact_TD)
- Theoretical alignments: rho(g_exact_E, g_exact_TD), rho(g_exact_TD, g_exact_MC), rho(g_exact_E, g_exact_MC)
- Relative spectral norm ||K||_2 / ||S||_2
- Squared gradient errors: ||g_samp_E - g_exact_E||^2
- Bias-Variance decomposition: Total MSE = Truncation Bias^2 + Sample Variance
- Policy returns

Saves per-environment results (.npz) and diagnostic plots.
"""

import os
import sys
import time
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

import jax
import jax.numpy as jnp
from ppo.sampled_E import make_train

# Set publication-quality font embedding
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ALL_ENVS = [
    "fourrooms-dense",
    "FourRooms-misc",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run single environment horizon sweep with PPO sampled E")
    parser.add_argument("--env_name", type=str, default=None, help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index of the environment (0-6)")
    parser.add_argument("--total_batch_size", type=int, default=16384, help="Fixed batch size N = B * T")
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 64, 128, 512, 1024], help="Horizons T to sweep")
    parser.add_argument("--num_updates", type=int, default=35, help="Number of policy updates")
    parser.add_argument("--num_seeds", type=int, default=4, help="Number of random seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID / folder name")
    parser.add_argument("--out_dir", type=str, default=None, help="Output root directory")
    return parser.parse_args()


from core.config import config as default_config


def build_config(env_name: str, horizon: int, total_batch_size: int, num_updates: int, base_seed: int) -> Dict[str, Any]:
    num_envs = total_batch_size // horizon
    is_mountaincar = "mountaincar" in env_name.lower()

    cfg = default_config.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": horizon,
        "NUM_ENVS": num_envs,
        "TOTAL_TIMESTEPS": total_batch_size * num_updates,
        "NUM_UPDATES": num_updates,
        "SEED": base_seed,
        "NETWORK_TYPE": "mlp" if is_mountaincar else "cnn",
        "USE_VISUAL_OBS": False if is_mountaincar else True,
        "LR": 3e-4,
        "NUM_MINIBATCHES": 4,
        "UPDATE_EPOCHS": 4,
        "NUM_EPOCHS": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "RETURN_LAMBDA": 1.0,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": True,
        "LOG_GRADIENT_METRICS": True,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
        "DEBUG": False,
    })
    return cfg


def run_horizon(config: Dict[str, Any], num_seeds: int, base_seed: int) -> Dict[str, np.ndarray]:
    """Runs a single horizon configuration across num_seeds via jax.vmap."""
    train_fn = make_train(config)
    vmapped_train = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))

    rng = jax.random.PRNGKey(base_seed)
    rngs = jax.random.split(rng, num_seeds)

    t0 = time.time()
    out = vmapped_train(rngs, None)
    # Block until ready
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - t0
    print(f"  Horizon T={config['NUM_STEPS']} completed in {elapsed:.2f}s")

    metrics = out["metrics"]

    # Extract metrics across seeds (num_seeds, num_updates)
    rho_SE_EE = np.array(metrics["rho_SE_EE"])
    rho_SE_ETD = np.array(metrics["rho_SE_ETD"])
    rho_STD_ETD = np.array(metrics["rho_STD_ETD"])
    rho_EE_ETD = np.array(metrics["rho_exactE_exactTD"])
    rho_ETD_EMC = np.array(metrics["rho_exactTD_exactMC"])
    rho_EE_EMC = np.array(metrics["rho_exactE_exactMC"])
    rel_spec_norm = np.array(metrics["relative_spectral_norm"])

    sq_err = np.array(metrics["sq_err"])
    sq_err_td = np.array(metrics["sq_err_td"])

    if "v_true_start" in metrics:
        policy_ret = np.array(metrics["v_true_start"])
    elif "mean_rew" in metrics:
        policy_ret = np.array(metrics["mean_rew"])
    else:
        policy_ret = np.zeros_like(rho_SE_EE)

    # Bias-Variance decomposition from stored flat gradients:
    # g_samp_E: (num_seeds, num_updates, num_params)
    # g_exact_E: (num_seeds, num_updates, num_params)
    if "g_samp_E" in metrics and "g_exact_E" in metrics:
        g_samp = np.array(metrics["g_samp_E"])
        g_exact = np.array(metrics["g_exact_E"])

        # Mean sampled gradient across seeds
        g_samp_bar = np.mean(g_samp, axis=0)  # (num_updates, num_params)
        g_exact_bar = np.mean(g_exact, axis=0)  # (num_updates, num_params)

        # Truncation bias^2: ||g_samp_bar - g_exact_bar||_2^2
        bias_sq = np.sum((g_samp_bar - g_exact_bar) ** 2, axis=-1)  # (num_updates,)

        # Sample variance: 1/M * sum_m ||g_samp_m - g_samp_bar||_2^2
        sample_var = np.mean(np.sum((g_samp - g_samp_bar[None, :, :]) ** 2, axis=-1), axis=0)  # (num_updates,)
        total_mse = np.mean(np.sum((g_samp - g_exact) ** 2, axis=-1), axis=0)
    else:
        total_mse = np.mean(sq_err, axis=0)
        bias_sq = total_mse * 0.5
        sample_var = total_mse * 0.5

    updates = np.arange(config["NUM_UPDATES"])

    return {
        "updates": updates,
        "rho_SE_EE_mean": np.mean(rho_SE_EE, axis=0),
        "rho_SE_EE_std": np.std(rho_SE_EE, axis=0),
        "rho_SE_ETD_mean": np.mean(rho_SE_ETD, axis=0),
        "rho_SE_ETD_std": np.std(rho_SE_ETD, axis=0),
        "rho_STD_ETD_mean": np.mean(rho_STD_ETD, axis=0),
        "rho_STD_ETD_std": np.std(rho_STD_ETD, axis=0),
        "rho_exactE_exactTD_mean": np.mean(rho_EE_ETD, axis=0),
        "rho_exactTD_exactMC_mean": np.mean(rho_ETD_EMC, axis=0),
        "rho_exactE_exactMC_mean": np.mean(rho_EE_EMC, axis=0),
        "relative_spectral_norm_mean": np.mean(rel_spec_norm, axis=0),
        "sq_err_mean": np.mean(sq_err, axis=0),
        "sq_err_std": np.std(sq_err, axis=0),
        "sq_err_td_mean": np.mean(sq_err_td, axis=0),
        "sq_err_td_std": np.std(sq_err_td, axis=0),
        "policy_return_mean": np.mean(policy_ret, axis=0),
        "policy_return_std": np.std(policy_ret, axis=0),
        "truncation_bias_sq": bias_sq,
        "sample_variance": sample_var,
        "total_mse": total_mse,
    }


def generate_env_plots(
    all_horizon_results: Dict[int, Dict[str, np.ndarray]],
    env_name: str,
    total_batch_size: int,
    num_seeds: int,
    out_dir: str,
):
    """Generates all 6 diagnostic figures for this single environment."""
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    sorted_steps = sorted(all_horizon_results.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(sorted_steps)))

    # 1. Returns and Cosine Similarity to Exact E
    fig1, (ax1_ret, ax1_rho) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for idx, t in enumerate(sorted_steps):
        res = all_horizon_results[t]
        u = res["updates"]
        b = total_batch_size // t
        label_str = f"T = {t} (B = {b})"

        ret_m = res["policy_return_mean"]
        ret_s = res["policy_return_std"] / np.sqrt(num_seeds)
        ax1_ret.plot(u, ret_m, color=colors[idx], lw=2.2, label=label_str)
        ax1_ret.fill_between(u, ret_m - 1.96 * ret_s, ret_m + 1.96 * ret_s, color=colors[idx], alpha=0.15)

        rho_m = res["rho_SE_EE_mean"]
        rho_s = res["rho_SE_EE_std"] / np.sqrt(num_seeds)
        ax1_rho.plot(u, rho_m, color=colors[idx], lw=2.0, label=label_str)
        ax1_rho.fill_between(u, rho_m - 1.96 * rho_s, rho_m + 1.96 * rho_s, color=colors[idx], alpha=0.15)

    ax1_ret.set_ylabel(r"Policy Return $V^\pi(s_0)$", fontsize=11, fontweight="bold")
    ax1_ret.set_title(f"Policy Optimization Dynamics ({env_name}, Fixed Batch $N = {total_batch_size}$, {num_seeds} Seeds)", fontsize=13, fontweight="bold")
    ax1_ret.legend(loc="lower right", frameon=True, fontsize=10)
    ax1_ret.grid(True, alpha=0.3)

    ax1_rho.axhline(1.0, color="#2ca02c", ls=":", lw=2.0, label=r"Perfect Alignment ($\rho = 1.0$)")
    ax1_rho.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax1_rho.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax1_rho.set_ylabel(r"$\rho(g_{\mathrm{sampled\_E}}, g_{\mathrm{exact\_E}})$", fontsize=11, fontweight="bold")
    ax1_rho.set_ylim(-1.05, 1.05)
    ax1_rho.legend(loc="lower right", frameon=True, fontsize=10)
    ax1_rho.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig1.savefig(os.path.join(out_dir, f"{env_name}_returns_and_sampled_vs_exact_E.{ext}"), dpi=300)
    plt.close(fig1)

    # 2. Theoretical 3-Way Cosine Alignment
    rep_t = sorted_steps[len(sorted_steps) // 2]
    rep_res = all_horizon_results[rep_t]
    u = rep_res["updates"]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(u, rep_res["rho_exactE_exactTD_mean"], color="#0984e3", lw=2.5, label=r"$\rho(g_{\mathrm{exact\_E}}, g_{\mathrm{exact\_TD}})$ (Symmetrized $A+A^\top$ vs. $A$)")
    ax2.plot(u, rep_res["rho_exactE_exactMC_mean"], color="#6c5ce7", lw=2.5, label=r"$\rho(g_{\mathrm{exact\_E}}, g_{\mathrm{exact\_MC}})$ ($S^\pi$ vs. $D^\pi$)")
    ax2.plot(u, rep_res["rho_exactTD_exactMC_mean"], color="#00b894", lw=2.5, label=r"$\rho(g_{\mathrm{exact\_TD}}, g_{\mathrm{exact\_MC}})$ (TD Semi-Grad vs. MC)")
    ax2.axhline(1.0, color="#27ae60", ls=":", lw=1.8, label=r"Perfect Alignment ($\rho = 1.0$)")
    ax2.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax2.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Directional Cosine Similarity $\rho$", fontsize=11, fontweight="bold")
    ax2.set_title(f"Theoretical Parameter-Space Alignments Across Policy Optimization ({env_name})", fontsize=13, fontweight="bold")
    ax2.set_ylim(-1.05, 1.05)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig2.savefig(os.path.join(out_dir, f"{env_name}_theoretical_three_way_cosine.{ext}"), dpi=300)
    plt.close(fig2)

    # 3. Rotation Test: Sampled E vs Exact E, and Sampled E vs Exact TD
    fig3, (ax3_e, ax3_td) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for idx, t in enumerate(sorted_steps):
        res = all_horizon_results[t]
        u = res["updates"]
        b = total_batch_size // t
        label_str = f"T = {t} (B = {b})"

        rho_e_m = res["rho_SE_EE_mean"]
        rho_e_s = res["rho_SE_EE_std"] / np.sqrt(num_seeds)
        ax3_e.plot(u, rho_e_m, color=colors[idx], lw=2.2, label=label_str)
        ax3_e.fill_between(u, rho_e_m - 1.96 * rho_e_s, rho_e_m + 1.96 * rho_e_s, color=colors[idx], alpha=0.15)

        rho_td_m = res["rho_SE_ETD_mean"]
        rho_td_s = res["rho_SE_ETD_std"] / np.sqrt(num_seeds)
        ax3_td.plot(u, rho_td_m, color=colors[idx], lw=2.2, label=label_str)
        ax3_td.fill_between(u, rho_td_m - 1.96 * rho_td_s, rho_td_m + 1.96 * rho_td_s, color=colors[idx], alpha=0.15)

    ax3_e.axhline(1.0, color="#2ca02c", ls=":", lw=2.0, label=r"Target Alignment ($\rho = 1.0$)")
    ax3_e.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax3_e.set_ylabel(r"$\rho(g_{\mathrm{sampled\_E}}^{(T)}, g_{\mathrm{exact\_E}})$", fontsize=11, fontweight="bold")
    ax3_e.set_title("Empirical Sampled E vs. Exact Expected Bellman Error (Target Objective)", fontsize=12, fontweight="bold")
    ax3_e.set_ylim(-1.05, 1.05)
    ax3_e.legend(loc="lower right", frameon=True, fontsize=10)
    ax3_e.grid(True, alpha=0.3)

    ax3_td.axhline(1.0, color="#2ca02c", ls=":", lw=2.0, label=r"Target Alignment ($\rho = 1.0$)")
    ax3_td.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax3_td.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax3_td.set_ylabel(r"$\rho(g_{\mathrm{sampled\_E}}^{(T)}, g_{\mathrm{exact\_TD}})$", fontsize=11, fontweight="bold")
    ax3_td.set_title("Empirical Sampled E vs. Exact Expected TD Semi-Gradient (Truncation Attractor)", fontsize=12, fontweight="bold")
    ax3_td.set_ylim(-1.05, 1.05)
    ax3_td.legend(loc="lower right", frameon=True, fontsize=10)
    ax3_td.grid(True, alpha=0.3)

    fig3.suptitle(
        f"Rollout Horizon Rotation Effect ({env_name}, Fixed Batch $N = {total_batch_size}$, {num_seeds} Seeds)\n"
        r"Testing if Shorter $T$ Rotates Empirical Update from Exact E towards Expected TD",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig3.savefig(os.path.join(out_dir, f"{env_name}_sampled_vs_exact_E_and_TD.{ext}"), dpi=300)
    plt.close(fig3)

    # 4. Squared Error Curves
    fig4, ax4 = plt.subplots(figsize=(10, 6.5))
    for idx, t in enumerate(sorted_steps):
        res = all_horizon_results[t]
        u = res["updates"]
        b = total_batch_size // t
        label_str = f"T = {t} (B = {b})"

        sq_m = res["sq_err_mean"]
        sq_s = res["sq_err_std"] / np.sqrt(num_seeds)
        ax4.plot(u, sq_m, color=colors[idx], lw=2.2, label=label_str)
        ax4.fill_between(u, np.maximum(sq_m - 1.96 * sq_s, 1e-8), sq_m + 1.96 * sq_s, color=colors[idx], alpha=0.15)

    ax4.set_yscale("log")
    ax4.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax4.set_ylabel(r"Squared Error $\|g_{\mathrm{sampled\_E}} - g_{\mathrm{exact\_E}}\|^2$ (Log Scale)", fontsize=11, fontweight="bold")
    ax4.set_title(
        f"Empirical Gradient Squared Error Across Policy Optimization ({env_name}, $N = {total_batch_size}$)\n"
        r"Tracking Truncation Error Magnitude Without Angular Instability",
        fontsize=13,
        fontweight="bold",
    )
    ax4.legend(loc="upper right", frameon=True, fontsize=10)
    ax4.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig4.savefig(os.path.join(out_dir, f"{env_name}_squared_error_learning_curves.{ext}"), dpi=300)
    plt.close(fig4)

    # 5. Average Metrics vs Horizon T
    fig5, (ax5_cos, ax5_sq) = plt.subplots(1, 2, figsize=(14, 5.5))
    avg_rho_EE = [float(np.mean(all_horizon_results[t]["rho_SE_EE_mean"])) for t in sorted_steps]
    avg_rho_ETD = [float(np.mean(all_horizon_results[t]["rho_SE_ETD_mean"])) for t in sorted_steps]
    avg_sq_err = [float(np.mean(all_horizon_results[t]["sq_err_mean"])) for t in sorted_steps]

    ax5_cos.plot(sorted_steps, avg_rho_EE, marker="o", lw=2.5, color="#d63031", label=r"Average $\rho(\bar{g}_{\mathrm{sampled\_E}}, g_{\mathrm{exact\_E}})$")
    ax5_cos.plot(sorted_steps, avg_rho_ETD, marker="s", lw=2.5, color="#0984e3", ls="--", label=r"Average $\rho(\bar{g}_{\mathrm{sampled\_E}}, g_{\mathrm{exact\_TD}})$")
    ax5_cos.axhline(1.0, color="#27ae60", ls=":", lw=1.8, label=r"Target Alignment ($\rho = 1.0$)")
    ax5_cos.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax5_cos.set_xscale("log", base=2)
    ax5_cos.set_xticks(sorted_steps)
    ax5_cos.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax5_cos.set_xlabel("Rollout Horizon $T$ (Fixed Batch $N$)", fontsize=11, fontweight="bold")
    ax5_cos.set_ylabel(r"Average Cosine Similarity $\bar{\rho}$", fontsize=11, fontweight="bold")
    ax5_cos.set_title(f"Directional Alignment vs. Horizon ({env_name})", fontsize=12, fontweight="bold")
    ax5_cos.set_ylim(-1.05, 1.05)
    ax5_cos.legend(loc="lower right", frameon=True, fontsize=9)
    ax5_cos.grid(True, alpha=0.3)

    ax5_sq.plot(sorted_steps, avg_sq_err, marker="d", lw=2.5, color="#6c5ce7", label=r"Mean $\|g_{\mathrm{sampled\_E}} - g_{\mathrm{exact\_E}}\|^2$")
    ax5_sq.set_xscale("log", base=2)
    ax5_sq.set_yscale("log")
    ax5_sq.set_xticks(sorted_steps)
    ax5_sq.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax5_sq.set_xlabel("Rollout Horizon $T$ (Fixed Batch $N$)", fontsize=11, fontweight="bold")
    ax5_sq.set_ylabel(r"Mean Squared Error (Log Scale)", fontsize=11, fontweight="bold")
    ax5_sq.set_title(f"Truncation Squared Error vs. Horizon ({env_name})", fontsize=12, fontweight="bold")
    ax5_sq.legend(loc="upper right", frameon=True, fontsize=9)
    ax5_sq.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig5.savefig(os.path.join(out_dir, f"{env_name}_avg_metrics_vs_horizon.{ext}"), dpi=300)
    plt.close(fig5)

    # 6. Bias-Variance Decomposition
    fig6, ax6 = plt.subplots(figsize=(9, 6))
    biases = [float(np.mean(all_horizon_results[t]["truncation_bias_sq"])) for t in sorted_steps]
    variances = [float(np.mean(all_horizon_results[t]["sample_variance"])) for t in sorted_steps]

    x_indices = np.arange(len(sorted_steps))
    width = 0.35
    ax6.bar(x_indices - width / 2, biases, width, label=r"Truncation $\mathrm{Bias}^2$", color="#e74c3c", alpha=0.85)
    ax6.bar(x_indices + width / 2, variances, width, label=r"Sample $\mathrm{Variance}$", color="#3498db", alpha=0.85)
    ax6.set_yscale("log")
    ax6.set_xticks(x_indices)
    ax6.set_xticklabels([f"T={t}" for t in sorted_steps], fontweight="bold")
    ax6.set_xlabel("Rollout Horizon $T$ (Fixed Batch $N$)", fontsize=11, fontweight="bold")
    ax6.set_ylabel(r"Error Component (Log Scale)", fontsize=11, fontweight="bold")
    ax6.set_title(
        f"Bias-Variance Decomposition Across Horizons ({env_name}, $N={total_batch_size}$)\n"
        r"$\mathrm{Total\ MSE} = \|\mathbb{E}[g_{\mathrm{samp}}] - g_{\mathrm{exact}}\|^2 + \mathrm{Tr}(\mathrm{Var}(g_{\mathrm{samp}}))$",
        fontsize=12,
        fontweight="bold",
    )
    ax6.legend(loc="upper right", frameon=True, fontsize=10)
    ax6.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig6.savefig(os.path.join(out_dir, f"{env_name}_bias_variance_decomposition.{ext}"), dpi=300)
    plt.close(fig6)
    print(f"Generated diagnostic plots in: {out_dir}")


def main():
    args = parse_args()

    # Determine environment name
    if args.env_name is not None:
        env_name = args.env_name
    elif args.env_idx is not None:
        if args.env_idx < 0 or args.env_idx >= len(ALL_ENVS):
            raise ValueError(f"Invalid env_idx {args.env_idx}. Must be between 0 and {len(ALL_ENVS)-1}.")
        env_name = ALL_ENVS[args.env_idx]
    else:
        raise ValueError("Must provide either --env_name or --env_idx")

    sweep_id = args.sweep_id or os.environ.get("SWEEP_ID") or time.strftime("%Y%m%d_%H%M%S")
    base_out_dir = args.out_dir or os.path.join(REPO_ROOT, "results/sweeps", sweep_id)
    env_plot_dir = os.path.join(base_out_dir, env_name)
    os.makedirs(env_plot_dir, exist_ok=True)

    print(f"================================================================")
    print(f"Running Horizon Sweep for: {env_name}")
    print(f"  Fixed Batch Size N: {args.total_batch_size}")
    print(f"  Horizons T:         {args.horizons}")
    print(f"  Updates:            {args.num_updates}")
    print(f"  Seeds:              {args.num_seeds}")
    print(f"  Output Dir:         {base_out_dir}")
    print(f"================================================================")

    all_horizon_results = {}
    for t in args.horizons:
        print(f"\n---> Horizon T = {t} (B = {args.total_batch_size // t})")
        cfg = build_config(env_name, t, args.total_batch_size, args.num_updates, args.seed)
        res = run_horizon(cfg, args.num_seeds, args.seed)
        all_horizon_results[t] = res

    # Save per-environment NPZ
    npz_path = os.path.join(base_out_dir, f"{env_name}.npz")
    save_dict = {"env_name": env_name, "horizons": np.array(args.horizons), "total_batch_size": args.total_batch_size}
    for t, res in all_horizon_results.items():
        for k, v in res.items():
            save_dict[f"T_{t}_{k}"] = v
    np.savez_compressed(npz_path, **save_dict)
    print(f"\nSaved environment sweep results to: {npz_path}")

    # Generate environment plots
    generate_env_plots(
        all_horizon_results=all_horizon_results,
        env_name=env_name,
        total_batch_size=args.total_batch_size,
        num_seeds=args.num_seeds,
        out_dir=env_plot_dir,
    )
    print(f"Finished successfully for {env_name}!")


if __name__ == "__main__":
    main()
