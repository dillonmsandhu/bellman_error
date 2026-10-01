"""
run_variance_experiment.py

Runs online policy optimization (PPO) on FourRooms or EightRooms while logging
closed-form function-space update metrics across TD(0), TD(lambda), MC, and E(lambda).

Produces:
1. Log-scale plot of Mean Update (||Delta v||_D) and Variance Band (sigma_v).
2. Signal-to-Noise Ratio (SNR_v) and Directional Alignment (rho_v) over time.
3. 2D Spatial variance heatmaps across the environment grid.
4. Saved numpy metrics (.npz) and JSON summary.
"""

from __future__ import annotations
import os
import sys
import argparse
import json
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import core.helpers as helpers
import core.networks as networks
import core.bellman_error as bellman_error
from core.variance_metrics import compute_all_variance_metrics


def network_inference(params, network, S, n_actions):
    pi_dist, v = network.apply(params, S)
    v_full = jnp.append(v, 0.0)
    pi_full = jnp.vstack([pi_dist.probs, jnp.ones([1, n_actions]) / n_actions])
    return pi_full, v_full


def run_experiment(env_name: str, num_updates: int = 150, lr: float = 3e-4, seed: int = 42, out_dir: str | None = None):
    print("=" * 70)
    print(f"Starting Closed-Form Variance Investigation on: {env_name}")
    print(f"Updates: {num_updates} | Learning Rate: {lr} | Seed: {seed}")
    print("=" * 70)

    config = {
        "ENV_NAME": env_name,
        "NETWORK_TYPE": "mlp",
        "LAYER_NORM": False,
        "GAMMA": 0.99,
        "LR": lr,
        "ACTOR_LR": lr,
        "NUM_EPOCHS": 4,
        "MINIBATCH_SIZE": 1,
        "NUM_MINIBATCHES": 1,
        "NUM_UPDATES": num_updates,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "GAE_LAMBDA": 0.95,
        "VALUE_LAMBDA": 0.95,
        "CALC_TRUE_VALUES": True,
        "USE_VISUAL_OBS": True,
        "k": 32,
    }

    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    if evaluator is None:
        evaluator = helpers.create_evaluator(config)

    obs_shape = evaluator.obs_stack.shape[1:]
    S = evaluator.obs_stack
    n_actions = evaluator.num_actions
    gamma = evaluator.gamma
    P = evaluator.P

    rng = jax.random.PRNGKey(seed)
    rng_net, rng_run = jax.random.split(rng)

    network, network_params = networks.initialize_network(
        rng_net, obs_shape, env, env_params, k=config["k"], n_heads=2, layer_norm=config["LAYER_NORM"]
    )
    train_state = networks.initialize_flax_train_state(config, network, network_params)

    # Metric accumulators
    history = {
        "update": [],
        "policy_return": [],
        # TD(0)
        "mean_dv_norm_td0": [],
        "sigma_v_td0": [],
        "sigma_v_sq_td0": [],
        "snr_v_td0": [],
        "rho_v_td0": [],
        # E(0)
        "mean_dv_norm_e0": [],
        "sigma_v_e0": [],
        "sigma_v_sq_e0": [],
        "snr_v_e0": [],
        "rho_v_e0": [],
        # TD(lambda)
        "mean_dv_norm_td_lambda": [],
        "sigma_v_td_lambda": [],
        "sigma_v_sq_td_lambda": [],
        "snr_v_td_lambda": [],
        "rho_v_td_lambda": [],
        # E(lambda)
        "mean_dv_norm_e_lambda": [],
        "sigma_v_e_lambda": [],
        "sigma_v_sq_e_lambda": [],
        "snr_v_e_lambda": [],
        "rho_v_e_lambda": [],
        # MC
        "mean_dv_norm_mc": [],
        "sigma_v_mc": [],
        "sigma_v_sq_mc": [],
        "snr_v_mc": [],
        "rho_v_mc": [],
    }

    spatial_grids = {
        "var_grid_td0": [],
        "var_grid_e0": [],
        "var_grid_td_lambda": [],
        "var_grid_e_lambda": [],
        "var_grid_mc": [],
        "steps": [],
    }

    start_time = time.time()
    alg_keys = ["td0", "e0", "td_lambda", "e_lambda", "mc"]

    for update_idx in range(num_updates):
        # 1. Compute exact closed-form variance metrics at the current policy snapshot
        var_metrics = compute_all_variance_metrics(
            evaluator=evaluator,
            network=network,
            params=train_state.params,
            lmbda=config["VALUE_LAMBDA"],
            alpha=config["LR"],
            random_policy=False,
        )

        # Policy evaluation (ground truth return from start state)
        pi_curr, _ = network_inference(train_state.params, network, S, n_actions)
        V_pi_curr = evaluator.compute_true_values_raw(pi_curr)
        start_val = float(V_pi_curr[evaluator.start_idx])

        # Record history
        history["update"].append(update_idx)
        history["policy_return"].append(start_val)

        for alg_key in alg_keys:
            history[f"mean_dv_norm_{alg_key}"].append(var_metrics[f"mean_dv_norm_{alg_key}"])
            history[f"sigma_v_{alg_key}"].append(var_metrics[f"sigma_v_{alg_key}"])
            history[f"sigma_v_sq_{alg_key}"].append(var_metrics[f"sigma_v_sq_{alg_key}"])
            history[f"snr_v_{alg_key}"].append(var_metrics[f"snr_v_{alg_key}"])
            history[f"rho_v_{alg_key}"].append(var_metrics[f"rho_v_{alg_key}"])

        # Save spatial heatmaps at key milestones (initialization, midpoint, final)
        if update_idx in [0, num_updates // 4, num_updates // 2, num_updates - 1]:
            spatial_grids["steps"].append(update_idx)
            for alg_key in alg_keys:
                spatial_grids[f"var_grid_{alg_key}"].append(np.array(var_metrics[f"var_grid_{alg_key}"]))

        if update_idx % 20 == 0 or update_idx == num_updates - 1:
            print(
                f"[Step {update_idx:3d}/{num_updates}] Return: {start_val:6.4f} | "
                f"||dv||_TD0: {history['mean_dv_norm_td0'][-1]:.2e} (std {history['sigma_v_td0'][-1]:.2e}) | "
                f"||dv||_E0: {history['mean_dv_norm_e0'][-1]:.2e} (std {history['sigma_v_e0'][-1]:.2e}) | "
                f"||dv||_E(λ): {history['mean_dv_norm_e_lambda'][-1]:.2e} (std {history['sigma_v_e_lambda'][-1]:.2e}) | "
                f"||dv||_TD(λ): {history['mean_dv_norm_td_lambda'][-1]:.2e} (std {history['sigma_v_td_lambda'][-1]:.2e}) | "
                f"||dv||_MC: {history['mean_dv_norm_mc'][-1]:.2e} (std {history['sigma_v_mc'][-1]:.2e})"
            )

        # 2. Perform 1 step of Exact PPO policy optimization
        old_pi_dist, old_v = network.apply(train_state.params, S)
        old_pi = old_pi_dist.probs
        terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
        old_pi_full = jnp.vstack([old_pi, terminal_policy])
        old_log_pi = jnp.log(old_pi + 1e-8)
        old_v_full = jnp.append(old_v, 0.0)

        P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
        R_pi = jnp.einsum("sa,sam,sam->s", old_pi_full, P, evaluator.R)
        mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
        mu = jnp.append(mu, 0.0)

        def T(v_in):
            return R_pi + gamma * P_pi @ v_in

        A = helpers.compute_exact_advantage(P, evaluator.R, P_pi, R_pi, old_v_full, gamma, config["GAE_LAMBDA"])
        w = mu[:-1, None] * old_pi
        A = helpers.post_process_advantage(A, config, weights=w)

        def loss_fn(p, net):
            pi_inf, v_inf = network_inference(p, net, S, n_actions)
            td_errors = v_inf - jax.lax.stop_gradient(T(v_inf))
            value_loss = 0.5 * jnp.sum(mu * (td_errors ** 2))

            log_pi = jnp.log(pi_inf[:-1, :] + 1e-8)
            entropy = -jnp.sum(mu[:-1] * jnp.sum(pi_inf[:-1, :] * log_pi, axis=-1))

            ratio = jnp.exp(log_pi - old_log_pi)
            surr1 = ratio * A
            surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * A
            actor_loss = -jnp.sum(mu[:-1, None] * jnp.exp(old_log_pi) * jnp.minimum(surr1, surr2))

            total_loss = config["VF_COEF"] * value_loss + actor_loss - config["ENT_COEF"] * entropy
            return total_loss

        grad_fn = jax.grad(loss_fn)
        for _ in range(config["NUM_EPOCHS"]):
            grads = grad_fn(train_state.params, network)
            train_state = train_state.apply_gradients(grads=grads)

    elapsed = time.time() - start_time
    print(f"Optimization finished in {elapsed:.1f}s.")

    # Convert history to numpy
    history_np = {k: np.array(v) for k, v in history.items()}

    # Output directory
    if out_dir is None:
        out_dir = os.path.join(REPO_ROOT, f"results/variance_investigation/{env_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics
    np.savez(os.path.join(out_dir, "metrics.npz"), **history_np)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "env_name": env_name,
                "num_updates": num_updates,
                "lr": lr,
                "seed": seed,
                "final_return": float(history_np["policy_return"][-1]),
                "runtime_seconds": float(elapsed),
            },
            f,
            indent=2,
        )

    # Plot final figures
    plot_results(history_np, spatial_grids, env_name, out_dir)
    print(f"Results and plots successfully saved to: {out_dir}")


def plot_results(history: dict, spatial_grids: dict, env_name: str, out_dir: str):
    """
    Generates publication-quality figures:
    1. variance_and_mean_update: 4-panel overview:
       - Policy Return
       - Expected Step Magnitude ||Delta v||_D
       - Direct CI Range / Standard Deviation sigma_v (Log Scale)
       - Normalized CI Range (sigma_v / ||Delta v||_D, Log Scale)
    2. algorithm_confidence_intervals: Dedicated per-algorithm subplots showing ||Delta v||_D +/- 1.96 * sigma_v.
    3. snr_and_alignment: Signal-to-Noise Ratio (SNR_v) and Directional Alignment (rho_v).
    4. spatial_variance_heatmaps: 2D spatial variance fields across training milestones.
    """
    updates = history["update"]

    # Color palette & labels for all 5 algorithms
    colors = {
        "td0": "#1f77b4",        # Blue
        "e0": "#17becf",         # Teal / Cyan
        "td_lambda": "#ff7f0e",  # Orange
        "e_lambda": "#2ca02c",   # Green
        "mc": "#d62728",         # Red
    }
    labels = {
        "td0": "TD(0)",
        "e0": "E(0)",
        "td_lambda": r"TD($\lambda=0.95$)",
        "e_lambda": r"$E(\lambda=0.95)$",
        "mc": "Monte Carlo",
    }
    algs = ["td0", "e0", "td_lambda", "e_lambda", "mc"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # FIGURE 1: 4-Panel Overview (Return, Mean Step, Direct CI, Normalized CI)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_ret, ax_mean = axes[0, 0], axes[0, 1]
    ax_std, ax_norm = axes[1, 0], axes[1, 1]

    # (a) Top-Left: Policy Return
    ax_ret.plot(updates, history["policy_return"], color="#333333", lw=2.2, label=r"Policy Return $V^\pi(s_0)$")
    ax_ret.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax_ret.set_ylabel("Expected Return", fontsize=11, fontweight="bold")
    ax_ret.set_title("Policy Performance", fontsize=12, fontweight="bold")
    ax_ret.legend(loc="lower right", frameon=True)
    ax_ret.grid(True, alpha=0.3)

    # (b) Top-Right: Expected Update Magnitude ||Delta v||_D
    for alg in algs:
        mean_norm = np.maximum(history[f"mean_dv_norm_{alg}"], 1e-12)
        ax_mean.plot(updates, mean_norm, color=colors[alg], lw=2.0, label=labels[alg])
    ax_mean.set_yscale("log")
    ax_mean.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax_mean.set_ylabel(r"$\|\overline{\Delta v}\|_D$ (Log Scale)", fontsize=11, fontweight="bold")
    ax_mean.set_title(r"Expected Function Step Magnitude $\|\overline{\Delta v}\|_D$", fontsize=12, fontweight="bold")
    ax_mean.legend(loc="upper right", frameon=True, fontsize=9)
    ax_mean.grid(True, which="both", alpha=0.3)

    # (c) Bottom-Left: Direct CI Range / Standard Deviation sigma_v
    for alg in algs:
        sigma = np.maximum(history[f"sigma_v_{alg}"], 1e-12)
        ax_std.plot(updates, sigma, color=colors[alg], lw=2.0, label=labels[alg])
    ax_std.set_yscale("log")
    ax_std.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax_std.set_ylabel(r"Direct CI Range / Std $\sigma_v$ (Log Scale)", fontsize=11, fontweight="bold")
    ax_std.set_title(r"Function-Space Standard Deviation $\sigma_v = \sqrt{\mathrm{Tr}(D K \Sigma_z K^\top)}$", fontsize=12, fontweight="bold")
    ax_std.legend(loc="upper right", frameon=True, fontsize=9)
    ax_std.grid(True, which="both", alpha=0.3)

    # (d) Bottom-Right: Normalized CI Range (sigma_v / ||Delta v||_D)
    for alg in algs:
        mean_norm = np.maximum(history[f"mean_dv_norm_{alg}"], 1e-12)
        sigma = np.maximum(history[f"sigma_v_{alg}"], 1e-12)
        rel_ci = sigma / mean_norm
        ax_norm.plot(updates, rel_ci, color=colors[alg], lw=2.0, label=labels[alg])
    ax_norm.axhline(1.0, color="#666666", ls="--", lw=1.5, label="Noise = Signal (100% of Step)")
    ax_norm.set_yscale("log")
    ax_norm.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax_norm.set_ylabel(r"Normalized CI Range $\sigma_v / \|\overline{\Delta v}\|_D$ (Log Scale)", fontsize=11, fontweight="bold")
    ax_norm.set_title(r"Relative Uncertainty Normalized by Mean ($\sigma_v / \|\overline{\Delta v}\|_D$)", fontsize=12, fontweight="bold")
    ax_norm.legend(loc="upper right", frameon=True, fontsize=9)
    ax_norm.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Function-Space Variance & Update Dynamics ({env_name})", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "variance_and_mean_update.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "variance_and_mean_update.pdf"))
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: Dedicated Per-Algorithm Confidence Interval Panels
    # =========================================================================
    fig_facets, axes_facets = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes_list = axes_facets.flatten()

    for idx, alg in enumerate(algs):
        ax = axes_list[idx]
        mean_norm = np.maximum(history[f"mean_dv_norm_{alg}"], 1e-12)
        sigma = np.maximum(history[f"sigma_v_{alg}"], 1e-12)

        # 95% Confidence Interval: [max(0, mu - 1.96 * sigma), mu + 1.96 * sigma]
        ci_lower = np.maximum(mean_norm - 1.96 * sigma, 1e-12)
        ci_upper = mean_norm + 1.96 * sigma

        ax.plot(updates, mean_norm, color=colors[alg], lw=2.2, label=r"Mean Step $\|\overline{\Delta v}\|_D$")
        ax.fill_between(updates, ci_lower, ci_upper, color=colors[alg], alpha=0.22, label=r"$95\%$ CI ($\pm 1.96 \sigma_v$)")

        ax.set_yscale("log")
        ax.set_title(f"{labels[alg]} Update & Uncertainty", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Function Update (Log Scale)", fontsize=10)
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("Policy Update Step", fontsize=10)

    # 6th panel: Direct comparison of relative confidence interval half-widths
    ax_comp = axes_list[5]
    for alg in algs:
        mean_norm = np.maximum(history[f"mean_dv_norm_{alg}"], 1e-12)
        sigma = np.maximum(history[f"sigma_v_{alg}"], 1e-12)
        ax_comp.plot(updates, sigma / mean_norm, color=colors[alg], lw=1.8, label=labels[alg])
    ax_comp.axhline(1.0, color="#444444", ls="--", lw=1.2)
    ax_comp.set_yscale("log")
    ax_comp.set_title(r"Summary: Relative Uncertainty $\sigma_v / \|\overline{\Delta v}\|_D$", fontsize=11, fontweight="bold")
    ax_comp.set_xlabel("Policy Update Step", fontsize=10)
    ax_comp.set_ylabel(r"Relative Spread (Log Scale)", fontsize=10)
    ax_comp.legend(loc="upper right", frameon=True, fontsize=8)
    ax_comp.grid(True, which="both", alpha=0.3)

    fig_facets.suptitle(f"Algorithm-Specific Confidence Intervals ({env_name})", fontsize=14, fontweight="bold", y=0.99)
    fig_facets.tight_layout()
    fig_facets.savefig(os.path.join(out_dir, "algorithm_confidence_intervals.png"), dpi=300)
    fig_facets.savefig(os.path.join(out_dir, "algorithm_confidence_intervals.pdf"))
    plt.close(fig_facets)

    # =========================================================================
    # FIGURE 3: Signal-to-Noise Ratio (SNR_v) & Directional Alignment (rho_v)
    # =========================================================================
    fig_snr, (ax_snr, ax_rho) = plt.subplots(1, 2, figsize=(14, 5))

    for alg in algs:
        snr = np.maximum(history[f"snr_v_{alg}"], 1e-12)
        rho = history[f"rho_v_{alg}"]

        ax_snr.plot(updates, snr, color=colors[alg], lw=2.0, label=labels[alg])
        ax_rho.plot(updates, rho, color=colors[alg], lw=2.0, label=labels[alg])

    ax_snr.set_yscale("log")
    ax_snr.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax_snr.set_ylabel(r"Signal-to-Noise Ratio ($\mathrm{SNR}_v$)", fontsize=11, fontweight="bold")
    ax_snr.set_title(r"Function-Space SNR ($\|\overline{\Delta v}\|_D^2 / \sigma_v^2$)", fontsize=12, fontweight="bold")
    ax_snr.legend(loc="best", frameon=True)
    ax_snr.grid(True, which="both", alpha=0.3)

    ax_rho.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax_rho.set_ylabel(r"Directional Alignment ($\rho_v \in [0, 1]$)", fontsize=11, fontweight="bold")
    ax_rho.set_title(r"Update Cosine Consistency ($\rho_v$)", fontsize=12, fontweight="bold")
    ax_rho.set_ylim(-0.05, 1.05)
    ax_rho.legend(loc="best", frameon=True)
    ax_rho.grid(True, alpha=0.3)

    fig_snr.tight_layout()
    fig_snr.savefig(os.path.join(out_dir, "snr_and_alignment.png"), dpi=300)
    fig_snr.savefig(os.path.join(out_dir, "snr_and_alignment.pdf"))
    plt.close(fig_snr)

    # =========================================================================
    # FIGURE 4: Spatial Variance Heatmaps
    # =========================================================================
    n_milestones = len(spatial_grids["steps"])
    if n_milestones > 0 and spatial_grids["var_grid_td0"][0] is not None:
        fig_hm, axes_hm = plt.subplots(len(algs), n_milestones, figsize=(4 * n_milestones, 3 * len(algs)))

        for row_idx, alg_key in enumerate(algs):
            for col_idx in range(n_milestones):
                step = spatial_grids["steps"][col_idx]
                grid = spatial_grids[f"var_grid_{alg_key}"][col_idx]
                ax = axes_hm[row_idx, col_idx] if n_milestones > 1 else axes_hm[row_idx]

                # Log-scale heatmap
                grid_pos = np.maximum(grid, 1e-12)
                im = ax.imshow(grid_pos, cmap="magma", origin="upper")
                if row_idx == 0:
                    ax.set_title(f"Step {step}", fontsize=12, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(labels[alg_key], fontsize=12, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

        fig_hm.suptitle(f"Spatial Variance Field Across Training ({env_name})", fontsize=15, fontweight="bold", y=0.99)
        fig_hm.tight_layout()
        fig_hm.savefig(os.path.join(out_dir, "spatial_variance_heatmaps.png"), dpi=300)
        fig_hm.savefig(os.path.join(out_dir, "spatial_variance_heatmaps.pdf"))
        plt.close(fig_hm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run closed-form variance experiment")
    parser.add_argument("--env-name", type=str, default="FourRooms-misc", choices=["FourRooms-misc", "eightrooms-misc", "EightRooms-misc"])
    parser.add_argument("--num-updates", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        env_name=args.env_name,
        num_updates=args.num_updates,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out_dir,
    )
