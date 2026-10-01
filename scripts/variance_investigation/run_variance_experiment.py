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
        # TD(lambda)
        "mean_dv_norm_td_lambda": [],
        "sigma_v_td_lambda": [],
        "sigma_v_sq_td_lambda": [],
        "snr_v_td_lambda": [],
        "rho_v_td_lambda": [],
        # MC
        "mean_dv_norm_mc": [],
        "sigma_v_mc": [],
        "sigma_v_sq_mc": [],
        "snr_v_mc": [],
        "rho_v_mc": [],
        # E(lambda)
        "mean_dv_norm_e_lambda": [],
        "sigma_v_e_lambda": [],
        "sigma_v_sq_e_lambda": [],
        "snr_v_e_lambda": [],
        "rho_v_e_lambda": [],
    }

    spatial_grids = {
        "var_grid_td0": [],
        "var_grid_td_lambda": [],
        "var_grid_mc": [],
        "var_grid_e_lambda": [],
        "steps": [],
    }

    start_time = time.time()

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

        for alg_key in ["td0", "td_lambda", "mc", "e_lambda"]:
            history[f"mean_dv_norm_{alg_key}"].append(var_metrics[f"mean_dv_norm_{alg_key}"])
            history[f"sigma_v_{alg_key}"].append(var_metrics[f"sigma_v_{alg_key}"])
            history[f"sigma_v_sq_{alg_key}"].append(var_metrics[f"sigma_v_sq_{alg_key}"])
            history[f"snr_v_{alg_key}"].append(var_metrics[f"snr_v_{alg_key}"])
            history[f"rho_v_{alg_key}"].append(var_metrics[f"rho_v_{alg_key}"])

        # Save spatial heatmaps at key milestones (initialization, midpoint, final)
        if update_idx in [0, num_updates // 4, num_updates // 2, num_updates - 1]:
            spatial_grids["steps"].append(update_idx)
            for alg_key in ["td0", "td_lambda", "mc", "e_lambda"]:
                spatial_grids[f"var_grid_{alg_key}"].append(np.array(var_metrics[f"var_grid_{alg_key}"]))

        if update_idx % 20 == 0 or update_idx == num_updates - 1:
            print(
                f"[Step {update_idx:3d}/{num_updates}] Return: {start_val:6.4f} | "
                f"||dv||_TD0: {history['mean_dv_norm_td0'][-1]:.2e} (std {history['sigma_v_td0'][-1]:.2e}) | "
                f"||dv||_E: {history['mean_dv_norm_e_lambda'][-1]:.2e} (std {history['sigma_v_e_lambda'][-1]:.2e}) | "
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
    1. Log-scale Mean Update with Variance Band (||Delta v||_D +/- sigma_v).
    2. Signal-to-Noise Ratio (SNR_v) and Directional Alignment (rho_v).
    3. Spatial variance heatmaps across the transition grid.
    """
    updates = history["update"]

    # Color palette
    colors = {
        "td0": "#1f77b4",        # Blue
        "td_lambda": "#ff7f0e",  # Orange
        "mc": "#d62728",         # Red
        "e_lambda": "#2ca02c",   # Green
    }
    labels = {
        "td0": "TD(0)",
        "td_lambda": r"TD($\lambda=0.95$)",
        "mc": "Monte Carlo",
        "e_lambda": r"$E(\lambda=0.95)$",
    }

    # =========================================================================
    # FIGURE 1: Log-Scale Mean Update with Variance Band
    # =========================================================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2.5]})

    # Top panel: Policy Performance
    ax1.plot(updates, history["policy_return"], color="#333333", lw=2.2, label=r"Policy Return ($V^\pi(s_0)$)")
    ax1.set_ylabel("Expected Return", fontsize=11, fontweight="bold")
    ax1.set_title(f"Policy Optimization & Function-Space Variance Dynamics ({env_name})", fontsize=14, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Mean Update and Variance Bands on Log Scale
    for alg in ["td0", "e_lambda", "td_lambda", "mc"]:
        mean_norm = np.maximum(history[f"mean_dv_norm_{alg}"], 1e-10)
        sigma = np.maximum(history[f"sigma_v_{alg}"], 1e-10)

        lower = np.maximum(mean_norm - sigma, 1e-10)
        upper = mean_norm + sigma

        line, = ax2.plot(updates, mean_norm, color=colors[alg], lw=2.2, label=labels[alg])
        ax2.fill_between(updates, lower, upper, color=colors[alg], alpha=0.18)

    ax2.set_yscale("log")
    ax2.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Function Update $\|\overline{\Delta v}\|_D \pm \sigma_v$ (Log Scale)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "variance_and_mean_update.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "variance_and_mean_update.pdf"))
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: Signal-to-Noise Ratio (SNR_v) & Directional Alignment (rho_v)
    # =========================================================================
    fig, (ax_snr, ax_rho) = plt.subplots(1, 2, figsize=(14, 5))

    for alg in ["td0", "e_lambda", "td_lambda", "mc"]:
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

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "snr_and_alignment.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "snr_and_alignment.pdf"))
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: Spatial Variance Heatmaps
    # =========================================================================
    n_milestones = len(spatial_grids["steps"])
    if n_milestones > 0 and spatial_grids["var_grid_td0"][0] is not None:
        fig, axes = plt.subplots(4, n_milestones, figsize=(4 * n_milestones, 12))
        algs = [("td0", "TD(0)"), ("td_lambda", r"TD($\lambda=0.95$)"), ("mc", "Monte Carlo"), ("e_lambda", r"$E(\lambda=0.95)$")]

        for row_idx, (alg_key, alg_title) in enumerate(algs):
            for col_idx in range(n_milestones):
                step = spatial_grids["steps"][col_idx]
                grid = spatial_grids[f"var_grid_{alg_key}"][col_idx]
                ax = axes[row_idx, col_idx] if n_milestones > 1 else axes[row_idx]

                # Log-scale heatmap
                grid_pos = np.maximum(grid, 1e-12)
                im = ax.imshow(grid_pos, cmap="magma", origin="upper")
                if row_idx == 0:
                    ax.set_title(f"Step {step}", fontsize=12, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(alg_title, fontsize=12, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f"Spatial Variance Field Across Training ({env_name})", fontsize=15, fontweight="bold", y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "spatial_variance_heatmaps.png"), dpi=300)
        fig.savefig(os.path.join(out_dir, "spatial_variance_heatmaps.pdf"))
        plt.close(fig)


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
