"""
run_empirical_variance_experiment.py

Runs online policy optimization (PPO) with parallel environments while computing
actual empirical critic updates across True TD(0), TD(lambda), Monte Carlo, E(0), and E(lambda).

For each parallel environment b in {1, ..., B}:
1. Computes the empirical loss and gradient g_b for each algorithm.
2. Applies the parameter step Delta theta_b = -alpha * g_b.
3. Computes the exact function update on all states: Delta v_b = v_{theta + Delta theta_b}(S) - v_theta(S).
4. Uses the actual runs and the true analytical mean update \bar{Delta v}_true to compute:
   - Empirical variance around true mean: \hat{sigma}_v^2 = (1/B) sum_b ||Delta v_b - \bar{Delta v}_true||_D^2
   - Empirical sample variance: \hat{sigma}_{v, sample}^2 = (1/(B-1)) sum_b ||Delta v_b - \hat{Delta v}||_D^2
   - Empirical directional alignment: \hat{rho}_v
5. Compares empirical metrics directly against closed-form theoretical sigma_v and rho_v.
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
from core.variance_metrics import (
    compute_all_variance_metrics,
    build_mdp_transition_statistics,
    extract_policy_matrix,
)


def run_empirical_experiment(
    env_name: str = "FourRooms-misc",
    num_updates: int = 150,
    num_envs: int = 64,
    num_steps: int = 64,
    lr: float = 3e-4,
    seed: int = 42,
    use_visual_obs: bool = False,
    out_dir: str | None = None,
):
    print("=" * 70)
    print(f"Starting Empirical Variance Investigation on: {env_name}")
    print(f"Updates: {num_updates} | Envs: {num_envs} | Steps/Env: {num_steps} | LR: {lr} | Seed: {seed}")
    print("=" * 70)

    config = {
        "ENV_NAME": env_name,
        "NETWORK_TYPE": "mlp",
        "LAYER_NORM": False,
        "GAMMA": 0.99,
        "LR": lr,
        "ACTOR_LR": lr,
        "NUM_EPOCHS": 4,
        "MINIBATCH_SIZE": (num_envs * num_steps) // 4,
        "NUM_MINIBATCHES": 4,
        "NUM_UPDATES": num_updates,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "GAE_LAMBDA": 0.95,
        "VALUE_LAMBDA": 0.95,
        "CALC_TRUE_VALUES": True,
        "USE_VISUAL_OBS": use_visual_obs,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "k": 32,
    }

    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    if evaluator is None:
        evaluator = helpers.create_evaluator(config)

    obs_shape = env.observation_space(env_params).shape
    n_actions = evaluator.num_actions
    gamma = config["GAMMA"]
    alpha = config["LR"]
    lmbda = config["VALUE_LAMBDA"]
    S_all = evaluator.obs_stack
    N = evaluator.num_states

    rng = jax.random.PRNGKey(seed)
    rng_net, rng_run = jax.random.split(rng)

    network, network_params = networks.initialize_network(
        rng_net, obs_shape, env, env_params, k=config["k"], n_heads=2, layer_norm=config["LAYER_NORM"]
    )
    train_state = networks.initialize_flax_train_state(config, network, network_params)

    # Initialize parallel environments
    rng_run, _rng = jax.random.split(rng_run)
    reset_rng = jax.random.split(_rng, num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    algs = ["td0", "e0", "td_lambda", "e_lambda", "mc"]

    # History tracking
    history = {
        "update": [],
        "policy_return": [],
    }
    for alg in algs:
        # Theoretical closed-form
        history[f"mean_dv_norm_{alg}_theory"] = []
        history[f"sigma_v_{alg}_theory"] = []
        history[f"rho_v_{alg}_theory"] = []
        # Empirical from parallel environments
        history[f"mean_dv_norm_{alg}_emp"] = []
        history[f"sigma_v_{alg}_emp"] = []
        history[f"sigma_v_{alg}_sample"] = []
        history[f"rho_v_{alg}_emp"] = []
        history[f"rel_unc_{alg}_emp"] = []

    start_time = time.time()

    for update_idx in range(num_updates):
        # ---------------------------------------------------------------------
        # 1. Theoretical Closed-Form Metrics at Current Policy Snapshot
        # ---------------------------------------------------------------------
        theory_metrics = compute_all_variance_metrics(
            evaluator=evaluator,
            network=network,
            params=train_state.params,
            lmbda=lmbda,
            alpha=alpha,
            random_policy=False,
        )

        pi_mat = extract_policy_matrix(evaluator, network, train_state.params)
        v_curr_all = network.apply(train_state.params, S_all, method=network.value).squeeze()
        V_pi_curr = evaluator.compute_true_values_raw(pi_mat)
        start_val = float(V_pi_curr[evaluator.start_idx])
        stats = build_mdp_transition_statistics(evaluator, pi_mat, v_curr_all)
        D = stats["D"]

        # Theoretical expected update vectors Delta v_true = -alpha * K * mean_z
        K = theory_metrics.get("K", None)
        if K is None:
            from core.ntk import compute_eNTK
            K = compute_eNTK(train_state.params, S_all, network)

        mean_z_td0 = stats["d_gamma"] * stats["delta"]
        I = jnp.eye(N)
        gl = gamma * lmbda
        L = jnp.linalg.inv(I - gl * stats["P_trans"])
        mean_z_td_lambda = stats["d_gamma"] * (L @ stats["delta"])
        mean_z_mc = stats["d_gamma"] * stats["e"]
        AL = stats["A"] @ L
        mean_z_e_lambda = 0.5 * (AL + AL.T) @ stats["e"]
        A_mat = stats["A"]
        mean_z_e0 = 0.5 * (A_mat + A_mat.T) @ stats["e"]

        true_dv = {
            "td0": -alpha * (K @ mean_z_td0),
            "td_lambda": -alpha * (K @ mean_z_td_lambda),
            "mc": -alpha * (K @ mean_z_mc),
            "e_lambda": -alpha * (K @ mean_z_e_lambda),
            "e0": -alpha * (K @ mean_z_e0),
        }

        # ---------------------------------------------------------------------
        # 2. Collect Rollouts Across All NUM_ENVS Parallel Environments
        # ---------------------------------------------------------------------
        def _env_step(carry, _):
            e_state, o, r = carry
            r, _r = jax.random.split(r)
            pi, val = network.apply(train_state.params, o)
            act = pi.sample(seed=_r)
            log_p = pi.log_prob(act)
            r, _r = jax.random.split(r)
            r_step = jax.random.split(_r, num_envs)
            next_o, next_es, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                r_step, e_state, act, env_params
            )
            real_next_o = info.get("real_next_obs", next_o)
            next_v = network.apply(train_state.params, real_next_o, method=network.value).squeeze()
            return (next_es, next_o, r), (o, real_next_o, rew, done, val, next_v, act, log_p)

        rng_run, _rng = jax.random.split(rng_run)
        (env_state, obsv, _), (obs_seq, next_obs_seq, rew_seq, done_seq, val_seq, next_val_seq, act_seq, log_p_seq) = (
            jax.lax.scan(_env_step, (env_state, obsv, _rng), None, num_steps)
        )

        # ---------------------------------------------------------------------
        # 3. Compute Per-Environment Target Sequences (Shape: NUM_STEPS, NUM_ENVS)
        # ---------------------------------------------------------------------
        # (a) True TD(0)
        targets_td0 = rew_seq + gamma * (1.0 - done_seq) * next_val_seq

        # (b) TD(lambda)
        deltas = rew_seq + gamma * (1.0 - done_seq) * next_val_seq - val_seq
        def gae_scan(carry, elem):
            gae = carry
            d, dn = elem
            gae = d + gamma * lmbda * (1.0 - dn) * gae
            return gae, gae
        _, adv_lambda = jax.lax.scan(gae_scan, jnp.zeros(num_envs), (deltas, done_seq), reverse=True)
        targets_td_lambda = val_seq + adv_lambda

        # (c) Monte Carlo
        def mc_scan(carry, elem):
            G = carry
            r, dn = elem
            G = r + gamma * (1.0 - dn) * G
            return G, G
        _, targets_mc = jax.lax.scan(mc_scan, next_val_seq[-1], (rew_seq, done_seq), reverse=True)

        # ---------------------------------------------------------------------
        # 4. Compute Per-Environment Critic Gradients & Function Updates Delta v_b
        # ---------------------------------------------------------------------
        # Swap axes to (NUM_ENVS, NUM_STEPS, ...) so env is leading dimension
        obs_by_env = jnp.swapaxes(obs_seq, 0, 1)
        next_obs_by_env = jnp.swapaxes(next_obs_seq, 0, 1)
        done_by_env = jnp.swapaxes(done_seq, 0, 1)
        targets_td0_env = jnp.swapaxes(targets_td0, 0, 1)
        targets_td_lambda_env = jnp.swapaxes(targets_td_lambda, 0, 1)
        targets_mc_env = jnp.swapaxes(targets_mc, 0, 1)

        # Helper: function update from single env gradient
        def compute_single_env_dv(g_b):
            p_new = jax.tree_util.tree_map(lambda p, g: p - alpha * g, train_state.params, g_b)
            v_new = network.apply(p_new, S_all, method=network.value).squeeze()
            return v_new - v_curr_all

        # Regression-based algorithms (TD0, TD_lambda, MC)
        def compute_regression_algo_dv(tgt_by_env):
            def loss_single_env(p, o_b, y_b):
                v_b = network.apply(p, o_b, method=network.value).squeeze()
                return 0.5 * jnp.mean((v_b - jax.lax.stop_gradient(y_b)) ** 2)
            grad_fn = jax.vmap(jax.grad(loss_single_env), in_axes=(None, 0, 0))
            grads = grad_fn(train_state.params, obs_by_env, tgt_by_env)
            return jax.vmap(compute_single_env_dv)(grads)

        dv_env = {
            "td0": compute_regression_algo_dv(targets_td0_env),
            "td_lambda": compute_regression_algo_dv(targets_td_lambda_env),
            "mc": compute_regression_algo_dv(targets_mc_env),
        }

        # E(0) algorithm
        def loss_e0_single_env(p, o_b, no_b, dn_b, tgt_b):
            v_i = network.apply(p, o_b, method=network.value).squeeze()
            v_j = network.apply(p, no_b, method=network.value).squeeze()
            e_i = jax.lax.stop_gradient(tgt_b) - v_i
            next_tgt = jnp.roll(tgt_b, shift=-1).at[-1].set(0.0)
            e_j = jnp.where(dn_b, 0.0, jax.lax.stop_gradient(next_tgt) - v_j)
            mag = (1.0 - gamma) * (e_i ** 2)
            lap = 0.5 * gamma * ((e_i - e_j) ** 2)
            return jnp.mean(mag + lap)

        grad_e0_fn = jax.vmap(jax.grad(loss_e0_single_env), in_axes=(None, 0, 0, 0, 0))
        grads_e0 = grad_e0_fn(train_state.params, obs_by_env, next_obs_by_env, done_by_env, targets_mc_env)
        dv_env["e0"] = jax.vmap(compute_single_env_dv)(grads_e0)

        # E(lambda) algorithm
        def loss_e_lambda_single_env(p, o_b, no_b, dn_b, tgt_b):
            v_i = network.apply(p, o_b, method=network.value).squeeze()
            v_j = network.apply(p, no_b, method=network.value).squeeze()
            e_i = jax.lax.stop_gradient(tgt_b) - v_i
            next_tgt = jnp.roll(tgt_b, shift=-1).at[-1].set(0.0)
            e_j = jnp.where(dn_b, 0.0, jax.lax.stop_gradient(next_tgt) - v_j)
            tilde_gamma = gamma * (1.0 - lmbda) / (1.0 - gamma * lmbda + 1e-8)
            mag = (1.0 - tilde_gamma) * (e_i ** 2)
            lap = 0.5 * tilde_gamma * ((e_i - e_j) ** 2)
            return jnp.mean(mag + lap)

        grad_e_lambda_fn = jax.vmap(jax.grad(loss_e_lambda_single_env), in_axes=(None, 0, 0, 0, 0))
        grads_e_lambda = grad_e_lambda_fn(train_state.params, obs_by_env, next_obs_by_env, done_by_env, targets_td_lambda_env)
        dv_env["e_lambda"] = jax.vmap(compute_single_env_dv)(grads_e_lambda)

        # ---------------------------------------------------------------------
        # 5. Compute Empirical Variance & Alignment Across Parallel Envs
        # ---------------------------------------------------------------------
        history["update"].append(update_idx)
        history["policy_return"].append(start_val)

        for alg in algs:
            dv_b = dv_env[alg]  # shape (NUM_ENVS, N)
            mu_true = true_dv[alg]  # shape (N,)

            # Empirical sample mean
            mu_emp = jnp.mean(dv_b, axis=0)
            norm_mu_emp = float(jnp.sqrt(jnp.maximum(mu_emp.T @ D @ mu_emp, 0.0)))
            norm_mu_true = float(jnp.sqrt(jnp.maximum(mu_true.T @ D @ mu_true, 0.0)))

            # Empirical variance around true mean: (1/B) sum_b ||Delta v_b - mu_true||_D^2
            diff_true = dv_b - mu_true[None, :]
            var_emp_true = float(jnp.mean(jnp.sum((diff_true @ D) * diff_true, axis=-1)))
            sigma_emp_true = float(jnp.sqrt(max(var_emp_true, 0.0)))

            # Empirical sample variance: (1/(B-1)) sum_b ||Delta v_b - mu_emp||_D^2
            diff_sample = dv_b - mu_emp[None, :]
            var_sample = float(jnp.sum(jnp.sum((diff_sample @ D) * diff_sample, axis=-1)) / max(num_envs - 1, 1))
            sigma_sample = float(jnp.sqrt(max(var_sample, 0.0)))

            # Empirical cosine directional alignment: mean_b <Delta v_b, mu_true>_D / (||Delta v_b|| * ||mu_true||)
            norms_dv_b = jnp.sqrt(jnp.maximum(jnp.sum((dv_b @ D) * dv_b, axis=-1), 1e-12))
            inners = jnp.sum((dv_b @ D) * mu_true[None, :], axis=-1)
            cos_sims = inners / (norms_dv_b * max(norm_mu_true, 1e-12))
            rho_emp = float(jnp.clip(jnp.mean(cos_sims), -1.0, 1.0))

            # Store metrics
            history[f"mean_dv_norm_{alg}_theory"].append(theory_metrics[f"mean_dv_norm_{alg}"])
            history[f"sigma_v_{alg}_theory"].append(theory_metrics[f"sigma_v_{alg}"])
            history[f"rho_v_{alg}_theory"].append(theory_metrics[f"rho_v_{alg}"])

            history[f"mean_dv_norm_{alg}_emp"].append(norm_mu_emp)
            history[f"sigma_v_{alg}_emp"].append(sigma_emp_true)
            history[f"sigma_v_{alg}_sample"].append(sigma_sample)
            history[f"rho_v_{alg}_emp"].append(rho_emp)
            history[f"rel_unc_{alg}_emp"].append(sigma_emp_true / max(norm_mu_true, 1e-12))

        if update_idx % 20 == 0 or update_idx == num_updates - 1:
            print(
                f"[Step {update_idx:3d}/{num_updates}] Return: {start_val:6.4f} | "
                f"σ_TD0: {history['sigma_v_td0_emp'][-1]:.2e} (thy {history['sigma_v_td0_theory'][-1]:.2e}) | "
                f"σ_E0: {history['sigma_v_e0_emp'][-1]:.2e} (thy {history['sigma_v_e0_theory'][-1]:.2e}) | "
                f"σ_E(λ): {history['sigma_v_e_lambda_emp'][-1]:.2e} (thy {history['sigma_v_e_lambda_theory'][-1]:.2e}) | "
                f"σ_TD(λ): {history['sigma_v_td_lambda_emp'][-1]:.2e} (thy {history['sigma_v_td_lambda_theory'][-1]:.2e}) | "
                f"σ_MC: {history['sigma_v_mc_emp'][-1]:.2e} (thy {history['sigma_v_mc_theory'][-1]:.2e})"
            )

        # ---------------------------------------------------------------------
        # 6. PPO Policy Optimization Step (Using Batched TD(lambda) Targets)
        # ---------------------------------------------------------------------
        # Flatten transitions for PPO mini-batch updates
        b_obs = obs_seq.reshape(-1, *obs_shape)
        b_act = act_seq.reshape(-1)
        b_log_p = log_p_seq.reshape(-1)
        b_adv = adv_lambda.reshape(-1)
        b_targets = targets_td_lambda.reshape(-1)
        b_adv_norm = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        def ppo_loss_fn(p):
            pi_inf, v_inf = network.apply(p, b_obs)
            # Actor loss
            log_prob = pi_inf.log_prob(b_act)
            ratio = jnp.exp(log_prob - b_log_p)
            surr1 = ratio * b_adv_norm
            surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * b_adv_norm
            actor_loss = -jnp.minimum(surr1, surr2).mean()
            entropy = pi_inf.entropy().mean()
            # Critic loss
            value_loss = 0.5 * jnp.mean((v_inf - b_targets) ** 2)
            return config["VF_COEF"] * value_loss + actor_loss - config["ENT_COEF"] * entropy

        grads = jax.grad(ppo_loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)

    elapsed = time.time() - start_time
    print(f"Empirical investigation finished in {elapsed:.1f}s.")

    # Save metrics
    if out_dir is None:
        out_dir = os.path.join(REPO_ROOT, f"results/empirical_variance_investigation/{env_name}")
    os.makedirs(out_dir, exist_ok=True)

    history_np = {k: np.array(v) for k, v in history.items()}
    np.savez(os.path.join(out_dir, "empirical_metrics.npz"), **history_np)
    with open(os.path.join(out_dir, "empirical_summary.json"), "w") as f:
        json.dump(
            {
                "env_name": env_name,
                "num_updates": num_updates,
                "num_envs": num_envs,
                "num_steps": num_steps,
                "lr": lr,
                "seed": seed,
                "final_return": float(history_np["policy_return"][-1]),
                "runtime_seconds": float(elapsed),
            },
            f,
            indent=2,
        )

    # Plot final figures
    plot_empirical_results(history_np, env_name, out_dir, num_envs)
    print(f"Empirical results and plots successfully saved to: {out_dir}")


def plot_empirical_results(history: dict, env_name: str, out_dir: str, num_envs: int):
    """
    Generates publication figures comparing empirical parallel-environment metrics to theoretical closed-form:
    1. empirical_vs_theoretical_variance: Side-by-side comparison of empirical hat{sigma}_v vs closed-form sigma_v.
    2. empirical_mean_and_confidence_intervals: Faceted plots showing mean step and empirical uncertainty.
    3. empirical_relative_uncertainty: Empirical noise-to-signal ratio hat{sigma}_v / ||Delta v_true||_D.
    4. empirical_alignment: Empirical directional cosine alignment hat{rho}_v.
    """
    updates = history["update"]

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
    # FIGURE 1: Empirical vs. Theoretical Variance Comparison
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Empirical Standard Deviation from B Parallel Environments
    for alg in algs:
        sig_emp = np.maximum(history[f"sigma_v_{alg}_emp"], 1e-12)
        sig_thy = np.maximum(history[f"sigma_v_{alg}_theory"], 1e-12)
        ax1.plot(updates, sig_emp, color=colors[alg], lw=2.2, label=f"{labels[alg]} (Empirical)")
        ax1.plot(updates, sig_thy, color=colors[alg], ls="--", lw=1.5, alpha=0.7, label=f"{labels[alg]} (Theory)")

    ax1.set_yscale("log")
    ax1.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Standard Deviation $\sigma_v$ (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Empirical vs. Theoretical Standard Deviation ({num_envs} Parallel Envs)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)
    ax1.grid(True, which="both", alpha=0.3)

    # Right: Policy Return
    ax2.plot(updates, history["policy_return"], color="#222222", lw=2.2, label=r"Policy Return $V^\pi(s_0)$")
    ax2.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Expected Return", fontsize=11, fontweight="bold")
    ax2.set_title("Policy Performance", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Empirical Parallel Environment Validation ({env_name})", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "empirical_vs_theoretical_variance.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "empirical_vs_theoretical_variance.pdf"))
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: Dedicated Per-Algorithm Confidence Interval Panels (Empirical Spread)
    # =========================================================================
    fig_facets, axes_facets = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes_list = axes_facets.flatten()

    for idx, alg in enumerate(algs):
        ax = axes_list[idx]
        mean_true = np.maximum(history[f"mean_dv_norm_{alg}_theory"], 1e-12)
        mean_emp = np.maximum(history[f"mean_dv_norm_{alg}_emp"], 1e-12)
        sig_emp = np.maximum(history[f"sigma_v_{alg}_emp"], 1e-12)

        ci_lower = np.maximum(mean_true - 1.96 * sig_emp, 1e-12)
        ci_upper = mean_true + 1.96 * sig_emp

        ax.plot(updates, mean_true, color="#222222", lw=2.0, ls="--", label=r"True Mean $\|\overline{\Delta v}\|_D$")
        ax.plot(updates, mean_emp, color=colors[alg], lw=2.2, label=r"Empirical Mean $\|\widehat{\Delta v}\|_D$")
        ax.fill_between(updates, ci_lower, ci_upper, color=colors[alg], alpha=0.22, label=rf"Empirical $95\%$ CI ($B={num_envs}$)")

        ax.set_yscale("log")
        ax.set_title(f"{labels[alg]}", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Function Step (Log Scale)", fontsize=10)
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("Policy Update Step", fontsize=10)

    # 6th panel: Empirical Relative Uncertainty Comparison
    ax_comp = axes_list[5]
    for alg in algs:
        rel_unc = np.maximum(history[f"rel_unc_{alg}_emp"], 1e-12)
        ax_comp.plot(updates, rel_unc, color=colors[alg], lw=1.8, label=labels[alg])
    ax_comp.axhline(1.0, color="#444444", ls="--", lw=1.2, label="Noise = Signal (100%)")
    ax_comp.set_yscale("log")
    ax_comp.set_title(r"Empirical Relative Uncertainty $\widehat{\sigma}_v / \|\overline{\Delta v}\|_D$", fontsize=11, fontweight="bold")
    ax_comp.set_xlabel("Policy Update Step", fontsize=10)
    ax_comp.set_ylabel(r"Relative Spread (Log Scale)", fontsize=10)
    ax_comp.legend(loc="upper right", frameon=True, fontsize=8)
    ax_comp.grid(True, which="both", alpha=0.3)

    fig_facets.suptitle(f"Empirical Function Updates & Confidence Intervals ({env_name})", fontsize=14, fontweight="bold", y=0.99)
    fig_facets.tight_layout()
    fig_facets.savefig(os.path.join(out_dir, "empirical_mean_and_confidence_intervals.png"), dpi=300)
    fig_facets.savefig(os.path.join(out_dir, "empirical_mean_and_confidence_intervals.pdf"))
    plt.close(fig_facets)

    # =========================================================================
    # FIGURE 3: Empirical Directional Alignment (rho_v)
    # =========================================================================
    fig_rho, (ax_rho1, ax_rho2) = plt.subplots(1, 2, figsize=(14, 5))

    for alg in algs:
        rho_emp = history[f"rho_v_{alg}_emp"]
        rho_thy = history[f"rho_v_{alg}_theory"]
        ax_rho1.plot(updates, rho_emp, color=colors[alg], lw=2.0, label=labels[alg])
        ax_rho2.plot(updates, rho_thy, color=colors[alg], lw=2.0, ls="--", label=labels[alg])

    ax_rho1.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax_rho1.set_ylabel(r"Empirical Cosine Similarity $\widehat{\rho}_v \in [-1, 1]$", fontsize=11, fontweight="bold")
    ax_rho1.set_title("Empirical Directional Alignment Across Envs", fontsize=12, fontweight="bold")
    ax_rho1.set_ylim(-0.2, 1.05)
    ax_rho1.legend(loc="best", frameon=True)
    ax_rho1.grid(True, alpha=0.3)

    ax_rho2.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax_rho2.set_ylabel(r"Theoretical Cosine Similarity $\rho_v \in [0, 1]$", fontsize=11, fontweight="bold")
    ax_rho2.set_title("Theoretical Closed-Form Alignment", fontsize=12, fontweight="bold")
    ax_rho2.set_ylim(-0.05, 1.05)
    ax_rho2.legend(loc="best", frameon=True)
    ax_rho2.grid(True, alpha=0.3)

    fig_rho.suptitle(f"Update Directional Consistency ({env_name})", fontsize=14, fontweight="bold", y=0.99)
    fig_rho.tight_layout()
    fig_rho.savefig(os.path.join(out_dir, "empirical_alignment.png"), dpi=300)
    fig_rho.savefig(os.path.join(out_dir, "empirical_alignment.pdf"))
    plt.close(fig_rho)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run empirical variance experiment across parallel environments")
    parser.add_argument("--env-name", type=str, default="FourRooms-misc", choices=["FourRooms-misc", "eightrooms-misc", "EightRooms-misc"])
    parser.add_argument("--num-updates", type=int, default=150)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-visual-obs", action="store_true", default=False)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    run_empirical_experiment(
        env_name=args.env_name,
        num_updates=args.num_updates,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        lr=args.lr,
        seed=args.seed,
        use_visual_obs=args.use_visual_obs,
        out_dir=args.out_dir,
    )
