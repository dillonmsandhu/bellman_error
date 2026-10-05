"""
run_fixed_batch_horizon_investigation.py

Investigates the effect of rollout horizon T under a strictly fixed total batch size
N = B * T (num_envs * num_steps) to isolate truncation bias (delta^lambda != e) from variance.

Produces:
1. Plot 1 (2 subplots):
   - Top: Policy return V^pi(s_0) vs update step for different T.
   - Bottom: Cosine similarity rho(g_sampled_E^(T), g_exact_E) vs update step for different T.
2. Plot 2 (3 subplots):
   - Theoretical parameter-space alignment during learning:
     Subplot 1: rho(g_exact_E, g_exact_TD)
     Subplot 2: rho(g_exact_E, g_exact_MC)
     Subplot 3: rho(g_exact_TD, g_exact_MC)
3. Plot 3 (2 subplots):
   - Rotation test:
     Subplot 1: rho(g_sampled_E^(T), g_exact_E) across different T
     Subplot 2: rho(g_sampled_E^(T), g_exact_TD) across different T
     (Directly testing if shorter T rotates the empirical update towards TD!)
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from core import helpers, networks
from ppo.sampled_E import Transition
from scripts.variance_investigation.run_sampled_vs_exact_e import extract_critic_flat_grads


def cos_similarity(a: jnp.ndarray, b: jnp.ndarray) -> float:
    """Computes directional cosine similarity between two flat vectors."""
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(jnp.dot(a, b) / (norm_a * norm_b))


def compute_all_exact_critic_gradients(
    train_state: TrainState,
    evaluator,
    network,
    gamma: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """
    Computes exact parameter gradients for:
    1. g_exact_E: Symmetrized Bellman error gradient
    2. g_exact_TD: Expected TD(0) semi-gradient (with detached target)
    3. g_exact_MC: Expected Monte Carlo mean squared error gradient
    along with the start-state value V^pi(s_0).
    """
    S_states = evaluator.obs_stack
    n_actions = evaluator.num_actions

    # Extract current policy matrix pi(s, a)
    pi_dist, _ = network.apply(train_state.params, S_states)
    if hasattr(pi_dist, "probs"):
        old_pi = pi_dist.probs
    else:
        action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
        old_log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
        old_pi = jax.nn.softmax(old_log_probs, axis=-1)

    terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
    old_pi_full = jnp.vstack([old_pi, terminal_policy])

    # Value function & stationary distribution
    V_true = evaluator.compute_true_values_raw(old_pi_full)
    start_val = V_true[evaluator.start_idx]

    mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
    mu_full = jnp.append(mu, 0.0)
    D = jnp.diag(mu_full)

    P = evaluator.P
    I = jnp.eye(evaluator.num_total_states)
    P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
    R_pi = jnp.einsum("sa,sam->s", old_pi_full, evaluator.R)

    A_mat = D @ (I - gamma * P_pi)
    S_mat = 0.5 * (A_mat + A_mat.T)
    K_mat = 0.5 * (A_mat - A_mat.T)

    norm_S_2 = jnp.max(jnp.abs(jnp.linalg.eigvalsh(S_mat)))
    norm_K_2 = jnp.max(jnp.abs(jnp.linalg.eigvalsh(1j * K_mat)))
    rel_spectral_norm = norm_K_2 / (norm_S_2 + 1e-12)

    # 1. Exact E Loss: (V_true - v)^T S_mat (V_true - v)
    def exact_e_loss(params):
        v = network.apply(params, S_states, method=network.value).squeeze()
        diff = V_true - jnp.append(v, 0.0)
        return diff.T @ S_mat @ diff

    # 2. Exact TD Loss: 0.5 * sum_s mu(s) (v(s) - sg(T(v)(s)))^2
    def exact_td_loss(params):
        v = network.apply(params, S_states, method=network.value).squeeze()
        v_full = jnp.append(v, 0.0)
        T_v = R_pi + gamma * (P_pi @ v_full)
        err = v_full - jax.lax.stop_gradient(T_v)
        return 0.5 * jnp.sum(mu_full * (err ** 2))

    # 3. Exact MC Loss: 0.5 * sum_s mu(s) (v(s) - V_true(s))^2
    def exact_mc_loss(params):
        v = network.apply(params, S_states, method=network.value).squeeze()
        v_full = jnp.append(v, 0.0)
        err = v_full - V_true
        return 0.5 * jnp.sum(mu_full * (err ** 2))

    g_exact_E = extract_critic_flat_grads(jax.grad(exact_e_loss)(train_state.params))
    g_exact_TD = extract_critic_flat_grads(jax.grad(exact_td_loss)(train_state.params))
    g_exact_MC = extract_critic_flat_grads(jax.grad(exact_mc_loss)(train_state.params))

    return g_exact_E, g_exact_TD, g_exact_MC, start_val, rel_spectral_norm


def run_single_horizon_experiment(
    env_name: str,
    num_steps: int,
    num_envs: int,
    num_updates: int = 50,
    seed: int = 42,
    num_seeds: int = 4,
    num_minibatches: int = 4,
    lr: float = 3e-4,
) -> Dict[str, np.ndarray]:
    """
    Runs PPO with sampled E critic for a specific (T, B) condition across multiple seeds in parallel via jax.vmap.
    Logs exact gradients, sampled gradient, squared errors, and pairwise cosine similarities across seeds.
    """
    config = {
        "ENV_NAME": env_name,
        "NETWORK_TYPE": "cnn",
        "LAYER_NORM": False,
        "GAMMA": 0.99,
        "LR": lr,
        "ACTOR_LR": lr,
        "NUM_EPOCHS": 4,
        "NUM_MINIBATCHES": num_minibatches,
        "NUM_UPDATES": num_updates,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "GAE_LAMBDA": 0.95,
        "RETURN_LAMBDA": 1.0,
        "CALC_TRUE_VALUES": True,
        "USE_VISUAL_OBS": True,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "k": 32,
        "MAX_STEPS_IN_EPISODE": int(1e6),
        "USE_TABULAR_SIMULATOR": True,
    }

    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    if evaluator is None:
        evaluator = helpers.create_evaluator(config)

    obs_shape = env.observation_space(env_params).shape
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    return_lambda = config["RETURN_LAMBDA"]

    rng = jax.random.PRNGKey(seed)
    rng_seeds = jax.random.split(rng, num_seeds)

    network, _ = networks.initialize_network(
        rng, obs_shape, env, env_params, k=config["k"], n_heads=2, layer_norm=config["LAYER_NORM"]
    )

    def init_single_seed(r):
        r_net, r_env = jax.random.split(r)
        _, net_params = networks.initialize_network(
            r_net, obs_shape, env, env_params, k=config["k"], n_heads=2, layer_norm=config["LAYER_NORM"]
        )
        ts = networks.initialize_flax_train_state(config, network, net_params)
        r_resets = jax.random.split(r_env, num_envs)
        o, es = jax.vmap(env.reset, in_axes=(0, None))(r_resets, env_params)
        return ts, es, o, r_env

    train_states, env_states, obsvs, rng_runs = jax.vmap(init_single_seed)(rng_seeds)

    # Rollout function for a single seed
    def rollout_and_batch_fn(t_state, e_state, o, r):
        def _env_step(carry, _):
            es, cur_o, cur_r = carry
            cur_r, r_act = jax.random.split(cur_r)
            pi, val = network.apply(t_state.params, cur_o)
            act = pi.sample(seed=r_act)
            log_p = pi.log_prob(act)
            cur_r, r_step_base = jax.random.split(cur_r)
            r_step = jax.random.split(r_step_base, num_envs)
            next_o, next_es, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                r_step, es, act, env_params
            )
            real_next_o = info.get("real_next_obs", next_o)
            next_v = network.apply(t_state.params, real_next_o, method=network.value).squeeze()
            return (next_es, next_o, cur_r), (cur_o, real_next_o, rew, done, val, next_v, act, log_p, info)

        r, r_scan = jax.random.split(r)
        (e_state, o, _), (
            obs_seq, next_obs_seq, rew_seq, done_seq, val_seq, next_val_seq, act_seq, log_p_seq, info_seq
        ) = jax.lax.scan(_env_step, (e_state, o, r_scan), None, num_steps)

        traj_batch = Transition(
            done=done_seq,
            action=act_seq,
            value=val_seq,
            next_value=next_val_seq,
            reward=rew_seq,
            log_prob=log_p_seq,
            obs=obs_seq,
            next_obs=next_obs_seq,
            next_target=jnp.zeros_like(val_seq),
            info=info_seq,
        )

        advantages, _ = helpers.calculate_gae(traj_batch, gamma, gae_lambda)
        _, targets = helpers.calculate_gae(traj_batch, gamma, return_lambda)

        is_timeout = traj_batch.info["is_timeout"]
        true_terminal = traj_batch.done & ~is_timeout

        next_targets = jnp.roll(targets, shift=-1, axis=0)
        next_targets = next_targets.at[-1].set(traj_batch.next_value[-1])
        next_targets = jnp.where(is_timeout, traj_batch.next_value, next_targets)
        next_targets = jnp.where(true_terminal, 0.0, next_targets)

        batch = (
            traj_batch.obs,
            traj_batch.action,
            traj_batch.log_prob,
            traj_batch.next_obs,
            true_terminal,
            next_targets,
            advantages,
            targets,
        )
        r, r_batch = jax.random.split(r)
        mbs = helpers.shuffle_and_batch(r_batch, batch, num_minibatches)
        return e_state, o, mbs, r

    # JIT + Vmap over seeds for rollout
    vmapped_rollout = jax.jit(jax.vmap(rollout_and_batch_fn, in_axes=(0, 0, 0, 0)))

    # PPO update function for a single seed
    def ppo_update_fn(t_state, mbs):
        def _update_epoch(update_state, unused):
            def _update_minbatch(cur_t_state, mb_info):
                obs_mb, action_mb, log_prob_mb, next_obs_mb, true_term_mb, next_tgt_mb, advantages_mb, targets_mb = mb_info

                def loss_fn(params):
                    pi = network.apply(params, obs_mb, method=network.policy)
                    log_prob = pi.log_prob(action_mb)
                    entropy = pi.entropy().mean()
                    ratio = jnp.exp(log_prob - log_prob_mb)
                    adv_norm = helpers.post_process_advantage(advantages_mb, config)
                    surr1 = ratio * adv_norm
                    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
                    actor_loss = -jnp.minimum(surr1, surr2).mean()

                    v_i = network.apply(params, obs_mb, method=network.value)
                    v_j = network.apply(params, next_obs_mb, method=network.value)
                    value_loss, _, _ = helpers.e_critic_loss(
                        v_i, targets_mb, v_j, next_tgt_mb, true_term_mb, gamma
                    )
                    return actor_loss + config["VF_COEF"] * value_loss - entropy * config["ENT_COEF"]

                grads = jax.grad(loss_fn)(cur_t_state.params)
                cur_t_state = cur_t_state.apply_gradients(grads=grads)
                return cur_t_state, None

            cur_t_state, cur_mbs = update_state
            cur_t_state, _ = jax.lax.scan(_update_minbatch, cur_t_state, cur_mbs)
            return (cur_t_state, cur_mbs), None

        (t_state, _), _ = jax.lax.scan(_update_epoch, (t_state, mbs), None, config["NUM_EPOCHS"])
        return t_state

    # JIT + Vmap over seeds for PPO update
    vmapped_ppo_update = jax.jit(jax.vmap(ppo_update_fn, in_axes=(0, 0)))

    # JIT + Vmap over seeds for exact gradients
    vmapped_exact = jax.jit(jax.vmap(lambda ts: compute_all_exact_critic_gradients(ts, evaluator, network, gamma)))

    def critic_loss_fn(params, mb_info):
        obs_mb, _, _, next_obs_mb, true_term_mb, next_tgt_mb, _, targets_mb = mb_info
        v_i = network.apply(params, obs_mb, method=network.value)
        v_j = network.apply(params, next_obs_mb, method=network.value)
        value_loss, _, _ = helpers.e_critic_loss(
            v_i, targets_mb, v_j, next_tgt_mb, true_term_mb, gamma
        )
        return value_loss

    def get_single_seed_mb_grad(ts, mb):
        return extract_critic_flat_grads(jax.grad(critic_loss_fn)(ts.params, mb))

    vmapped_mb_grad = jax.jit(jax.vmap(get_single_seed_mb_grad, in_axes=(0, 0)))

    # Multi-seed History storage
    history = {
        "update": [],
        "policy_return_mean": [],
        "policy_return_std": [],
        "policy_return_seeds": [],
        "relative_spectral_norm_mean": [],
        "relative_spectral_norm_std": [],
        "sq_err_mean": [],
        "sq_err_std": [],
        "sq_err_seeds": [],
        "sq_err_td_mean": [],
        "sq_err_td_std": [],
        "rho_SE_EE_mean": [],
        "rho_SE_EE_std": [],
        "rho_SE_EE_seeds": [],
        "rho_SE_ETD_mean": [],
        "rho_SE_ETD_std": [],
        "rho_exactE_exactTD_mean": [],
        "rho_exactE_exactMC_mean": [],
        "rho_exactTD_exactMC_mean": [],
        "truncation_bias_sq": [],
        "sample_variance": [],
    }

    for update_idx in range(num_updates):
        # 1. Compute exact gradients & spectral alignment across seeds in parallel
        g_exact_E, g_exact_TD, g_exact_MC, v0_seeds, rel_spec_seeds = vmapped_exact(train_states)

        # 2. Rollout and build minibatches across seeds in parallel
        env_states, obsvs, minibatches, rng_runs = vmapped_rollout(
            train_states, env_states, obsvs, rng_runs
        )

        # 3. Evaluate sampled critic gradients across minibatches & compute average
        mb_grads = []
        for mb_idx in range(num_minibatches):
            mb_item = jax.tree_util.tree_map(lambda x: x[:, mb_idx], minibatches)
            mb_grads.append(vmapped_mb_grad(train_states, mb_item))

        g_sampled_E = jnp.mean(jnp.stack(mb_grads, axis=0), axis=0)  # (num_seeds, num_params)

        # 4. Multi-seed metrics
        # Squared errors
        sq_err_seeds = jnp.sum((g_sampled_E - g_exact_E) ** 2, axis=-1)
        sq_err_td_seeds = jnp.sum((g_sampled_E - g_exact_TD) ** 2, axis=-1)

        # Directional cosine similarities
        norm_samp = jnp.linalg.norm(g_sampled_E, axis=-1)
        norm_exact = jnp.linalg.norm(g_exact_E, axis=-1)
        rho_SE_EE_seeds = jnp.sum(g_sampled_E * g_exact_E, axis=-1) / jnp.maximum(norm_samp * norm_exact, 1e-12)

        norm_td = jnp.linalg.norm(g_exact_TD, axis=-1)
        rho_SE_ETD_seeds = jnp.sum(g_sampled_E * g_exact_TD, axis=-1) / jnp.maximum(norm_samp * norm_td, 1e-12)

        # Theoretical alignments
        norm_mc = jnp.linalg.norm(g_exact_MC, axis=-1)
        rho_EE_ETD_seeds = jnp.sum(g_exact_E * g_exact_TD, axis=-1) / jnp.maximum(norm_exact * norm_td, 1e-12)
        rho_EE_EMC_seeds = jnp.sum(g_exact_E * g_exact_MC, axis=-1) / jnp.maximum(norm_exact * norm_mc, 1e-12)
        rho_ETD_EMC_seeds = jnp.sum(g_exact_TD * g_exact_MC, axis=-1) / jnp.maximum(norm_td * norm_mc, 1e-12)

        # Bias-Variance decomposition
        mean_samp_g = jnp.mean(g_sampled_E, axis=0)
        mean_exact_g = jnp.mean(g_exact_E, axis=0)
        bias_sq = float(jnp.sum((mean_samp_g - mean_exact_g) ** 2))
        sample_var = float(jnp.mean(jnp.sum((g_sampled_E - mean_samp_g) ** 2, axis=-1)))

        history["update"].append(update_idx)
        history["policy_return_mean"].append(float(jnp.mean(v0_seeds)))
        history["policy_return_std"].append(float(jnp.std(v0_seeds)))
        history["policy_return_seeds"].append(np.array(v0_seeds))

        history["relative_spectral_norm_mean"].append(float(jnp.mean(rel_spec_seeds)))
        history["relative_spectral_norm_std"].append(float(jnp.std(rel_spec_seeds)))

        history["sq_err_mean"].append(float(jnp.mean(sq_err_seeds)))
        history["sq_err_std"].append(float(jnp.std(sq_err_seeds)))
        history["sq_err_seeds"].append(np.array(sq_err_seeds))

        history["sq_err_td_mean"].append(float(jnp.mean(sq_err_td_seeds)))
        history["sq_err_td_std"].append(float(jnp.std(sq_err_td_seeds)))

        history["rho_SE_EE_mean"].append(float(jnp.mean(rho_SE_EE_seeds)))
        history["rho_SE_EE_std"].append(float(jnp.std(rho_SE_EE_seeds)))
        history["rho_SE_EE_seeds"].append(np.array(rho_SE_EE_seeds))

        history["rho_SE_ETD_mean"].append(float(jnp.mean(rho_SE_ETD_seeds)))
        history["rho_SE_ETD_std"].append(float(jnp.std(rho_SE_ETD_seeds)))

        history["rho_exactE_exactTD_mean"].append(float(jnp.mean(rho_EE_ETD_seeds)))
        history["rho_exactE_exactMC_mean"].append(float(jnp.mean(rho_EE_EMC_seeds)))
        history["rho_exactTD_exactMC_mean"].append(float(jnp.mean(rho_ETD_EMC_seeds)))

        history["truncation_bias_sq"].append(bias_sq)
        history["sample_variance"].append(sample_var)

        # 5. PPO update across all seeds in parallel
        train_states = vmapped_ppo_update(train_states, minibatches)

    return {k: np.array(v) for k, v in history.items()}


def generate_all_investigation_plots(
    all_horizon_results: Dict[int, Dict[str, np.ndarray]],
    env_name: str,
    total_batch_size: int,
    out_dir: str,
    num_seeds: int = 4,
):
    """
    Generates all investigation figures with shaded error bands across seeds:
    - Plot 1 (2 subplots): Returns & rho(g_sampled_E, g_exact_E) across different T.
    - Plot 2 (1 plot, 3 lines): Theoretical exact alignments rho(EE, ETD), rho(EE, EMC), rho(ETD, EMC).
    - Plot 3 (2 subplots): Sampled E vs Exact E, and Sampled E vs Exact TD across different T.
    - Plot 4: Squared Error over Learning ||g_sampled_E - g_exact_E||^2 across different T.
    - Plot 5: Average Squared Error and Cosine Alignment vs. Horizon T.
    - Plot 6: Bias-Variance Decomposition across horizons T.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    sorted_steps = sorted(all_horizon_results.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(sorted_steps)))

    # =========================================================================
    # PLOT 1: Two Subplots (Returns and Cosine Similarity to Exact E across T)
    # =========================================================================
    fig1, (ax1_ret, ax1_rho) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for idx, t in enumerate(sorted_steps):
        res = all_horizon_results[t]
        updates = res["update"]
        b = total_batch_size // t
        label_str = f"T = {t} (B = {b})"

        ret_mean = res["policy_return_mean"]
        ret_std = res["policy_return_std"] / np.sqrt(num_seeds)
        ax1_ret.plot(updates, ret_mean, color=colors[idx], lw=2.2, label=label_str)
        ax1_ret.fill_between(updates, ret_mean - 1.96 * ret_std, ret_mean + 1.96 * ret_std, color=colors[idx], alpha=0.15)

        rho_mean = res["rho_SE_EE_mean"]
        rho_std = res["rho_SE_EE_std"] / np.sqrt(num_seeds)
        ax1_rho.plot(updates, rho_mean, color=colors[idx], lw=2.0, label=label_str)
        ax1_rho.fill_between(updates, rho_mean - 1.96 * rho_std, rho_mean + 1.96 * rho_std, color=colors[idx], alpha=0.15)

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
    plot1_path = os.path.join(out_dir, f"{env_name}_returns_and_sampled_vs_exact_E.png")
    fig1.savefig(plot1_path, dpi=300)
    plt.close(fig1)
    print(f"Saved Plot 1 to: {plot1_path}")

    # =========================================================================
    # PLOT 2: Theoretical Parameter-Space Alignment (Single Plot, 3 Lines)
    # =========================================================================
    rep_t = sorted_steps[len(sorted_steps)//2]
    rep_res = all_horizon_results[rep_t]
    updates = rep_res["update"]

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(
        updates,
        rep_res["rho_exactE_exactTD_mean"],
        color="#0984e3",
        lw=2.5,
        label=r"$\rho(g_{\mathrm{exact\_E}}, g_{\mathrm{exact\_TD}})$ (Symmetrized $A+A^\top$ vs. $A$)",
    )
    ax2.plot(
        updates,
        rep_res["rho_exactE_exactMC_mean"],
        color="#6c5ce7",
        lw=2.5,
        label=r"$\rho(g_{\mathrm{exact\_E}}, g_{\mathrm{exact\_MC}})$ ($S^\pi$ vs. $D^\pi$)",
    )
    ax2.plot(
        updates,
        rep_res["rho_exactTD_exactMC_mean"],
        color="#00b894",
        lw=2.5,
        label=r"$\rho(g_{\mathrm{exact\_TD}}, g_{\mathrm{exact\_MC}})$ (TD Semi-Grad vs. MC)",
    )

    ax2.axhline(1.0, color="#27ae60", ls=":", lw=1.8, label=r"Perfect Alignment ($\rho = 1.0$)")
    ax2.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.5)

    ax2.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Directional Cosine Similarity $\rho$", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Theoretical Parameter-Space Alignments Across Policy Optimization ({env_name})",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_ylim(-1.05, 1.05)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot2_path = os.path.join(out_dir, f"{env_name}_theoretical_three_way_cosine.png")
    fig2.savefig(plot2_path, dpi=300)
    plt.close(fig2)
    print(f"Saved Plot 2 to: {plot2_path}")

    # =========================================================================
    # PLOT 3: Rotation Test (Two Subplots: Sampled E vs Exact E, and Sampled E vs Exact TD)
    # =========================================================================
    fig3, (ax3_e, ax3_td) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for idx, t in enumerate(sorted_steps):
        res = all_horizon_results[t]
        updates = res["update"]
        b = total_batch_size // t
        label_str = f"T = {t} (B = {b})"

        rho_e_mean = res["rho_SE_EE_mean"]
        rho_e_std = res["rho_SE_EE_std"] / np.sqrt(num_seeds)
        ax3_e.plot(updates, rho_e_mean, color=colors[idx], lw=2.2, label=label_str)
        ax3_e.fill_between(updates, rho_e_mean - 1.96 * rho_e_std, rho_e_mean + 1.96 * rho_e_std, color=colors[idx], alpha=0.15)

        rho_td_mean = res["rho_SE_ETD_mean"]
        rho_td_std = res["rho_SE_ETD_std"] / np.sqrt(num_seeds)
        ax3_td.plot(updates, rho_td_mean, color=colors[idx], lw=2.2, label=label_str)
        ax3_td.fill_between(updates, rho_td_mean - 1.96 * rho_td_std, rho_td_mean + 1.96 * rho_td_std, color=colors[idx], alpha=0.15)

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
    plot3_path = os.path.join(out_dir, f"{env_name}_sampled_vs_exact_E_and_TD.png")
    fig3.savefig(plot3_path, dpi=300)
    plt.close(fig3)
    print(f"Saved Plot 3 to: {plot3_path}")

    # =========================================================================
    # PLOT 4: Squared Error over Learning ||g_sampled_E - g_exact_E||^2
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6.5))

    for idx, t in enumerate(sorted_steps):
        res = all_horizon_results[t]
        updates = res["update"]
        b = total_batch_size // t
        label_str = f"T = {t} (B = {b})"

        sq_mean = res["sq_err_mean"]
        sq_std = res["sq_err_std"] / np.sqrt(num_seeds)
        ax4.plot(updates, sq_mean, color=colors[idx], lw=2.2, label=label_str)
        ax4.fill_between(updates, np.maximum(sq_mean - 1.96 * sq_std, 1e-8), sq_mean + 1.96 * sq_std, color=colors[idx], alpha=0.15)

    ax4.set_yscale("log")
    ax4.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax4.set_ylabel(r"Squared Error $\|g_{\mathrm{sampled\_E}} - g_{\mathrm{exact\_E}}\|^2$ (Log Scale)", fontsize=11, fontweight="bold")
    ax4.set_title(
        f"Empirical Gradient Squared Error Across Policy Optimization ({env_name}, $N = {total_batch_size}$)\n"
        r"Tracking Truncation Error Magnitude Without Near-Zero Angular Instability",
        fontsize=13,
        fontweight="bold",
    )
    ax4.legend(loc="upper right", frameon=True, fontsize=10)
    ax4.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plot4_path = os.path.join(out_dir, f"{env_name}_squared_error_learning_curves.png")
    fig4.savefig(plot4_path, dpi=300)
    plt.close(fig4)
    print(f"Saved Plot 4 to: {plot4_path}")

    # =========================================================================
    # PLOT 5: Average Metrics vs. Horizon T
    # =========================================================================
    fig5, (ax5_cos, ax5_sq) = plt.subplots(1, 2, figsize=(14, 5.5))
    import matplotlib.ticker as ticker

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
    plot5_path = os.path.join(out_dir, f"{env_name}_avg_metrics_vs_horizon.png")
    fig5.savefig(plot5_path, dpi=300)
    plt.close(fig5)
    print(f"Saved Plot 5 to: {plot5_path}")

    # =========================================================================
    # PLOT 6: Bias-Variance Decomposition Across Horizons T
    # =========================================================================
    fig6, ax6 = plt.subplots(figsize=(9, 6))
    biases = [float(np.mean(all_horizon_results[t]["truncation_bias_sq"])) for t in sorted_steps]
    variances = [float(np.mean(all_horizon_results[t]["sample_variance"])) for t in sorted_steps]

    x_indices = np.arange(len(sorted_steps))
    width = 0.35

    ax6.bar(x_indices - width/2, biases, width, label=r"Truncation $\mathrm{Bias}^2$", color="#e74c3c", alpha=0.85)
    ax6.bar(x_indices + width/2, variances, width, label=r"Sample $\mathrm{Variance}$", color="#3498db", alpha=0.85)

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
    plot6_path = os.path.join(out_dir, f"{env_name}_bias_variance_decomposition.png")
    fig6.savefig(plot6_path, dpi=300)
    plt.close(fig6)
    print(f"Saved Plot 6 to: {plot6_path}")


def main():
    parser = argparse.ArgumentParser(description="Fixed Batch Horizon Investigation")
    parser.add_argument(
        "--env_name",
        type=str,
        default="fourrooms-dense",
        choices=[
            "FourRooms-misc",
            "fourrooms-dense",
            "eightrooms-misc",
            "eightrooms-dense",
            "whirlpool-misc",
            "mountaincar",
            "mountaincar-v0",
            "MountainCar-v0",
            "mountaincar-dense",
            "SpaceInvadersExactValue",
            "spaceinvaders",
        ],
        help="Target environment",
    )
    parser.add_argument("--total_batch_size", type=int, default=16384, help="Fixed total batch size N = B * T (default 128^2 = 16384)")
    parser.add_argument("--num_updates", type=int, default=35, help="Number of policy updates")
    parser.add_argument("--num_seeds", type=int, default=4, help="Number of random seeds in parallel (vmap)")
    parser.add_argument("--seed", type=int, default=42, help="PRNG base seed")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, f"results/horizon_investigation_16k/{args.env_name}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Candidate horizons: T in [8, 64, 128, 512, 1024]
    candidate_T = [8, 64, 128, 512, 1024]
    N = args.total_batch_size
    horizon_list = [t for t in candidate_T if N % t == 0]

    print("=" * 80)
    print(f"STARTING FIXED-BATCH HORIZON INVESTIGATION ({args.env_name})")
    print(f"Total Batch Size N: {N} (STRICTLY FIXED)")
    print(f"Rollout Horizons T: {horizon_list}")
    print(f"Parallel Envs    B: {[N // t for t in horizon_list]}")
    print(f"Policy Updates:     {args.num_updates} | Seeds: {args.num_seeds} (vmapped) | Base Seed: {args.seed}")
    print(f"Output Directory:   {args.out_dir}")
    print("=" * 80)

    all_horizon_results = {}
    for t in horizon_list:
        b = N // t
        print(f"\n--> Running Condition: T = {t:3d}, B = {b:3d} (N = {b*t}, Seeds = {args.num_seeds})...", flush=True)
        t_start = time.time()
        res = run_single_horizon_experiment(
            env_name=args.env_name,
            num_steps=t,
            num_envs=b,
            num_updates=args.num_updates,
            seed=args.seed,
            num_seeds=args.num_seeds,
        )
        elapsed = time.time() - t_start
        all_horizon_results[t] = res
        print(
            f"    Completed in {elapsed:.1f}s | "
            f"Final Return: {res['policy_return_mean'][-1]:.4f} +/- {res['policy_return_std'][-1]:.4f} | "
            f"Mean SqErr: {np.mean(res['sq_err_mean']):.6f} | "
            f"Mean rho(SE, EE): {np.mean(res['rho_SE_EE_mean']):.4f}"
        )

    # Save complete dataset
    data_path = os.path.join(args.out_dir, f"{args.env_name}_fixed_batch_data.npz")
    save_dict = {f"T_{t}_{k}": v for t, d in all_horizon_results.items() for k, v in d.items()}
    np.savez(data_path, **save_dict)
    print(f"\nSaved raw dataset to: {data_path}")

    # Generate all requested figures
    generate_all_investigation_plots(
        all_horizon_results,
        env_name=args.env_name,
        total_batch_size=N,
        out_dir=args.out_dir,
        num_seeds=args.num_seeds,
    )
    print("\nInvestigation complete!")


if __name__ == "__main__":
    main()
