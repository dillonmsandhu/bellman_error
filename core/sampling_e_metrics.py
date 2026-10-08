"""
core/sampling_e_metrics.py

Computes exact vs sampled Tang & Munos E objective metrics and critic parameter gradients:
1. Exact Matrix E: e^T S e
2. Expected Corrected E: (1-gamma)*E_mu[e^2] + (gamma/2)*E_{mu P}[(e - e')^2] + (gamma/2)*(E_mu[e^2] - E_{mu P}[e'^2])
3. Expected Uncorrected E: (1-gamma)*E_mu[e^2] + (gamma/2)*E_{mu P}[(e - e')^2]
4. Empirical Sampled E (corrected & uncorrected): rollouts of B trajectories for T steps
5. Critic parameter gradient cosine similarities and norms
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Dict, Any

from core.gradient_tracking import extract_critic_flat_grads, cos_similarity


def compute_sampling_e_metrics(
    evaluator,
    network,
    params,
    mu: jnp.ndarray,
    V_true: jnp.ndarray,
    P_pi: jnp.ndarray,
    S_mat: jnp.ndarray,
    gamma: float,
    rng: jax.Array,
    num_trajectories: int = 128,
    num_steps: int = 128,
) -> Dict[str, jnp.ndarray]:
    """
    Computes exact, expected, and sampled E metrics and gradients for logging.
    """
    S_states = evaluator.obs_stack
    N = evaluator.num_states
    start_idx = evaluator.start_idx

    # Forward value prediction helper
    def get_e_full(p):
        v = network.apply(p, S_states, method=network.value).squeeze()
        v_full = jnp.append(v, 0.0)
        return V_true - v_full

    # 1. Rollout sampling from mu: B parallel trajectories of T steps
    rng_init, rng_steps = jax.random.split(rng)
    mu_trans = mu[:N] / jnp.maximum(jnp.sum(mu[:N]), 1e-12)
    s0 = jax.random.choice(rng_init, N, shape=(num_trajectories,), p=mu_trans)

    def step_fn(carry_s, step_rng):
        step_rngs = jax.random.split(step_rng, num_trajectories)
        logits = jnp.log(jnp.maximum(P_pi[carry_s], 1e-12))
        next_s_raw = jax.vmap(jax.random.categorical)(step_rngs, logits)
        next_s_carry = jnp.where(next_s_raw == N, start_idx, next_s_raw)
        return next_s_carry, (carry_s, next_s_raw)

    step_rngs = jax.random.split(rng_steps, num_steps)
    _, (traj_s, traj_next_s) = jax.lax.scan(step_fn, s0, step_rngs)

    # 2. Objective functions (differentiable wrt critic params)
    def loss_exact(p):
        e_cur = get_e_full(p)
        return e_cur.T @ S_mat @ e_cur

    def loss_exp_corr(p):
        e_cur = get_e_full(p)
        mc = (1.0 - gamma) * jnp.sum(mu * (e_cur ** 2))
        dirichlet = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e_cur[:, None] - e_cur[None, :]) ** 2))
        corr = 0.5 * gamma * (jnp.sum(mu * (e_cur ** 2)) - jnp.sum((mu @ P_pi) * (e_cur ** 2)))
        return mc + dirichlet + corr

    def loss_exp_uncorr(p):
        e_cur = get_e_full(p)
        mc = (1.0 - gamma) * jnp.sum(mu * (e_cur ** 2))
        dirichlet = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e_cur[:, None] - e_cur[None, :]) ** 2))
        return mc + dirichlet

    def loss_exp_oct8(p):
        e_cur = get_e_full(p)
        mc = (1.0 - gamma) * jnp.sum(mu * (e_cur ** 2))
        dirichlet = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e_cur[:, None] - e_cur[None, :]) ** 2))
        flux = mu - (mu @ P_pi)
        start_anchor = 0.5 * gamma * jnp.sum(flux * (e_cur ** 2))
        return mc + dirichlet + start_anchor

    # Overall sample losses over full rollout
    e_full_cur = get_e_full(params)
    e_s_all = e_full_cur[traj_s]
    e_next_all = e_full_cur[traj_next_s]
    mc_all = (1.0 - gamma) * jnp.mean(e_s_all ** 2)
    dir_all = 0.5 * gamma * jnp.mean((e_s_all - e_next_all) ** 2)
    nu_all = jnp.mean(traj_next_s == N)

    l_exact = loss_exact(params)
    l_exp_corr = loss_exp_corr(params)
    l_exp_uncorr = loss_exp_uncorr(params)
    l_exp_oct8 = loss_exp_oct8(params)
    l_deficit = l_exp_corr - l_exp_uncorr

    l_samp_uncorr = mc_all + dir_all
    l_samp_corr = mc_all + dir_all + 0.5 * gamma * (jnp.mean(e_s_all ** 2) - jnp.mean(e_next_all ** 2))
    l_samp_oct8 = mc_all + dir_all + 0.5 * gamma * nu_all * (e_full_cur[start_idx] ** 2)

    # 3. Exact & Expected Critic Parameter Gradients
    g_exact = extract_critic_flat_grads(jax.grad(loss_exact)(params))
    g_exp_corr = extract_critic_flat_grads(jax.grad(loss_exp_corr)(params))
    g_exp_uncorr = extract_critic_flat_grads(jax.grad(loss_exp_uncorr)(params))
    g_exp_oct8 = extract_critic_flat_grads(jax.grad(loss_exp_oct8)(params))

    norm_g_exact = jnp.linalg.norm(g_exact)
    norm_g_exp_uncorr = jnp.linalg.norm(g_exp_uncorr)

    rho_exp_corr_exact = cos_similarity(g_exp_corr, g_exact)
    rho_exp_uncorr_exact = cos_similarity(g_exp_uncorr, g_exact)
    rho_exp_oct8_exact = cos_similarity(g_exp_oct8, g_exact)

    # 4. Trajectory-Level Gradient Analysis & Full Batch Cosine Alignment
    # Shape of traj_s: (num_steps, num_trajectories) -> transpose to (num_trajectories, num_steps)
    s_trajs = jnp.transpose(traj_s, (1, 0))
    next_s_trajs = jnp.transpose(traj_next_s, (1, 0))

    def single_traj_grad_uncorr(s_b, next_b):
        def l_fn(p):
            e_c = get_e_full(p)
            es = e_c[s_b]
            en = e_c[next_b]
            return (1.0 - gamma) * jnp.mean(es ** 2) + 0.5 * gamma * jnp.mean((es - en) ** 2)
        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    def single_traj_grad_corr(s_b, next_b):
        def l_fn(p):
            e_c = get_e_full(p)
            es = e_c[s_b]
            en = e_c[next_b]
            return (
                (1.0 - gamma) * jnp.mean(es ** 2)
                + 0.5 * gamma * jnp.mean((es - en) ** 2)
                + 0.5 * gamma * (jnp.mean(es ** 2) - jnp.mean(en ** 2))
            )
        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    def single_traj_grad_oct8(s_b, next_b):
        def l_fn(p):
            e_c = get_e_full(p)
            es = e_c[s_b]
            en = e_c[next_b]
            nu_b = jnp.mean(next_b == N)
            return (
                (1.0 - gamma) * jnp.mean(es ** 2)
                + 0.5 * gamma * jnp.mean((es - en) ** 2)
                + 0.5 * gamma * nu_b * (e_c[start_idx] ** 2)
            )
        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    # Vectorized gradient computation across all B individual trajectories
    G_uncorr = jax.vmap(single_traj_grad_uncorr)(s_trajs, next_s_trajs)  # (B, D)
    G_corr = jax.vmap(single_traj_grad_corr)(s_trajs, next_s_trajs)      # (B, D)
    G_oct8 = jax.vmap(single_traj_grad_oct8)(s_trajs, next_s_trajs)      # (B, D)

    # Total average gradient across the full rollout batch (B trajectories)
    g_bar_u = jnp.mean(G_uncorr, axis=0)
    g_bar_c = jnp.mean(G_corr, axis=0)
    g_bar_o = jnp.mean(G_oct8, axis=0)

    # Full batch cosine similarity of the total average gradient to exact g*
    rho_s_u = cos_similarity(g_bar_u, g_exact)
    rho_s_c = cos_similarity(g_bar_c, g_exact)
    rho_s_o = cos_similarity(g_bar_o, g_exact)

    # Trajectory gradient spread: Tr(Var_traj(g)) = (1/B) sum ||g_b - g_bar||^2
    var_traj_u = jnp.mean(jnp.sum((G_uncorr - g_bar_u[None, :]) ** 2, axis=1))
    var_traj_c = jnp.mean(jnp.sum((G_corr - g_bar_c[None, :]) ** 2, axis=1))
    var_traj_o = jnp.mean(jnp.sum((G_oct8 - g_bar_o[None, :]) ** 2, axis=1))

    # Variance of the average gradient estimator g_bar: Var(g_bar) = Var_traj / B
    var_u = var_traj_u / float(num_trajectories)
    var_c = var_traj_c / float(num_trajectories)
    var_o = var_traj_o / float(num_trajectories)

    # Total squared error of the average gradient to exact g*
    mse_u = jnp.sum((g_bar_u - g_exact) ** 2)
    mse_c = jnp.sum((g_bar_c - g_exact) ** 2)
    mse_o = jnp.sum((g_bar_o - g_exact) ** 2)

    norm_u = jnp.linalg.norm(g_bar_u)
    norm_c = jnp.linalg.norm(g_bar_c)
    norm_o = jnp.linalg.norm(g_bar_o)

    return {
        "E_exact": l_exact,
        "E_exp_corr": l_exp_corr,
        "E_exp_uncorr": l_exp_uncorr,
        "E_exp_oct8": l_exp_oct8,
        "E_deficit": l_deficit,
        "E_samp_corr": l_samp_corr,
        "E_samp_uncorr": l_samp_uncorr,
        "E_samp_oct8": l_samp_oct8,
        "norm_g_exact": norm_g_exact,
        "norm_g_exp_uncorr": norm_g_exp_uncorr,
        # Cosine similarities of expected gradients
        "rho_exp_corr_exact": rho_exp_corr_exact,
        "rho_exp_uncorr_exact": rho_exp_uncorr_exact,
        "rho_exp_oct8_exact": rho_exp_oct8_exact,
        # Full-Batch Average Gradient Cosine Similarities to Exact E
        "rho_samp_uncorr_exact": rho_s_u,
        "rho_samp_corr_exact": rho_s_c,
        "rho_samp_oct8_exact": rho_s_o,
        # Trajectory gradient variance (spread of trajectories)
        "var_traj_samp_uncorr": var_traj_u,
        "var_traj_samp_corr": var_traj_c,
        "var_traj_samp_oct8": var_traj_o,
        # Variance of the average gradient estimator
        "var_samp_uncorr": var_u,
        "var_samp_corr": var_c,
        "var_samp_oct8": var_o,
        # Total MSE of average gradient to exact g*
        "mse_samp_uncorr": mse_u,
        "mse_samp_corr": mse_c,
        "mse_samp_oct8": mse_o,
        # Gradient norms of average gradients
        "norm_g_samp_uncorr": norm_u,
        "norm_g_samp_corr": norm_c,
        "norm_g_samp_oct8": norm_o,
    }
