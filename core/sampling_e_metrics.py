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

    def loss_samp_corr(p):
        e_cur = get_e_full(p)
        e_s = e_cur[traj_s]
        e_next = e_cur[traj_next_s]
        mc = (1.0 - gamma) * jnp.mean(e_s ** 2)
        dirichlet = 0.5 * gamma * jnp.mean((e_s - e_next) ** 2)
        corr = 0.5 * gamma * (jnp.mean(e_s ** 2) - jnp.mean(e_next ** 2))
        return mc + dirichlet + corr

    def loss_samp_uncorr(p):
        e_cur = get_e_full(p)
        e_s = e_cur[traj_s]
        e_next = e_cur[traj_next_s]
        mc = (1.0 - gamma) * jnp.mean(e_s ** 2)
        dirichlet = 0.5 * gamma * jnp.mean((e_s - e_next) ** 2)
        return mc + dirichlet

    # Compute loss scalar values
    l_exact = loss_exact(params)
    l_exp_corr = loss_exp_corr(params)
    l_exp_uncorr = loss_exp_uncorr(params)
    l_deficit = l_exp_corr - l_exp_uncorr
    l_samp_corr = loss_samp_corr(params)
    l_samp_uncorr = loss_samp_uncorr(params)

    # Compute critic parameter gradients
    g_exact = extract_critic_flat_grads(jax.grad(loss_exact)(params))
    g_exp_corr = extract_critic_flat_grads(jax.grad(loss_exp_corr)(params))
    g_exp_uncorr = extract_critic_flat_grads(jax.grad(loss_exp_uncorr)(params))
    g_samp_corr = extract_critic_flat_grads(jax.grad(loss_samp_corr)(params))
    g_samp_uncorr = extract_critic_flat_grads(jax.grad(loss_samp_uncorr)(params))

    # Gradient directional alignments and norms
    rho_samp_corr_exact = cos_similarity(g_samp_corr, g_exact)
    rho_samp_uncorr_exact = cos_similarity(g_samp_uncorr, g_exact)
    rho_exp_corr_exact = cos_similarity(g_exp_corr, g_exact)
    rho_exp_uncorr_exact = cos_similarity(g_exp_uncorr, g_exact)

    norm_g_exact = jnp.linalg.norm(g_exact)
    norm_g_samp_corr = jnp.linalg.norm(g_samp_corr)
    norm_g_samp_uncorr = jnp.linalg.norm(g_samp_uncorr)
    norm_g_exp_uncorr = jnp.linalg.norm(g_exp_uncorr)

    return {
        "E_exact": l_exact,
        "E_exp_corr": l_exp_corr,
        "E_exp_uncorr": l_exp_uncorr,
        "E_deficit": l_deficit,
        "E_samp_corr": l_samp_corr,
        "E_samp_uncorr": l_samp_uncorr,
        "rho_samp_corr_exact": rho_samp_corr_exact,
        "rho_samp_uncorr_exact": rho_samp_uncorr_exact,
        "rho_exp_corr_exact": rho_exp_corr_exact,
        "rho_exp_uncorr_exact": rho_exp_uncorr_exact,
        "norm_g_exact": norm_g_exact,
        "norm_g_samp_corr": norm_g_samp_corr,
        "norm_g_samp_uncorr": norm_g_samp_uncorr,
        "norm_g_exp_uncorr": norm_g_exp_uncorr,
    }
