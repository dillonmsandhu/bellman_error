"""
scripts/sampling_E_test/verify_e_lambda_matrix.py

Verification script for Sampled E(lambda) against the exact compound matrix S_lambda:
S_lambda = 0.5 * (A_lambda + A_lambda.T), where A_lambda = D (I - gamma P) (I - gamma lambda P)^{-1}.

Evaluates across lambda in [0.0, 0.5, 0.8, 0.95]:
- Exact Matrix E(lambda) loss and gradient norm
- Sampled gradient mean cosine similarity to exact gradient
- Gradient MSE to exact gradient
- Gradient empirical trace variance across independent batch draws
"""

from __future__ import annotations
import os
import sys
import argparse
from typing import Dict, Any
from functools import partial

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from core.config import config as base_cfg
from core.helpers import make_env, create_evaluator, e_lambda_geometric_critic_loss
from core.networks import initialize_network, initialize_flax_train_state
from core.gradient_tracking import extract_critic_flat_grads, cos_similarity


def run_verification(
    env_name: str = "FourRooms-misc",
    lambdas: list[float] = [0.0, 0.5, 0.8, 0.95],
    num_batches: int = 50,
    num_trajectories: int = 64,
    num_steps: int = 128,
    seed: int = 42,
):
    print(f"=== Running Sampled E(lambda) Verification on {env_name} ===")
    cfg = base_cfg.copy()
    cfg["ENV_NAME"] = env_name
    env, env_params = make_env(cfg)
    evaluator = create_evaluator(cfg, env, env_params)

    S_states = evaluator.obs_stack
    N = evaluator.num_states
    start_idx = evaluator.start_idx
    P = evaluator.P
    gamma = 0.99

    # Policy and stationary occupancy
    old_pi = jnp.ones((N, 4)) / 4.0
    terminal_policy = jnp.ones([1, 4]) / 4.0
    old_pi_full = jnp.vstack([old_pi, terminal_policy])

    mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
    mu = jnp.append(mu, 0.0)
    D = jnp.diag(mu)
    P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
    I = jnp.eye(N + 1)
    V_true = evaluator.compute_true_values_raw(old_pi_full)

    # Initialize main critic network
    rng = jax.random.PRNGKey(seed)
    rng, rng_net = jax.random.split(rng)
    obs_shape = evaluator.obs_stack.shape[1:]
    network, params = initialize_network(
        rng_net, obs_shape, env, env_params, 16, n_heads=1, layer_norm=False
    )

    # Initialize auxiliary critic network v_aux
    rng, rng_aux = jax.random.split(rng)
    aux_network, aux_params = initialize_network(
        rng_aux, obs_shape, env, env_params, 16, n_heads=1, layer_norm=False
    )
    aux_cfg = cfg.copy()
    aux_cfg["NUM_UPDATES"] = num_batches * 10
    aux_train_state = initialize_flax_train_state(aux_cfg, aux_network, aux_params)

    def get_e(p):
        v = network.apply(p, S_states, method=network.value).squeeze()
        return V_true - jnp.append(v, 0.0)

    # Rollout simulator
    mu_trans = mu[:N] / jnp.maximum(jnp.sum(mu[:N]), 1e-12)
    v_cur = network.apply(params, S_states, method=network.value).squeeze()
    v_full_cur = jnp.append(v_cur, 0.0)

    @jax.jit
    def sample_rollout(roll_rng):
        r_init, r_steps = jax.random.split(roll_rng)
        s0 = jax.random.choice(r_init, N, shape=(num_trajectories,), p=mu_trans)

        def step_fn(carry_s, step_rng):
            rng_a, rng_s = jax.random.split(step_rng)
            step_rngs_a = jax.random.split(rng_a, num_trajectories)
            step_rngs_s = jax.random.split(rng_s, num_trajectories)
            logits_pi = jnp.log(jnp.maximum(old_pi_full[carry_s], 1e-12))
            actions = jax.vmap(jax.random.categorical)(step_rngs_a, logits_pi)
            P_sa = P[carry_s, actions]
            logits_p = jnp.log(jnp.maximum(P_sa, 1e-12))
            next_s_raw = jax.vmap(jax.random.categorical)(step_rngs_s, logits_p)
            rewards = evaluator.R[carry_s, actions, next_s_raw]
            dones = (next_s_raw == N)
            next_s_carry = jnp.where(dones, start_idx, next_s_raw)
            return next_s_carry, (carry_s, next_s_raw, rewards, dones)

        step_rngs = jax.random.split(r_steps, num_steps)
        final_s, (traj_s, traj_next_s, traj_r, traj_done) = jax.lax.scan(step_fn, s0, step_rngs)
        return traj_s, traj_next_s, traj_r, traj_done, final_s

    @jax.jit
    def compute_returns(traj_r, traj_done, final_s):
        v_final = v_full_cur[final_s]

        def rev_scan_fn(g_next, step_data):
            r_t, d_t = step_data
            g_cur = r_t + gamma * jnp.where(d_t, 0.0, g_next)
            return g_cur, g_cur

        _, returns = jax.lax.scan(rev_scan_fn, v_final, (traj_r, traj_done), reverse=True)
        return returns

    # Pre-train aux network on Monte Carlo rollout returns
    print("Pre-training auxiliary critic on Monte Carlo rollout returns...")
    pretrain_keys = jax.random.split(rng, 20)
    for pk in pretrain_keys:
        traj_s, _, traj_r, traj_done, final_s = sample_rollout(pk)
        returns = compute_returns(traj_r, traj_done, final_s)

        def aux_step(ts, _):
            def loss(p_a):
                v_pred = aux_network.apply(p_a, S_states).squeeze()
                v_full = jnp.append(v_pred, 0.0)
                return jnp.mean((returns - v_full[traj_s]) ** 2)
            grads = jax.grad(loss)(ts.params)
            return ts.apply_gradients(grads=grads), None

        aux_train_state, _ = jax.lax.scan(aux_step, aux_train_state, None, length=5)

    @partial(jax.jit, static_argnums=(6,))
    def compute_sample_grad_jit(p, p_aux, traj_s, returns, traj_done, r_jump, lmbda):
        v_aux_all = aux_network.apply(p_aux, S_states, method=aux_network.value).squeeze()
        v_aux_full = jnp.append(v_aux_all, 0.0)
        aux_values = v_aux_full[traj_s]

        def loss(p_curr):
            v_a = network.apply(p_curr, S_states, method=network.value).squeeze()
            v_f = jnp.append(v_a, 0.0)
            vals = v_f[traj_s]
            val_loss, _, _ = e_lambda_geometric_critic_loss(
                values=vals,
                targets=returns,
                dones=traj_done,
                gamma=gamma,
                lmbda=lmbda,
                rng=r_jump,
                aux_values=aux_values,
                true_terminals=traj_done,
                return_details=False,
            )
            return val_loss

        grads = jax.grad(loss)(p)
        return extract_critic_flat_grads(grads)

    results = []

    for lmbda in lambdas:
        print(f"\n--- Evaluating lambda = {lmbda} ---")
        # 1. Exact Compound Matrix S_lambda
        L_lmbda = jnp.linalg.inv(I - gamma * lmbda * P_pi)
        A_lmbda = D @ (I - gamma * P_pi) @ L_lmbda
        S_lmbda = 0.5 * (A_lmbda + A_lmbda.T)

        def loss_exact(p):
            e = get_e(p)
            return 0.5 * (e.T @ S_lmbda @ e)

        g_exact = extract_critic_flat_grads(jax.grad(loss_exact)(params))
        g_exact_norm = float(jnp.linalg.norm(g_exact))
        g_exact_np = np.array(g_exact)

        # 2. Sample across num_batches independent rollout draws
        sample_grads = []
        batch_keys = jax.random.split(rng, num_batches)

        for bk in batch_keys:
            r_roll, r_jump = jax.random.split(bk)
            traj_s, traj_next_s, traj_r, traj_done, final_s = sample_rollout(r_roll)
            returns = compute_returns(traj_r, traj_done, final_s)

            g_sample = compute_sample_grad_jit(
                params, aux_train_state.params, traj_s, returns, traj_done, r_jump, lmbda
            )
            sample_grads.append(np.array(g_sample))

        sample_grads = np.stack(sample_grads, axis=0)
        mean_g = np.mean(sample_grads, axis=0)
        cov = np.cov(sample_grads, rowvar=False)
        total_var = float(np.trace(cov)) if cov.ndim == 2 else float(cov)

        cos_sim = float(cos_similarity(mean_g, g_exact_np))
        mse_err = float(np.mean((mean_g - g_exact_np) ** 2))
        norm_ratio = float(np.linalg.norm(mean_g)) / max(1e-8, g_exact_norm)

        print(f"Cosine Similarity to Exact: {cos_sim:+.4f}")
        print(f"Gradient MSE:               {mse_err:.6e}")
        print(f"Empirical Total Variance:   {total_var:.6e}")
        print(f"Norm Ratio (sample / exact): {norm_ratio:.4f}")

        results.append({
            "lambda": lmbda,
            "cos_sim": cos_sim,
            "mse": mse_err,
            "total_variance": total_var,
            "norm_ratio": norm_ratio,
            "exact_norm": g_exact_norm,
        })

    df = pd.DataFrame(results)
    print("\n=== Verification Summary Table ===")
    print(df.to_string(index=False))

    out_csv = os.path.join(REPO_ROOT, "results", "sampling_E_test", "verify_e_lambda_matrix.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="FourRooms-misc")
    parser.add_argument("--num_batches", type=int, default=50)
    args = parser.parse_args()
    run_verification(env_name=args.env_name, num_batches=args.num_batches)
