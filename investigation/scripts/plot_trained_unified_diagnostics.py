"""
Trained-trajectory version of unified_diagnostics_optimal:
Trains each (env, representation) for 100 Adam steps (lr=3e-3, 3 seeds) to fit
the optimal-policy value function V^{pi*} (epsilon=0.05 greedy action noise) via
exact E(0) minimization, and averages:
  - eNTK Spectral Ratio: sum_m |beta_m(Theta A)| / sum_m alpha_m(Theta A)
  - Centered Next-State Gradient Similarity: rho_c(grad v(s), grad v(s'))
  - Forward-Only Alignment: cos(nabla_theta E, nabla_theta E_Fwd) on centered e_t

Saves investigation/figures/unified_diagnostics_optimal_trained.{png,pdf}.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from investigation.scripts.env_vs_entk_rotation import (
    MLPNet,
    GridCNN2D,
    GridCNN1D,
    extract_grid_xy,
    mountaincar_to_grid,
    eig_rotation_metric,
)
from investigation.scripts.plot_forward_alignment import load_env_with_V
from investigation.scripts.plot_unified_diagnostics import make_plot
from investigation.scripts.chain_demo_1 import build_chain_mrp, ChainCNN


def eval_checkpoint(net, params, obs_j, jac_fn, mu, P_pi, A, S, A_fwd, V):
    n_states = obs_j.shape[0]
    v_t = np.asarray(net.apply(params, obs_j))
    e_raw = V - v_t
    e = e_raw - float(np.sum(mu * e_raw))

    chunks = []
    bs = 256
    for start in range(0, n_states, bs):
        end = min(n_states, start + bs)
        J_tree = jac_fn(params, obs_j[start:end])
        J_c = np.concatenate(
            [np.reshape(np.asarray(x), (end - start, -1)) for x in jax.tree_util.tree_leaves(J_tree)],
            axis=1,
        )
        chunks.append(J_c)
    J = np.concatenate(chunks, axis=0)

    mu_col = mu[:, None]
    J_cent = J - np.sum(mu_col * J, axis=0, keepdims=True)
    D_Jc = mu_col * J_cent
    rho_c = float(np.sum(D_Jc * (P_pi @ J_cent)) / max(np.sum(D_Jc * J_cent), 1e-12))

    Theta = J @ J.T
    ev = np.linalg.eigvals(Theta @ A)
    ntk_sum = eig_rotation_metric(ev, metric="sum")

    g_E = J.T @ (S @ e)
    g_Fwd = J.T @ (A_fwd @ e)
    cos_f = float((g_E @ g_Fwd) / (max(float(np.linalg.norm(g_E)), 1e-12) * max(float(np.linalg.norm(g_Fwd)), 1e-12)))
    return ntk_sum, rho_c, cos_f


def train_and_average(
    net,
    obs_arr: np.ndarray,
    mu: np.ndarray,
    P_pi: np.ndarray,
    V: np.ndarray,
    gamma: float = 0.99,
    steps: int = 100,
    eval_every: int = 20,
    lr: float = 3e-3,
    seeds=(0, 1, 2),
):
    obs_j = jnp.asarray(obs_arr, dtype=jnp.float32)
    n_states = obs_arr.shape[0]
    D = np.diag(mu)
    A = D @ (np.eye(n_states) - gamma * P_pi)
    S = 0.5 * (A + A.T)
    A_fwd = 0.5 * (A + (1.0 - gamma) * D)

    S_j = jnp.asarray(S, dtype=jnp.float32)
    V_j = jnp.asarray(V, dtype=jnp.float32)

    tx = optax.adam(learning_rate=lr)

    def loss_fn(p):
        v = net.apply(p, obs_j)
        e = V_j - v
        return 0.5 * e @ S_j @ e

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update_chunk(params, opt_state):
        def step_fn(carry, _):
            p, os_ = carry
            _, g = grad_fn(p)
            updates, os_ = tx.update(g, os_, p)
            p = optax.apply_updates(p, updates)
            return (p, os_), None

        (params, opt_state), _ = jax.lax.scan(step_fn, (params, opt_state), None, length=eval_every)
        return params, opt_state

    def single_v(p, x):
        return net.apply(p, x[None, ...])[0]

    jac_fn = jax.jit(jax.vmap(jax.jacrev(single_v), in_axes=(None, 0)))

    n_chunks = steps // eval_every
    ntk_all, rho_all, cos_all = [], [], []

    for s in seeds:
        params = net.init(jax.random.PRNGKey(s), obs_j[:1])
        opt_state = tx.init(params)

        ntk0, rho0, cos0 = eval_checkpoint(net, params, obs_j, jac_fn, mu, P_pi, A, S, A_fwd, V)
        ntk_all.append(ntk0)
        rho_all.append(rho0)
        cos_all.append(cos0)

        for _ in range(n_chunks):
            params, opt_state = update_chunk(params, opt_state)
            ntk_c, rho_c, cos_c = eval_checkpoint(net, params, obs_j, jac_fn, mu, P_pi, A, S, A_fwd, V)
            ntk_all.append(ntk_c)
            rho_all.append(rho_c)
            cos_all.append(cos_c)

    return float(np.mean(ntk_all)), float(np.mean(rho_all)), float(np.mean(cos_all))


def main():
    t0 = time.time()
    gamma = 0.99
    policy = "optimal"
    records = []

    env_specs = [
        ("FourRooms-misc", "FourRooms", True, True),
        ("fourrooms-mines", "FourRooms + Mines", True, True),
        ("eightrooms-misc", "EightRooms", True, True),
        ("whirlpool-misc", "Whirlpool", True, True),
        ("mountaincar-dense", "MountainCar", True, True),
        ("SpaceInvadersExactValue", "Space Invaders", False, True),
    ]

    for env_name, label, has_xy, has_cnn in env_specs:
        obs_stack, mu, P_pi, V = load_env_with_V(env_name, gamma=gamma, policy=policy)
        n_states = len(obs_stack)
        A = np.diag(mu) @ (np.eye(n_states) - gamma * P_pi)
        env_sum = eig_rotation_metric(np.linalg.eigvals(A), metric="sum")

        ntk_s, rho, cf = train_and_average(MLPNet(), np.eye(n_states, dtype=np.float32), mu, P_pi, V, gamma)
        records.append({"env": label, "rep": "1-Hot + MLP", "env_sum": env_sum, "ntk_sum": ntk_s, "rho_c": rho, "cos_fwd": cf})

        if has_xy:
            if "mountaincar" in env_name.lower():
                lo, hi = obs_stack.min(axis=0, keepdims=True), obs_stack.max(axis=0, keepdims=True)
                xy_obs = ((obs_stack - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)
            else:
                xy_obs = extract_grid_xy(obs_stack)
            ntk_s, rho, cf = train_and_average(MLPNet(), xy_obs, mu, P_pi, V, gamma)
            records.append({"env": label, "rep": "(x, y) + MLP", "env_sum": env_sum, "ntk_sum": ntk_s, "rho_c": rho, "cos_fwd": cf})

        if has_cnn:
            cnn_obs = mountaincar_to_grid(obs_stack) if "mountaincar" in env_name.lower() else obs_stack
            ntk_s, rho, cf = train_and_average(GridCNN2D(), cnn_obs, mu, P_pi, V, gamma)
            records.append({"env": label, "rep": "Grid CNN", "env_sum": env_sum, "ntk_sum": ntk_s, "rho_c": rho, "cos_fwd": cf})

    mrp_c = build_chain_mrp(N=32, gamma=gamma, p_fwd=0.75, ring=False)
    N = mrp_c["N"]
    A_c = np.diag(mrp_c["mu"]) @ (np.eye(N) - gamma * mrp_c["P_ep"])
    env_sum_c = eig_rotation_metric(np.linalg.eigvals(A_c), metric="sum")
    for rep_name, net, obs_c in [
        ("1-Hot + MLP", MLPNet(), np.eye(N, dtype=np.float32)),
        ("(x, y) + MLP", MLPNet(), np.arange(N, dtype=np.float32)[:, None] / (N - 1)),
        ("Grid CNN", GridCNN1D(), np.eye(N, dtype=np.float32)[:, :, None]),
        ("CNN (AvgPool)", ChainCNN(use_avg_pool=True), np.eye(N, dtype=np.float32)[:, :, None]),
    ]:
        ntk_s, rho, cf = train_and_average(net, obs_c, mrp_c["mu"], mrp_c["P_ep"], mrp_c["V"], gamma)
        records.append({"env": "Chain (p_fwd=0.75)", "rep": rep_name, "env_sum": env_sum_c, "ntk_sum": ntk_s, "rho_c": rho, "cos_fwd": cf})

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    make_plot(df, policy="optimal", out_name="unified_diagnostics_optimal_trained", title_suffix=", Avg over 100 Steps")
    print(f"\nSaved unified_diagnostics_optimal_trained.{{png,pdf}} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
