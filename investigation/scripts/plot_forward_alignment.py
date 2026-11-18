"""
Plots Forward-Only Gradient Alignment:
    cos(nabla_theta E(0), nabla_theta E_Fwd(0))
against Centered Next-State Gradient Similarity rho_c(grad v(s), grad v(s'))
across environments and representations at random initialization (5 seeds).

Shows directly why:
  - Low-rho_c envs (Atari/Space Invaders, 1-Hot MLP) have high Forward-Only alignment
    (so TD / Forward-Only already tracks nabla E without Term (1)'s sampling noise).
  - High-rho_c envs (MountainCar (pos, vel), Chain CNN+AvgPool) lose Forward-Only
    alignment because Term (1) + Term (2) cancellation is destroyed.

Saves investigation/figures/forward_only_alignment.{png,pdf}.
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from investigation.scripts.env_vs_entk_rotation import (
    MLPNet,
    GridCNN2D,
    GridCNN1D,
    extract_grid_xy,
    mountaincar_to_grid,
    load_env_mdp,
)
from investigation.scripts.chain_demo_1 import build_chain_mrp, ChainCNN


def load_env_with_V(env_name: str, gamma: float = 0.99, policy: str = "random", eps: float = 0.05):
    from types import SimpleNamespace
    from scripts.exact_random_policy_sweep.run_single_env import build_env_config
    from core import helpers

    args = SimpleNamespace(
        seed=42, gamma=gamma, k_values=[64], lrs=[0.001],
        num_updates=10, n_seeds=1, value_lambda=0.8,
    )
    cfg = build_env_config(env_name, args)
    env, env_params = helpers.make_env(cfg)
    ev = helpers.initialize_evaluator(cfg, env, env_params)
    obs_stack = np.asarray(ev.obs_stack, dtype=np.float32)
    n_states = len(obs_stack)
    n_act = env.action_space(env_params).n
    P = np.array(ev.P)
    R = np.array(ev.R)
    R_sa = np.sum(P * R, axis=2)

    if policy == "random":
        Pi = np.ones((n_states + 1, n_act)) / n_act
    else:
        V_tmp = np.zeros(P.shape[0])
        for _ in range(1000):
            Q = R_sa + gamma * np.einsum("sam,m->sa", P, V_tmp)
            Q[-1, :] = 0.0
            V_new = np.max(Q, axis=1)
            if np.max(np.abs(V_new - V_tmp)) < 1e-8:
                break
            V_tmp = V_new
        best_a = np.argmax(Q[:n_states], axis=1)
        Pi_active = np.full((n_states, n_act), eps / n_act)
        Pi_active[np.arange(n_states), best_a] += (1.0 - eps)
        Pi = np.vstack([Pi_active, np.ones((1, n_act)) / n_act])

    P_pi = np.einsum("sa,sam->sm", Pi[: P.shape[0], :], P)[:n_states, :n_states]
    R_pi = np.einsum("sa,sa->s", Pi[:n_states, :], R_sa[:n_states, :])
    V = np.linalg.solve(np.eye(n_states) - gamma * P_pi, R_pi)

    mu_raw, _ = ev.compute_stationary_distribution_raw(jnp.array(Pi[:n_states, :]))
    mu = np.maximum(np.array(mu_raw), 0.0)
    mu /= mu.sum()
    return obs_stack, mu, P_pi, V


def compute_alignment_metrics(net, obs_arr: np.ndarray, mu: np.ndarray, P_pi: np.ndarray, V: np.ndarray, gamma: float = 0.99, seeds=(0, 1, 2, 3, 4)):
    obs_j = jnp.asarray(obs_arr, dtype=jnp.float32)
    n_states = obs_arr.shape[0]
    D = np.diag(mu)
    A = D @ (np.eye(n_states) - gamma * P_pi)
    S = 0.5 * (A + A.T)
    A_fwd = 0.5 * (A + (1.0 - gamma) * D)

    def single_v(p, x):
        return net.apply(p, x[None, ...])[0]

    jac_fn = jax.jit(jax.vmap(jax.jacrev(single_v), in_axes=(None, 0)))

    rhos, cos_fwds, cos_tds = [], [], []
    for s in seeds:
        params = net.init(jax.random.PRNGKey(s), obs_j[:1])
        v0 = np.asarray(net.apply(params, obs_j))
        e_raw = V - v0
        # Center out the trivial constant offset E_mu[e] (learned in the first few steps)
        # so alignment measures relative value-shape / advantage learning.
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

        g_E = J.T @ (S @ e)
        g_Fwd = J.T @ (A_fwd @ e)
        g_TD = J.T @ (A @ e)

        nE = max(float(np.linalg.norm(g_E)), 1e-12)
        nF = max(float(np.linalg.norm(g_Fwd)), 1e-12)
        nT = max(float(np.linalg.norm(g_TD)), 1e-12)

        rhos.append(rho_c)
        cos_fwds.append(float((g_E @ g_Fwd) / (nE * nF)))
        cos_tds.append(float((g_E @ g_TD) / (nE * nT)))

    return float(np.mean(rhos)), float(np.mean(cos_fwds)), float(np.mean(cos_tds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "optimal"], default="random")
    cli_args = parser.parse_args()
    gamma = 0.99

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
        obs_stack, mu, P_pi, V = load_env_with_V(env_name, gamma=gamma, policy=cli_args.policy)
        n_states = len(obs_stack)

        rho, cf, ct = compute_alignment_metrics(MLPNet(), np.eye(n_states, dtype=np.float32), mu, P_pi, V, gamma)
        records.append({"env": label, "rep": "1-Hot + MLP", "rho_c": rho, "cos_fwd": cf, "cos_td": ct})

        if has_xy:
            if "mountaincar" in env_name.lower():
                lo, hi = obs_stack.min(axis=0, keepdims=True), obs_stack.max(axis=0, keepdims=True)
                xy_obs = ((obs_stack - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)
            else:
                xy_obs = extract_grid_xy(obs_stack)
            rho, cf, ct = compute_alignment_metrics(MLPNet(), xy_obs, mu, P_pi, V, gamma)
            records.append({"env": label, "rep": "(x, y) + MLP", "rho_c": rho, "cos_fwd": cf, "cos_td": ct})

        if has_cnn:
            cnn_obs = mountaincar_to_grid(obs_stack) if "mountaincar" in env_name.lower() else obs_stack
            rho, cf, ct = compute_alignment_metrics(GridCNN2D(), cnn_obs, mu, P_pi, V, gamma)
            records.append({"env": label, "rep": "Grid CNN", "rho_c": rho, "cos_fwd": cf, "cos_td": ct})

    # Add Chain (p_fwd = 0.75) including CNN (AvgPool)
    mrp_c = build_chain_mrp(N=32, gamma=gamma, p_fwd=0.75, ring=False)
    N = mrp_c["N"]
    for rep_name, net, obs_c in [
        ("1-Hot + MLP", MLPNet(), np.eye(N, dtype=np.float32)),
        ("(x, y) + MLP", MLPNet(), np.arange(N, dtype=np.float32)[:, None] / (N - 1)),
        ("Grid CNN", GridCNN1D(), np.eye(N, dtype=np.float32)[:, :, None]),
        ("CNN (AvgPool)", ChainCNN(use_avg_pool=True), np.eye(N, dtype=np.float32)[:, :, None]),
    ]:
        rho, cf, ct = compute_alignment_metrics(net, obs_c, mrp_c["mu"], mrp_c["P_ep"], mrp_c["V"], gamma)
        records.append({"env": "Chain (p_fwd=0.75)", "rep": rep_name, "rho_c": rho, "cos_fwd": cf, "cos_td": ct})

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 13})
    fig, ax = plt.subplots(figsize=(9.6, 6.0), constrained_layout=True)

    rep_styles = {
        "1-Hot + MLP": ("o", "#1f77b4"),
        "(x, y) + MLP": ("^", "#e67e22"),
        "Grid CNN": ("s", "#2ca02c"),
        "CNN (AvgPool)": ("D", "#7b2cbf"),
    }
    for rep, (mk, col) in rep_styles.items():
        sub = df[df["rep"] == rep]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["rho_c"], sub["cos_fwd"],
            marker=mk, color=col, s=120, edgecolors="#111111", linewidths=1.0,
            label=rep, zorder=4,
        )

    # Highlight key points without clutter
    for _, r in df.iterrows():
        env_l, rep_l = r["env"], r["rep"]
        x, y = r["rho_c"], r["cos_fwd"]
        disp_rep = "(pos, vel)" if (env_l == "MountainCar" and rep_l == "(x, y) + MLP") else rep_l
        if env_l in ["Space Invaders", "MountainCar", "Whirlpool", "Chain (p_fwd=0.75)"]:
            dx, dy, ha = 0.02, 0.00, "left"
            if x > 0.72:
                dx, ha = -0.02, "right"
            if cli_args.policy == "random":
                if env_l == "Chain (p_fwd=0.75)" and rep_l == "Grid CNN":
                    dx, dy = 0.02, 0.03
                elif env_l == "Chain (p_fwd=0.75)" and rep_l == "1-Hot + MLP":
                    dx, dy = 0.02, -0.035
                elif env_l == "Whirlpool" and rep_l == "1-Hot + MLP":
                    dx, dy = 0.02, -0.035
                elif env_l == "Whirlpool" and rep_l == "Grid CNN":
                    dx, dy = 0.02, -0.02
                elif env_l == "Space Invaders" and rep_l == "Grid CNN":
                    dx, dy = 0.02, 0.02
                elif env_l == "Space Invaders" and rep_l == "1-Hot + MLP":
                    dx, dy = 0.02, 0.00
                elif env_l == "MountainCar" and rep_l == "1-Hot + MLP":
                    dx, dy = 0.02, 0.03
                elif env_l == "MountainCar" and rep_l == "Grid CNN":
                    dx, dy, ha = -0.02, -0.03, "right"
                elif env_l == "MountainCar" and rep_l == "(x, y) + MLP":
                    dx, dy, ha = -0.02, 0.00, "right"
                elif env_l == "Whirlpool" and rep_l == "(x, y) + MLP":
                    dx, dy, ha = -0.02, 0.00, "right"
            else:
                if env_l == "Chain (p_fwd=0.75)" and rep_l == "Grid CNN":
                    dx, dy = 0.02, 0.03
                elif env_l == "Chain (p_fwd=0.75)" and rep_l == "1-Hot + MLP":
                    dx, dy = 0.02, -0.03
                elif env_l == "MountainCar" and rep_l == "1-Hot + MLP":
                    dx, dy = 0.02, 0.00
                elif env_l == "Chain (p_fwd=0.75)" and rep_l == "CNN (AvgPool)":
                    dx, dy, ha = -0.02, 0.03, "right"
                elif env_l == "Whirlpool" and rep_l == "(x, y) + MLP":
                    dx, dy, ha = -0.02, 0.03, "right"
                elif env_l == "MountainCar" and rep_l == "(x, y) + MLP":
                    dx, dy, ha = -0.02, -0.03, "right"
                elif env_l == "MountainCar" and rep_l == "Grid CNN":
                    dx, dy, ha = -0.02, -0.01, "right"
            ax.text(x + dx, y + dy, f"{env_l} ({disp_rep})", fontsize=8.0, ha=ha, va="center", zorder=5)

    ax.axhline(0.0, color="#888888", ls=":", lw=1.0)
    ax.set_xlim(-0.08, 1.06)
    ax.set_ylim(-0.34, 1.06)
    ax.set_xlabel("Centered Next-State Gradient Similarity $\\rho_c(\\nabla v(s), \\nabla v(s'))$", fontweight="bold")
    ax.set_ylabel("Forward-Only Alignment: $\\cos(\\nabla_\\theta E, \\nabla_\\theta E_{\\mathrm{Fwd}})$", fontweight="bold")
    pol_title = "Uniform Random Policy" if cli_args.policy == "random" else "Optimal Policy ($\\epsilon=0.05$)"
    ax.set_title(f"Forward-Only Gradient Alignment vs. $\\rho_c$ ({pol_title})", fontweight="bold", pad=10)
    ax.grid(True, alpha=0.28, lw=0.7)
    ax.legend(loc="lower left", fontsize=9.2, framealpha=0.94)

    fig_dir = os.path.join(REPO_ROOT, "investigation", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    pol_suf = "" if cli_args.policy == "random" else "_optimal"
    png_path = os.path.join(fig_dir, f"forward_only_alignment{pol_suf}.png")
    pdf_path = os.path.join(fig_dir, f"forward_only_alignment{pol_suf}.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"\nSaved {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
