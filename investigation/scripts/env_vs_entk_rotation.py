"""
Compares Intrinsic Env Rotation (max_m |beta_m(A)| / alpha_m(A)) against
Effective eNTK Rotation (max_m |beta_m(Theta A)| / alpha_m(Theta A)) across
three representations per environment (at random initialization, no training):

  1. 1-Hot + MLP
  2. Low-Dim Coordinate (x, y) / (pos, vel) + MLP   (when defined)
  3. Grid CNN                                       (2D Conv / 1D Conv)

Runs on CPU from scratch and saves investigation/figures/env_vs_entk_rotation.{png,pdf}.
"""

from __future__ import annotations
import os
import sys
import argparse
from types import SimpleNamespace
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

from scripts.exact_random_policy_sweep.run_single_env import build_env_config
from core import helpers
from investigation.scripts.chain_demo_1 import build_chain_mrp


class MLPNet(nn.Module):
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
        x = nn.tanh(x)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x).squeeze(-1)


class GridCNN2D(nn.Module):
    features: int = 8
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                    kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
        x = nn.tanh(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
        x = nn.tanh(x)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x).squeeze(-1)


class GridCNN1D(nn.Module):
    features: int = 8
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(5,), padding="SAME",
                    kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
        x = nn.tanh(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
        x = nn.tanh(x)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x).squeeze(-1)


def extract_grid_xy(obs_stack: np.ndarray) -> np.ndarray:
    """Extracts normalized (y, x) in [0, 1]^2 from channel 1 of (N, H, W, 2) grid obs."""
    n, h, w, _ = obs_stack.shape
    ch1 = obs_stack[..., 1].reshape(n, h * w)
    idx = np.argmax(ch1, axis=1)
    ys = (idx // w) / max(h - 1, 1)
    xs = (idx % w) / max(w - 1, 1)
    return np.stack([ys, xs], axis=1).astype(np.float32)


def mountaincar_to_grid(obs_stack: np.ndarray, bins: int = 16) -> np.ndarray:
    """Converts discretized (pos, vel) obs_stack (N, 2) to a (N, bins, bins, 1) 1-hot grid."""
    pos_vals = np.unique(np.round(obs_stack[:, 0], 6))
    vel_vals = np.unique(np.round(obs_stack[:, 1], 6))
    p_idx = (np.searchsorted(pos_vals, np.round(obs_stack[:, 0], 6)) * bins) // len(pos_vals)
    v_idx = (np.searchsorted(vel_vals, np.round(obs_stack[:, 1], 6)) * bins) // len(vel_vals)
    grid = np.zeros((len(obs_stack), bins, bins, 1), dtype=np.float32)
    grid[np.arange(len(obs_stack)), p_idx, v_idx, 0] = 1.0
    return grid


def eig_rotation_metric(eigvals: np.ndarray, metric: str = "sum") -> float:
    active = eigvals[np.abs(eigvals) > 1e-6 * np.max(np.abs(eigvals))]
    if metric == "max":
        return float(np.max(np.abs(active.imag) / np.maximum(active.real, 1e-12)))
    elif metric == "sum":
        return float(np.sum(np.abs(active.imag)) / max(float(np.sum(active.real)), 1e-12))
    raise ValueError(f"Unknown metric '{metric}'")


def compute_entk_metrics(net, obs_arr: np.ndarray, mu: np.ndarray, P_pi: np.ndarray, A: np.ndarray, seeds=(0, 1, 2), metric: str = "sum"):
    obs_j = jnp.asarray(obs_arr, dtype=jnp.float32)
    n_states = obs_arr.shape[0]

    def single_v(p, x):
        return net.apply(p, x[None, ...])[0]

    jac_fn = jax.jit(jax.vmap(jax.jacrev(single_v), in_axes=(None, 0)))

    ntk_ratios = []
    rho_cs = []
    for s in seeds:
        params = net.init(jax.random.PRNGKey(s), obs_j[:1])
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
        rho_cs.append(rho_c)

        Theta = J @ J.T
        ev = np.linalg.eigvals(Theta @ A)
        ntk_ratios.append(eig_rotation_metric(ev, metric=metric))

    return float(np.mean(ntk_ratios)), float(np.mean(rho_cs))


def load_env_mdp(env_name: str, gamma: float = 0.99, policy: str = "random", eps: float = 0.05, metric: str = "sum"):
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

    if policy == "random":
        Pi = np.ones((n_states + 1, n_act)) / n_act
    elif policy == "optimal":
        R_sa = np.sum(P * R, axis=2)
        V = np.zeros(P.shape[0])
        for _ in range(1000):
            Q = R_sa + gamma * np.einsum("sam,m->sa", P, V)
            Q[-1, :] = 0.0
            V_new = np.max(Q, axis=1)
            if np.max(np.abs(V_new - V)) < 1e-8:
                break
            V = V_new
        best_a = np.argmax(Q[:n_states], axis=1)
        Pi_active = np.full((n_states, n_act), eps / n_act)
        Pi_active[np.arange(n_states), best_a] += (1.0 - eps)
        Pi = np.vstack([Pi_active, np.ones((1, n_act)) / n_act])
    else:
        raise ValueError(f"Unknown policy '{policy}'")

    P_pi = np.einsum("sa,sam->sm", Pi[: P.shape[0], :], P)[:n_states, :n_states]
    mu_raw, _ = ev.compute_stationary_distribution_raw(jnp.array(Pi[:n_states, :]))
    mu = np.maximum(np.array(mu_raw), 0.0)
    mu /= mu.sum()
    D = np.diag(mu)
    A = D @ (np.eye(n_states) - gamma * P_pi)
    ev_A = np.linalg.eigvals(A)
    state_max_ba = eig_rotation_metric(ev_A, metric=metric)
    return obs_stack, mu, P_pi, A, state_max_ba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "optimal"], default="random")
    parser.add_argument("--metric", choices=["sum", "max"], default="sum")
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
        obs_stack, mu, P_pi, A, state_max_ba = load_env_mdp(env_name, gamma=gamma, policy=cli_args.policy, metric=cli_args.metric)
        n_states = len(obs_stack)

        # 1. 1-Hot + MLP
        onehot_obs = np.eye(n_states, dtype=np.float32)
        ntk_ba, rho_c = compute_entk_metrics(MLPNet(), onehot_obs, mu, P_pi, A, metric=cli_args.metric)
        records.append({"env": label, "rep": "1-Hot + MLP", "state_ba": state_max_ba, "ntk_ba": ntk_ba, "rho_c": rho_c})

        # 2. (x, y) / (pos, vel) + MLP
        if has_xy:
            if "mountaincar" in env_name.lower():
                lo = obs_stack.min(axis=0, keepdims=True)
                hi = obs_stack.max(axis=0, keepdims=True)
                xy_obs = ((obs_stack - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)
            else:
                xy_obs = extract_grid_xy(obs_stack)
            ntk_ba, rho_c = compute_entk_metrics(MLPNet(), xy_obs, mu, P_pi, A, metric=cli_args.metric)
            records.append({"env": label, "rep": "(x, y) + MLP", "state_ba": state_max_ba, "ntk_ba": ntk_ba, "rho_c": rho_c})

        # 3. Grid CNN
        if has_cnn:
            if "mountaincar" in env_name.lower():
                cnn_obs = mountaincar_to_grid(obs_stack)
            else:
                cnn_obs = obs_stack
            ntk_ba, rho_c = compute_entk_metrics(GridCNN2D(), cnn_obs, mu, P_pi, A, metric=cli_args.metric)
            records.append({"env": label, "rep": "Grid CNN", "state_ba": state_max_ba, "ntk_ba": ntk_ba, "rho_c": rho_c})

    # Add 1D Chain (p_fwd = 0.75)
    mrp_chain = build_chain_mrp(N=32, gamma=gamma, p_fwd=0.75, ring=False)
    N = mrp_chain["N"]
    mu_c, P_c = mrp_chain["mu"], mrp_chain["P_ep"]
    A_c = np.diag(mu_c) @ (np.eye(N) - gamma * P_c)
    ev_Ac = np.linalg.eigvals(A_c)
    state_ba_c = eig_rotation_metric(ev_Ac, metric=cli_args.metric)

    for rep_name, net, obs_c in [
        ("1-Hot + MLP", MLPNet(), np.eye(N, dtype=np.float32)),
        ("(x, y) + MLP", MLPNet(), (np.arange(N, dtype=np.float32)[:, None] / (N - 1))),
        ("Grid CNN", GridCNN1D(), np.eye(N, dtype=np.float32)[:, :, None]),
    ]:
        ntk_ba, rho_c = compute_entk_metrics(net, obs_c, mu_c, P_c, A_c, metric=cli_args.metric)
        records.append({"env": "Chain (p_fwd=0.75)", "rep": rep_name, "state_ba": state_ba_c, "ntk_ba": ntk_ba, "rho_c": rho_c})

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    # Plot: Intrinsic Env Rotation vs Effective eNTK Rotation
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 13})
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)

    markers = {"1-Hot + MLP": "o", "(x, y) + MLP": "^", "Grid CNN": "s"}
    for rep, mk in markers.items():
        sub = df[df["rep"] == rep]
        sc = ax.scatter(
            sub["state_ba"],
            sub["ntk_ba"],
            c=sub["rho_c"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            marker=mk,
            s=125,
            edgecolors="#111111",
            linewidths=1.1,
            label=rep,
            zorder=4,
        )

    # Connect representations of the same environment vertically
    for env_lbl, grp in df.groupby("env"):
        if grp["state_ba"].iloc[0] > 0.1 or grp["ntk_ba"].max() > 0.1:
            ax.plot(
                [grp["state_ba"].iloc[0]] * len(grp),
                grp["ntk_ba"].values,
                color="#999999",
                ls=":",
                lw=1.1,
                zorder=2,
            )

    ax.plot([0, 1.55], [0, 1.55], color="#c0392b", ls="--", lw=1.2, label="$y = x$ (no amplification)", zorder=2)

    # Clean annotations
    xmax = float(df["state_ba"].max()) * 1.18 if float(df["state_ba"].max()) > 0 else 1.0
    ymax = float(df["ntk_ba"].max()) * 1.18 if float(df["ntk_ba"].max()) > 0 else 1.0
    if cli_args.metric == "max":
        xmax = max(1.62, float(df["state_ba"].max()) + 0.22)
        ymax = max(3.60, float(df["ntk_ba"].max()) + 0.35)
        if cli_args.policy == "random":
            ax.annotate(
                "FourRooms, FourRooms+Mines,\nEightRooms (all reps) &\nChain ((x) + MLP): all near 0.00",
                xy=(0.0, 0.04),
                xytext=(0.14, 0.62),
                fontsize=8.6,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
                zorder=5,
            )
            label_specs = [
                ("Space Invaders", "Grid CNN", (-0.05, 0.10, "right")),
                ("Space Invaders", "1-Hot + MLP", (-0.05, -0.04, "right")),
                ("Whirlpool", "(x, y) + MLP", (-0.06, 0.08, "right")),
                ("Whirlpool", "1-Hot + MLP", (-0.12, -0.32, "right")),
                ("Whirlpool", "Grid CNN", (-0.12, -0.60, "right")),
                ("MountainCar", "(x, y) + MLP", (-0.05, 0.02, "right")),
                ("MountainCar", "1-Hot + MLP", (-0.05, 0.50, "right")),
                ("MountainCar", "Grid CNN", (0.04, -0.32, "left")),
            ]
        else:
            ax.annotate(
                "FourRooms, FourRooms+Mines,\nEightRooms & Chain ((x, y) + MLP):\nall stay at 0.00",
                xy=(0.02, 0.01),
                xytext=(0.20, 0.28),
                fontsize=8.3,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
                zorder=5,
            )
            ax.annotate(
                "FourRooms, Mines, EightRooms:\n1-Hot MLP & Grid CNN (0.77–1.18)",
                xy=(0.04, 1.05),
                xytext=(0.20, 1.15),
                fontsize=8.3,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
                zorder=5,
            )
            label_specs = [
                ("Space Invaders", "1-Hot + MLP", (-0.06, 0.25, "right")),
                ("Space Invaders", "Grid CNN", (-0.06, -0.10, "right")),
                ("Whirlpool", "1-Hot + MLP", (0.06, 0.14, "left")),
                ("Whirlpool", "Grid CNN", (0.06, 0.02, "left")),
                ("Whirlpool", "(x, y) + MLP", (0.06, -0.14, "left")),
                ("MountainCar", "1-Hot + MLP", (-0.06, 0.35, "right")),
                ("MountainCar", "(x, y) + MLP", (-0.06, 0.05, "right")),
                ("MountainCar", "Grid CNN", (-0.06, -0.18, "right")),
            ]

        ax.annotate(
            "Chain (p_fwd=0.75):\n1-Hot MLP (2.65) & 1D CNN (2.59)",
            xy=(0.0, 2.62),
            xytext=(0.18, 2.65),
            fontsize=8.6,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
            zorder=5,
        )
        for env_l, rep_l, (dx, dy, ha) in label_specs:
            r = df[(df["env"] == env_l) & (df["rep"] == rep_l)].iloc[0]
            x, y = r["state_ba"], r["ntk_ba"]
            disp_rep = "(pos, vel) + MLP" if (env_l == "MountainCar" and rep_l == "(x, y) + MLP") else rep_l
            txt = f"{env_l} ({disp_rep})"
            if abs(dx) > 0.08 or abs(dy) > 0.15:
                ax.annotate(
                    txt, xy=(x, y), xytext=(x + dx, y + dy), fontsize=8.2, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkA=0, shrinkB=4), zorder=5,
                )
            else:
                ax.text(x + dx, y + dy, txt, fontsize=8.2, ha=ha, va="center", zorder=5)
    else:
        if cli_args.policy == "random":
            xmax, ymax = 0.50, 0.48
            ax.annotate(
                "FourRooms, FourRooms+Mines,\nEightRooms (all reps) &\nChain ((x) + MLP): near 0.00",
                xy=(0.0, 0.005),
                xytext=(0.07, 0.045),
                fontsize=8.2,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
                zorder=5,
            )
            sum_specs = [
                ("Space Invaders", "Grid CNN", (0.015, 0.01, "left")),
                ("Space Invaders", "1-Hot + MLP", (0.015, 0.00, "left")),
                ("Chain (p_fwd=0.75)", "1-Hot + MLP", (0.015, 0.01, "left")),
                ("Chain (p_fwd=0.75)", "Grid CNN", (0.015, -0.01, "left")),
                ("MountainCar", "1-Hot + MLP", (-0.015, 0.025, "right")),
                ("MountainCar", "(x, y) + MLP", (-0.015, -0.015, "right")),
                ("MountainCar", "Grid CNN", (-0.015, 0.00, "right")),
                ("Whirlpool", "1-Hot + MLP", (0.015, 0.00, "left")),
                ("Whirlpool", "(x, y) + MLP", (0.015, 0.00, "left")),
                ("Whirlpool", "Grid CNN", (0.015, 0.00, "left")),
            ]
        else:
            xmax, ymax = 0.60, 0.58
            ax.annotate(
                "FourRooms, Mines, EightRooms &\nChain ((x, y) + MLP): all at 0.00",
                xy=(0.001, 0.002),
                xytext=(0.055, 0.015),
                fontsize=8.1,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", alpha=0.92),
                zorder=5,
            )
            ax.annotate(
                "FourRooms, Mines, EightRooms:\nGrid CNN (0.06–0.08)",
                xy=(0.001, 0.07),
                xytext=(0.055, 0.115),
                fontsize=8.1,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", alpha=0.92),
                zorder=5,
            )
            ax.annotate(
                "FourRooms, Mines, EightRooms & Chain:\n1-Hot MLP (0.29–0.31) & 1D CNN (0.27)",
                xy=(0.001, 0.295),
                xytext=(0.055, 0.365),
                fontsize=8.1,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", alpha=0.92),
                zorder=5,
            )
            sum_specs = [
                ("MountainCar", "1-Hot + MLP", (0.012, 0.015, "left")),
                ("MountainCar", "Grid CNN", (-0.012, 0.00, "right")),
                ("MountainCar", "(x, y) + MLP", (0.012, -0.025, "left")),
                ("Whirlpool", "1-Hot + MLP", (0.012, 0.01, "left")),
                ("Whirlpool", "Grid CNN", (0.012, 0.00, "left")),
                ("Whirlpool", "(x, y) + MLP", (0.012, 0.00, "left")),
                ("Space Invaders", "1-Hot + MLP", (-0.012, 0.01, "right")),
                ("Space Invaders", "Grid CNN", (-0.012, -0.005, "right")),
            ]

        for env_l, rep_l, (dx, dy, ha) in sum_specs:
            r = df[(df["env"] == env_l) & (df["rep"] == rep_l)].iloc[0]
            x, y = r["state_ba"], r["ntk_ba"]
            disp_rep = "(pos, vel) + MLP" if (env_l == "MountainCar" and rep_l == "(x, y) + MLP") else rep_l
            ax.text(x + dx, y + dy, f"{env_l} ({disp_rep})", fontsize=8.0, ha=ha, va="center", zorder=5)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Centered Next-State Gradient Similarity $\\rho_c(\\nabla v(s), \\nabla v(s'))$", fontweight="bold")

    ax.plot([0, xmax], [0, xmax], color="#c0392b", ls="--", lw=1.2, zorder=2)
    ax.set_xlim(-0.06 * xmax, xmax)
    ax.set_ylim(-0.06 * ymax, ymax)
    if cli_args.metric == "max":
        ax.set_xlabel("Intrinsic Env Rotation: $\\max_m \\,|\\beta_m(A)| / \\alpha_m(A)$", fontweight="bold")
        ax.set_ylabel("Effective eNTK Rotation: $\\max_m \\,|\\beta_m(\\Theta A)| / \\alpha_m(\\Theta A)$", fontweight="bold")
        ax.legend(loc="upper left", fontsize=9.2, framealpha=0.94)
    else:
        ax.set_xlabel("Intrinsic Env Rotation: $\\sum_m |\\beta_m(A)| / \\sum_m \\alpha_m(A)$", fontweight="bold")
        ax.set_ylabel("Effective eNTK Rotation: $\\sum_m |\\beta_m(\\Theta A)| / \\sum_m \\alpha_m(\\Theta A)$", fontweight="bold")
        leg_loc = "lower right" if cli_args.policy == "random" else "upper left"
        ax.legend(loc=leg_loc, fontsize=9.2, framealpha=0.94)
    pol_title = "Uniform Random Policy" if cli_args.policy == "random" else "Optimal Policy ($\\epsilon=0.05$)"
    ax.set_title(f"Intrinsic Env Rotation vs. Effective eNTK Rotation ({pol_title})", fontweight="bold", pad=10)
    ax.grid(True, alpha=0.28, lw=0.7)

    fig_dir = os.path.join(REPO_ROOT, "investigation", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    pol_suf = "" if cli_args.policy == "random" else "_optimal"
    met_suf = "" if cli_args.metric == "max" else "_sum"
    png_path = os.path.join(fig_dir, f"env_vs_entk_rotation{pol_suf}{met_suf}.png")
    pdf_path = os.path.join(fig_dir, f"env_vs_entk_rotation{pol_suf}{met_suf}.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"\nSaved {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
