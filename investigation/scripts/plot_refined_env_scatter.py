"""
Refined diagnostic scatter plot for the 9 benchmark environments from
results/exact_random_policy_sweep/, without overwriting existing paper figures.

Computes:
  - State-space rotation: max_m |beta_m(A)| / alpha_m(A)
  - eNTK-space rotation:  max_m |beta_m(Theta A)| / alpha_m(Theta A)
  - Centered next-state gradient cosine similarity: rho_c(grad v(s), grad v(s'))
  - Flipped Y-axis ratio: AUC VE(TD(0)) / AUC VE(E(0)) (higher = E(0) wins by more)

Saves to investigation/figures/refined_env_scatter.{png,pdf}.
"""

from __future__ import annotations
import os
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.exact_random_policy_sweep.run_single_env import build_env_config
from core import helpers, networks


DISPLAY_NAMES = {
    "FourRooms-misc": "FourRooms (Sparse)",
    "fourrooms-dense": "FourRooms (Dense)",
    "fourrooms-mines": "FourRooms + Mines",
    "fourrooms-mines-dense": "FourRooms + Mines (Dense)",
    "eightrooms-misc": "EightRooms (Sparse)",
    "eightrooms-dense": "EightRooms (Dense)",
    "whirlpool-misc": "Whirlpool",
    "mountaincar-dense": "MountainCar (Dense)",
    "SpaceInvadersExactValue": "Space Invaders",
}


def compute_env_and_ntk_metrics(env_name: str, k_val: int = 64, gamma: float = 0.99):
    args = SimpleNamespace(
        seed=42,
        gamma=gamma,
        k_values=[k_val],
        lrs=[0.001],
        num_updates=10,
        n_seeds=1,
        value_lambda=0.8,
    )
    cfg = build_env_config(env_name, args)
    env, env_params = helpers.make_env(cfg)
    ev = helpers.initialize_evaluator(cfg, env, env_params)
    obs_stack = ev.obs_stack
    n_states = len(obs_stack)
    n_act = env.action_space(env_params).n

    Pi = np.ones((n_states + 1, n_act)) / n_act
    P = np.array(ev.P)
    P_pi = np.einsum("sa,sam->sm", Pi[: P.shape[0], :], P)[:n_states, :n_states]
    mu_raw, _ = ev.compute_stationary_distribution_raw(jnp.array(Pi[:n_states, :]))
    mu = np.maximum(np.array(mu_raw), 0.0)
    mu /= mu.sum()

    D = np.diag(mu)
    A = D @ (np.eye(n_states) - gamma * P_pi)
    S = 0.5 * (A + A.T)
    K = 0.5 * (A - A.T)
    ks_fro = float(np.linalg.norm(K, "fro") / (np.linalg.norm(S, "fro") + 1e-12))

    eig_A = np.linalg.eigvals(A)
    state_max_ba = float(np.max(np.abs(eig_A.imag) / np.maximum(eig_A.real, 1e-12)))

    # Network Jacobian J at initialization (exact architecture from exact_random_policy_sweep)
    obs_shape = env.observation_space(env_params).shape
    rng = jax.random.PRNGKey(42)
    network, params = networks.initialize_network(
        rng, obs_shape, env, env_params, k_val, n_heads=1, layer_norm=cfg["LAYER_NORM"]
    )

    def single_v(p, x):
        return network.apply(p, x[None, ...])[0]

    # Compute J in batches to avoid excessive memory on n=1974
    batch_size = 256
    J_chunks = []
    jac_fn = jax.jit(jax.vmap(jax.jacrev(single_v), in_axes=(None, 0)))
    for start in range(0, n_states, batch_size):
        end = min(n_states, start + batch_size)
        J_tree = jac_fn(params, obs_stack[start:end])
        J_chunk = np.concatenate(
            [np.reshape(np.asarray(x), (end - start, -1)) for x in jax.tree_util.tree_leaves(J_tree)],
            axis=1,
        )
        J_chunks.append(J_chunk)
    J = np.concatenate(J_chunks, axis=0)  # (n_states, n_params)

    # Centered next-state gradient similarity rho_c(grad v(s), grad v(s'))
    mu_col = mu[:, None]
    J_c = J - np.sum(mu_col * J, axis=0, keepdims=True)
    D_Jc = mu_col * J_c
    rho_c = float(np.sum(D_Jc * (P_pi @ J_c)) / max(np.sum(D_Jc * J_c), 1e-12))

    # eNTK rotation max_m |beta_m(Theta A)| / alpha_m(Theta A)
    Theta = J @ J.T
    eig_TA = np.linalg.eigvals(Theta @ A)
    active = eig_TA[np.abs(eig_TA) > 1e-5 * np.max(np.abs(eig_TA))]
    ntk_max_ba = float(np.max(np.abs(active.imag) / np.maximum(active.real, 1e-12)))

    return {
        "env": env_name,
        "label": DISPLAY_NAMES[env_name],
        "n_states": n_states,
        "ks_fro": ks_fro,
        "state_max_ba": state_max_ba,
        "ntk_max_ba": ntk_max_ba,
        "rho_c": rho_c,
    }


def main():
    sweep_dir = os.path.join(REPO_ROOT, "results", "exact_random_policy_sweep")
    df64 = pd.read_csv(os.path.join(sweep_dir, "summary_table_k64.csv"))
    df8 = pd.read_csv(os.path.join(sweep_dir, "summary_table_k8.csv"))

    envs = list(DISPLAY_NAMES.keys())
    rows = []
    for env_name in envs:
        m64 = compute_env_and_ntk_metrics(env_name, k_val=64)
        sub64 = df64[df64["env"] == env_name].set_index("algo")
        sub8 = df8[df8["env"] == env_name].set_index("algo")
        m64["ratio_k64"] = float(sub64.loc["e0", "auc_ve_mean"] / sub64.loc["td0", "auc_ve_mean"])
        m64["ratio_k8"] = float(sub8.loc["e0", "auc_ve_mean"] / sub8.loc["td0", "auc_ve_mean"])
        rows.append(m64)
        print(
            f"{m64['label']:26s} | ks_fro={m64['ks_fro']:.3f} | state |b|/a={m64['state_max_ba']:.2f} | "
            f"ntk |b|/a={m64['ntk_max_ba']:.2f} | rho_c={m64['rho_c']:.3f} | E/TD(k=64)={m64['ratio_k64']:.2f}"
        )

    res_df = pd.DataFrame(rows)

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12.0})
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), constrained_layout=True)

    # Panel 1: Intrinsic Env Rotation vs Effective eNTK Rotation
    ax1 = axes[0]
    sc1 = ax1.scatter(
        res_df["state_max_ba"],
        res_df["ntk_max_ba"],
        c=res_df["rho_c"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=115,
        edgecolors="#111111",
        linewidths=1.1,
        zorder=4,
    )
    ax1.plot([0, 1.55], [0, 1.55], color="#666666", ls="--", lw=1.2, label="$y = x$ (no amplification)", zorder=2)
    offsets_p1 = {
        "FourRooms (Sparse)": (0.08, 0.45),
        "FourRooms (Dense)": (0.28, 0.25),
        "FourRooms + Mines": (0.45, 0.12),
        "FourRooms + Mines (Dense)": (0.10, 0.72),
        "EightRooms (Sparse)": (0.35, -0.05),
        "EightRooms (Dense)": (0.55, 0.35),
        "Whirlpool": (-0.06, 0.22),
        "MountainCar (Dense)": (-0.08, -0.15),
        "Space Invaders": (-0.06, 0.22),
    }
    for _, r in res_df.iterrows():
        x, y, lbl = r["state_max_ba"], r["ntk_max_ba"], r["label"]
        dx, dy = offsets_p1[lbl]
        ha = "right" if dx < 0 else "left"
        if abs(dx) > 0.15 or abs(dy) > 0.15:
            ax1.annotate(
                lbl, xy=(x, y), xytext=(x + dx, y + dy), fontsize=8.3, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkA=0, shrinkB=4), zorder=5,
            )
        else:
            ax1.text(x + dx, y + dy, lbl, fontsize=8.3, ha=ha, va="center", zorder=5)
    ax1.set_xlim(-0.08, 1.55)
    ax1.set_ylim(-0.25, 4.25)
    ax1.set_xlabel("Intrinsic Env Rotation: $\\max_m \\,|\\beta_m(A)|/\\alpha_m(A)$", fontweight="bold")
    ax1.set_ylabel("Effective eNTK Rotation: $\\max_m \\,|\\beta_m(\\Theta A)|/\\alpha_m(\\Theta A)$", fontweight="bold")
    ax1.set_title("How Env Rotation Maps to eNTK Rotation ($\\Theta A$)", fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.28, lw=0.7)
    ax1.legend(loc="upper left", fontsize=8.8, framealpha=0.92)

    # Panel 2: AUC VE(E(0)) / AUC VE(TD(0)) vs ||K||_F / ||S||_F (flipped back, colored by rho_c)
    ax2 = axes[1]
    sc2 = ax2.scatter(
        res_df["ks_fro"],
        res_df["ratio_k64"],
        c=res_df["rho_c"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=115,
        edgecolors="#111111",
        linewidths=1.1,
        zorder=4,
    )
    ax2.axhline(1.0, color="#c0392b", ls="--", lw=1.3, label="Parity (E(0) = TD(0))", zorder=2)
    ax2.axhspan(0.0, 1.0, color="#e8f5e9", alpha=0.45, zorder=1, label="E(0) beats TD(0) (y < 1.0)")
    offsets_p2 = {
        "FourRooms (Sparse)": (0.04, -0.10),
        "FourRooms (Dense)": (0.04, -0.02),
        "FourRooms + Mines": (0.02, 0.05),
        "FourRooms + Mines (Dense)": (0.02, -0.04),
        "EightRooms (Sparse)": (0.02, 0.05),
        "EightRooms (Dense)": (0.08, 0.02),
        "Whirlpool": (-0.02, 0.05),
        "MountainCar (Dense)": (-0.02, 0.05),
        "Space Invaders": (-0.02, 0.04),
    }
    for _, r in res_df.iterrows():
        x, y, lbl = r["ks_fro"], r["ratio_k64"], r["label"]
        dx, dy = offsets_p2[lbl]
        ha = "right" if dx < 0 else "left"
        if abs(dx) > 0.03 or abs(dy) > 0.06:
            ax2.annotate(
                lbl, xy=(x, y), xytext=(x + dx, y + dy), fontsize=8.3, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkA=0, shrinkB=4), zorder=5,
            )
        else:
            ax2.text(x + dx, y + dy, lbl, fontsize=8.3, ha=ha, va="center", zorder=5)
    ax2.set_xlim(-0.02, 0.50)
    ax2.set_ylim(0.12, 1.12)
    ax2.set_xlabel("Degree of Skew Symmetry ($\\|K\\|_F / \\|S\\|_F$)", fontweight="bold")
    ax2.set_ylabel("AUC VE(E(0)) / AUC VE(TD(0))  (lower = E wins)", fontweight="bold")
    ax2.set_title("Performance Ratio vs. $\\|K\\|_F / \\|S\\|_F$ (k = 64)", fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.28, lw=0.7)
    ax2.legend(loc="lower left", fontsize=8.8, framealpha=0.92)

    cbar = fig.colorbar(sc2, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Centered Next-State Gradient Similarity $\\rho_c(\\nabla v(s), \\nabla v(s'))$", fontweight="bold")

    fig.suptitle(
        "Env Rotation vs. eNTK Rotation and Skew-Symmetry Performance (k = 64)",
        fontweight="bold",
        fontsize=13.5,
    )

    fig_dir = os.path.join(REPO_ROOT, "investigation", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    png_path = os.path.join(fig_dir, "refined_env_scatter.png")
    pdf_path = os.path.join(fig_dir, "refined_env_scatter.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"\nSaved {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
