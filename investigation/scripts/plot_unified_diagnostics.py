"""
Unified 2-panel diagnostic figure putting all 4 contending measures side by side:
  Panel 1: Env Spectral Ratio vs. eNTK Spectral Ratio (colored by rho_c)
  Panel 2: rho_c vs. Forward-Only Gradient Alignment (colored by representation)

Generates both Random Policy and Optimal Policy (eps=0.05) versions:
  investigation/figures/unified_diagnostics_random.{png,pdf}
  investigation/figures/unified_diagnostics_optimal.{png,pdf}
"""

from __future__ import annotations
import os
import sys
import argparse
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

from investigation.scripts.env_vs_entk_rotation import (
    MLPNet,
    GridCNN2D,
    GridCNN1D,
    extract_grid_xy,
    mountaincar_to_grid,
    eig_rotation_metric,
)
from investigation.scripts.plot_forward_alignment import load_env_with_V
from investigation.scripts.chain_demo_1 import build_chain_mrp, ChainCNN


def compute_all_measures(net, obs_arr: np.ndarray, mu: np.ndarray, P_pi: np.ndarray, V: np.ndarray, gamma: float = 0.99, seeds=(0, 1, 2)):
    obs_j = jnp.asarray(obs_arr, dtype=jnp.float32)
    n_states = obs_arr.shape[0]
    D = np.diag(mu)
    A = D @ (np.eye(n_states) - gamma * P_pi)
    S = 0.5 * (A + A.T)
    A_fwd = 0.5 * (A + (1.0 - gamma) * D)

    def single_v(p, x):
        return net.apply(p, x[None, ...])[0]

    jac_fn = jax.jit(jax.vmap(jax.jacrev(single_v), in_axes=(None, 0)))

    ntk_sums, rhos, cos_fwds = [], [], []
    for s in seeds:
        params = net.init(jax.random.PRNGKey(s), obs_j[:1])
        v0 = np.asarray(net.apply(params, obs_j))
        e_raw = V - v0
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

        ntk_sums.append(ntk_sum)
        rhos.append(rho_c)
        cos_fwds.append(cos_f)

    return float(np.mean(ntk_sums)), float(np.mean(rhos)), float(np.mean(cos_fwds))


def gather_dataframe(policy: str = "random", gamma: float = 0.99) -> pd.DataFrame:
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

        ntk_s, rho, cf = compute_all_measures(MLPNet(), np.eye(n_states, dtype=np.float32), mu, P_pi, V, gamma)
        records.append({"env": label, "rep": "1-Hot + MLP", "env_sum": env_sum, "ntk_sum": ntk_s, "rho_c": rho, "cos_fwd": cf})

        if has_xy:
            if "mountaincar" in env_name.lower():
                lo, hi = obs_stack.min(axis=0, keepdims=True), obs_stack.max(axis=0, keepdims=True)
                xy_obs = ((obs_stack - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)
            else:
                xy_obs = extract_grid_xy(obs_stack)
            ntk_s, rho, cf = compute_all_measures(MLPNet(), xy_obs, mu, P_pi, V, gamma)
            records.append({"env": label, "rep": "(x, y) + MLP", "env_sum": env_sum, "ntk_sum": ntk_s, "rho_c": rho, "cos_fwd": cf})

        if has_cnn:
            cnn_obs = mountaincar_to_grid(obs_stack) if "mountaincar" in env_name.lower() else obs_stack
            ntk_s, rho, cf = compute_all_measures(GridCNN2D(), cnn_obs, mu, P_pi, V, gamma)
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
        ntk_s, rho, cf = compute_all_measures(net, obs_c, mrp_c["mu"], mrp_c["P_ep"], mrp_c["V"], gamma)
        records.append({"env": "Chain (p_fwd=0.75)", "rep": rep_name, "env_sum": env_sum_c, "ntk_sum": ntk_s, "rho_c": rho, "cos_fwd": cf})

    return pd.DataFrame(records)


def make_plot(df: pd.DataFrame, policy: str, out_name: str | None = None, title_suffix: str = ""):
    plt.rcParams.update({"font.size": 10.0, "axes.titlesize": 11.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.8), constrained_layout=True)

    rep_styles = {
        "1-Hot + MLP": ("o", "#1f77b4"),
        "(x, y) + MLP": ("^", "#e67e22"),
        "Grid CNN": ("s", "#2ca02c"),
        "CNN (AvgPool)": ("D", "#7b2cbf"),
    }

    # --- Panel 1: Env Spectral Ratio vs eNTK Spectral Ratio (colored by rho_c) ---
    for rep, (mk, _) in rep_styles.items():
        sub = df[df["rep"] == rep]
        sc1 = ax1.scatter(
            sub["env_sum"], sub["ntk_sum"],
            c=sub["rho_c"], cmap="viridis", vmin=0.0, vmax=1.0,
            marker=mk, s=110, edgecolors="#111111", linewidths=1.0, label=rep, zorder=4,
        )
    for env_l, grp in df.groupby("env"):
        if grp["env_sum"].iloc[0] > 0.02 or grp["ntk_sum"].max() > 0.05:
            ax1.plot([grp["env_sum"].iloc[0]] * len(grp), grp["ntk_sum"].values, color="#999999", ls=":", lw=1.0, zorder=2)

    if policy == "random":
        xmax1 = 0.50
        ymax1 = max(0.48, float(df["ntk_sum"].max()) * 1.12)
        ax1.annotate(
            "FourRooms, Mines, EightRooms (all)\n& Chain ((x)+MLP): near 0.00",
            xy=(0.0, 0.005), xytext=(0.08, 0.045), fontsize=7.8,
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92), zorder=5,
        )
        p1_specs = [
            ("Space Invaders", "Grid CNN", (0.015, 0.01, "left")),
            ("Space Invaders", "1-Hot + MLP", (0.015, 0.00, "left")),
            ("Chain (p_fwd=0.75)", "1-Hot + MLP", (0.015, 0.012, "left")),
            ("Chain (p_fwd=0.75)", "Grid CNN", (0.015, -0.012, "left")),
            ("Chain (p_fwd=0.75)", "CNN (AvgPool)", (0.015, 0.00, "left")),
            ("MountainCar", "1-Hot + MLP", (-0.015, 0.025, "right")),
            ("MountainCar", "(x, y) + MLP", (-0.015, -0.015, "right")),
            ("MountainCar", "Grid CNN", (-0.015, 0.00, "right")),
            ("Whirlpool", "1-Hot + MLP", (0.015, 0.00, "left")),
            ("Whirlpool", "(x, y) + MLP", (0.015, 0.00, "left")),
            ("Whirlpool", "Grid CNN", (0.015, 0.00, "left")),
        ]
    else:
        xmax1, ymax1 = 0.60, 0.58
        ax1.annotate(
            "FourRooms, Mines, EightRooms &\nChain ((x,y)+MLP): at 0.00",
            xy=(0.001, 0.002), xytext=(0.06, 0.018), fontsize=7.8,
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92), zorder=5,
        )
        ax1.annotate(
            "FourRooms, Mines, EightRooms & Chain:\n1-Hot MLP (0.29–0.31) & 1D CNN (0.27)",
            xy=(0.001, 0.295), xytext=(0.06, 0.375), fontsize=7.8,
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92), zorder=5,
        )
        p1_specs = [
            ("MountainCar", "1-Hot + MLP", (0.012, 0.015, "left")),
            ("MountainCar", "Grid CNN", (-0.012, 0.00, "right")),
            ("MountainCar", "(x, y) + MLP", (0.012, -0.025, "left")),
            ("Whirlpool", "1-Hot + MLP", (0.012, 0.01, "left")),
            ("Whirlpool", "Grid CNN", (0.012, 0.00, "left")),
            ("Whirlpool", "(x, y) + MLP", (0.012, 0.00, "left")),
            ("Space Invaders", "1-Hot + MLP", (-0.012, 0.01, "right")),
            ("Space Invaders", "Grid CNN", (-0.012, -0.005, "right")),
        ]

    for env_l, rep_l, (dx, dy, ha) in p1_specs:
        sub_r = df[(df["env"] == env_l) & (df["rep"] == rep_l)]
        if len(sub_r) == 0:
            continue
        r = sub_r.iloc[0]
        disp_rep = "(pos, vel)" if (env_l == "MountainCar" and rep_l == "(x, y) + MLP") else rep_l
        ax1.text(r["env_sum"] + dx, r["ntk_sum"] + dy, f"{env_l} ({disp_rep})", fontsize=7.5, ha=ha, va="center", zorder=5)

    ax1.plot([0, xmax1], [0, xmax1], color="#c0392b", ls="--", lw=1.2, label="$y = x$", zorder=2)
    ax1.set_xlim(-0.06 * xmax1, xmax1)
    ax1.set_ylim(-0.06 * ymax1, ymax1)
    ax1.set_xlabel("Env Spectral Ratio: $\\sum_m |\\beta_m(A)| / \\sum_m \\alpha_m(A)$", fontweight="bold")
    ax1.set_ylabel("eNTK Spectral Ratio: $\\sum_m |\\beta_m(\\Theta A)| / \\sum_m \\alpha_m(\\Theta A)$", fontweight="bold")
    ax1.set_title("Axis 2: Intrinsic Env vs. eNTK Rotation", fontweight="bold", pad=8)
    ax1.grid(True, alpha=0.28, lw=0.7)
    ax1.legend(loc="lower right" if policy == "random" else "upper left", fontsize=8.3, framealpha=0.92)
    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.042, pad=0.025)
    cbar1.set_label("$\\rho_c(\\nabla v(s), \\nabla v(s'))$", fontweight="bold")

    # --- Panel 2: rho_c vs Forward-Only Alignment (colored by representation) ---
    for rep, (mk, col) in rep_styles.items():
        sub = df[df["rep"] == rep]
        ax2.scatter(
            sub["rho_c"], sub["cos_fwd"],
            marker=mk, color=col, s=110, edgecolors="#111111", linewidths=1.0, label=rep, zorder=4,
        )

    for _, r in df.iterrows():
        env_l, rep_l = r["env"], r["rep"]
        x, y = r["rho_c"], r["cos_fwd"]
        disp_rep = "(pos, vel)" if (env_l == "MountainCar" and rep_l == "(x, y) + MLP") else rep_l
        if env_l in ["Space Invaders", "MountainCar", "Whirlpool", "Chain (p_fwd=0.75)"]:
            dx, dy, ha = 0.02, 0.00, "left"
            if x > 0.72:
                dx, ha = -0.02, "right"
            if policy == "random":
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
                elif env_l == "MountainCar" and rep_l == "1-Hot + MLP":
                    dx, dy = 0.02, 0.03
                elif env_l == "MountainCar" and rep_l == "Grid CNN":
                    dx, dy, ha = -0.02, -0.03, "right"
                elif env_l == "Chain (p_fwd=0.75)" and rep_l == "CNN (AvgPool)":
                    dx, dy, ha = -0.02, 0.03, "right"
                elif env_l == "Chain (p_fwd=0.75)" and rep_l == "(x, y) + MLP":
                    dx, dy, ha = -0.02, -0.03, "right"
            else:
                if env_l == "Chain (p_fwd=0.75)" and rep_l == "Grid CNN":
                    dx, dy = 0.02, 0.03
                elif env_l == "Chain (p_fwd=0.75)" and rep_l == "1-Hot + MLP":
                    dx, dy = 0.02, -0.03
                elif env_l == "Chain (p_fwd=0.75)" and rep_l == "CNN (AvgPool)":
                    dx, dy, ha = -0.02, 0.035, "right"
                elif env_l == "Chain (p_fwd=0.75)" and rep_l == "(x, y) + MLP":
                    dx, dy, ha = -0.02, -0.035, "right"
                elif env_l == "Whirlpool" and rep_l == "(x, y) + MLP":
                    dx, dy, ha = -0.02, 0.03, "right"
                elif env_l == "MountainCar" and rep_l == "(x, y) + MLP":
                    dx, dy, ha = -0.02, -0.03, "right"
                elif env_l == "MountainCar" and rep_l == "Grid CNN":
                    dx, dy, ha = -0.02, -0.01, "right"
            ax2.text(x + dx, y + dy, f"{env_l} ({disp_rep})", fontsize=7.5, ha=ha, va="center", zorder=5)

    ax2.axhline(0.0, color="#888888", ls=":", lw=1.0)
    ax2.set_xlim(-0.08, 1.06)
    ax2.set_ylim(-0.34, 1.06)
    ax2.set_xlabel("Centered Next-State Gradient Similarity $\\rho_c(\\nabla v(s), \\nabla v(s'))$", fontweight="bold")
    ax2.set_ylabel("Forward-Only Alignment: $\\cos(\\nabla_\\theta E, \\nabla_\\theta E_{\\mathrm{Fwd}})$", fontweight="bold")
    ax2.set_title("Axis 1: Spillover $\\rho_c$ Collapses Forward-Only Alignment", fontweight="bold", pad=8)
    ax2.grid(True, alpha=0.28, lw=0.7)
    ax2.legend(loc="lower left", fontsize=8.3, framealpha=0.92)

    pol_title = "Uniform Random Policy" if policy == "random" else "Optimal Policy ($\\epsilon=0.05$)"
    fig.suptitle(f"Unified Diagnostics: Rotation Ratios, $\\rho_c$, and Forward-Only Alignment ({pol_title}{title_suffix})", fontweight="bold", fontsize=13.0)

    fig_dir = os.path.join(REPO_ROOT, "investigation", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    stem = out_name or f"unified_diagnostics_{policy}"
    png_path = os.path.join(fig_dir, f"{stem}.png")
    pdf_path = os.path.join(fig_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {png_path} and {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "optimal", "both"], default="both")
    args = parser.parse_args()
    policies = ["random", "optimal"] if args.policy == "both" else [args.policy]
    for pol in policies:
        df = gather_dataframe(policy=pol)
        make_plot(df, policy=pol)


if __name__ == "__main__":
    main()
