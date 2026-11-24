"""
Plots the 3x3 grid of Term 1 vs. Raw Term 2 vs. Gated Term 2 (cancellation of contraction cancellation)
under Rotation-Gated E with omega_t = max(0, (r_rot_ema - tau) / (r_rot_ema + tau)), tau = 0.25:
  Rows:    p_fwd in {0.50 (Reversible), 0.75 (Drifting), 1.00 (Directed)}
  Columns: {1-Hot + MLP, 1-Hot + CNN, 1-Hot + CNN (AvgPool)}

Also generates a 2x3 companion figure on 1-Hot + CNN (AvgPool) showing:
  Row 1: Relative MSE curves (TD(0) vs. Rotation-Gated E vs. Full E(0))
  Row 2: Online r_rot_ema(t) and gate omega_t(t) trajectories.

Saves:
  - investigation/figures/rotation_gated_3x3_term1_vs_term2.png (.pdf)
  - investigation/figures/rotation_gated_chain_summary.png (.pdf)
"""

from __future__ import annotations
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from investigation.scripts.chain_demo_1 import build_chain_mrp
from investigation.scripts.test_step5_chain_metrics import (
    make_model_and_obs,
    compute_jacobian_and_preds,
)


def run_chain_gated_trajectory(
    p_fwd: float,
    col: str,
    mode: str = "adaptive",
    tau: float = 0.25,
    ema: float = 0.8,
    lr: float = 3e-3,
    steps: int = 720,
    eval_every: int = 8,
):
    mrp = build_chain_mrp(N=32, gamma=0.99, p_fwd=p_fwd, ring=False)
    V = np.asarray(mrp["V"], dtype=np.float64)
    V_j = jnp.asarray(mrp["V"], dtype=jnp.float32)
    P_ep = jnp.asarray(mrp["P_ep"], dtype=jnp.float32)
    R_pi = jnp.asarray(mrp["R_pi"], dtype=jnp.float32)
    mu = jnp.asarray(mrp["mu"], dtype=jnp.float32)
    gamma = float(mrp["gamma"])
    V_norm = float(jnp.sum(mu * (V_j ** 2)))

    S_np = np.asarray(mrp["S"], dtype=np.float64)
    D_np = np.diag(np.asarray(mrp["mu"], dtype=np.float64))
    A_np = D_np @ (np.eye(32) - gamma * np.asarray(mrp["P_ep"], dtype=np.float64))
    K_np = 0.5 * (A_np - A_np.T)

    S_j = jnp.asarray(S_np, dtype=jnp.float32)
    K_j = jnp.asarray(K_np, dtype=jnp.float32)

    model, obs = make_model_and_obs(col, 32)
    params = model.init(jax.random.PRNGKey(0), obs)
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)

    def v_fn(p_):
        return model.apply(p_, obs)

    @jax.jit
    def compute_rot_stats(p_):
        def td_loss(pp):
            v = v_fn(pp)
            target = jax.lax.stop_gradient(R_pi + gamma * (P_ep @ v))
            return 0.5 * jnp.sum(mu * (v - target) ** 2)

        g_td = jax.grad(td_loss)(p_)
        g_flat, unravel = ravel_pytree(g_td)
        u_tree = unravel(g_flat / jnp.maximum(jnp.linalg.norm(g_flat), 1e-12))

        _, j_u = jax.jvp(v_fn, (p_,), (u_tree,))
        alpha_eff = jnp.dot(j_u, S_j @ j_u)

        _, vjp_v = jax.vjp(v_fn, p_)
        ku_tree = vjp_v(K_j @ j_u)[0]
        ku_flat, _ = ravel_pytree(ku_tree)
        beta_eff = jnp.linalg.norm(ku_flat)
        return alpha_eff, beta_eff

    @jax.jit
    def step_fn(p_, os_, omega_val):
        # Expected gated descent gradient:
        #   g(omega) = -[(1 + omega) J^T S e + (1 - omega) J^T K e] / (1 + omega)
        # Normalized by 1/(1 + omega) so step size lr is directly comparable across omega in [0, 1].
        def surrogate_loss(pp):
            v = v_fn(pp)
            e = jax.lax.stop_gradient(V_j - v)
            w_s = (1.0 + omega_val) * (S_j @ e) + (1.0 - omega_val) * (K_j @ e)
            return -jnp.dot(v, w_s) / (1.0 + omega_val)

        g = jax.grad(surrogate_loss)(p_)
        u, os_ = tx.update(g, os_, p_)
        return optax.apply_updates(p_, u), os_

    hist = {
        "step": [], "t1": [], "t2_raw": [], "t2_gated": [],
        "net_raw": [], "net_gated": [], "err": [], "rel_mse": [],
        "omega": [], "r_rot": [],
    }

    alpha_ema = 0.0
    beta_ema = 0.0

    for s in range(steps):
        a_eff, b_eff = compute_rot_stats(params)
        a_eff_f, b_eff_f = float(a_eff), float(b_eff)
        if s == 0:
            alpha_ema, beta_ema = a_eff_f, b_eff_f
        else:
            alpha_ema = ema * alpha_ema + (1.0 - ema) * a_eff_f
            beta_ema = ema * beta_ema + (1.0 - ema) * b_eff_f
        r_rot = beta_ema / max(alpha_ema, 1e-6)

        if mode == "adaptive":
            omega = max(0.0, (r_rot - tau) / (r_rot + tau))
        elif mode == "td0":
            omega = 0.0
        elif mode == "e0":
            omega = 1.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if s % eval_every == 0:
            v_pred, J = compute_jacobian_and_preds(model, params, obs)
            e = V - v_pred
            e_s_sq = max(float(e @ S_np @ e), 1e-12)
            rel_mse = float(np.sum(np.asarray(mrp["mu"]) * (e ** 2))) / V_norm
            jse = J.T @ (S_np @ e)
            jke = J.T @ (K_np @ e)
            t1_n = float(np.dot(jse, jse)) / e_s_sq
            t2_raw_n = float(-np.dot(jse, jke)) / e_s_sq
            scale_k = (1.0 - omega) / (1.0 + omega)
            t2_gated_n = scale_k * t2_raw_n
            hist["step"].append(s)
            hist["t1"].append(t1_n)
            hist["t2_raw"].append(t2_raw_n)
            hist["t2_gated"].append(t2_gated_n)
            hist["net_raw"].append(-t1_n + t2_raw_n)
            hist["net_gated"].append(-t1_n + t2_gated_n)
            hist["err"].append(float(np.sqrt(e_s_sq)))
            hist["rel_mse"].append(rel_mse)
            hist["omega"].append(omega)
            hist["r_rot"].append(r_rot)

        params, opt_state = step_fn(params, opt_state, jnp.asarray(omega, dtype=jnp.float32))

    return {k: np.asarray(v) for k, v in hist.items()}


def main():
    tau = 0.25
    p_fwds = [0.50, 0.75, 1.00]
    row_titles = [
        "Reversible ($p_{\\mathrm{fwd}}=0.50$, $K=0$)",
        "Drifting ($p_{\\mathrm{fwd}}=0.75$, $K \\neq 0$)",
        "Directed ($p_{\\mathrm{fwd}}=1.00$, $K \\neq 0$)",
    ]
    cols = ["onehot_mlp", "onehot_cnn", "onehot_cnn_pool"]
    col_titles = ["1-Hot + MLP", "1-Hot + CNN", "1-Hot + CNN (AvgPool)"]

    plt.rcParams.update({"font.size": 9.2, "axes.titlesize": 10.0})
    fig, axes = plt.subplots(3, 3, figsize=(15.4, 10.6), constrained_layout=True)

    print("=" * 116)
    print(
        f"{'p_fwd':<6} | {'Network':<22} | {'mean r_rot':>10} | {'mean omega':>10} | "
        f"{'Raw Cancel %':>12} | {'Gated Cancel %':>14} | {'Gated Non-Contr':>15}"
    )
    print("-" * 116)

    for r_i, p_fwd in enumerate(p_fwds):
        for c_i, col in enumerate(cols):
            # Run TD(0) to get raw ungated cancellation baseline, and adaptive for gated trajectory
            hist_td = run_chain_gated_trajectory(p_fwd, col, mode="td0", tau=tau, lr=3e-3, steps=720, eval_every=8)
            hist_g = run_chain_gated_trajectory(p_fwd, col, mode="adaptive", tau=tau, lr=3e-3, steps=720, eval_every=8)

            sl = slice(3, None)
            steps = hist_g["step"][sl]
            active_g = (hist_g["err"][sl] / max(hist_g["err"][0], 1e-12)) >= 0.03
            active_td = (hist_td["err"][sl] / max(hist_td["err"][0], 1e-12)) >= 0.03

            t1 = hist_g["t1"][sl] * active_g
            t2_raw = hist_g["t2_raw"][sl] * active_g
            t2_gated = hist_g["t2_gated"][sl] * active_g
            net_gated = hist_g["net_gated"][sl] * active_g

            # Raw TD cancellation ratio on the ungated TD trajectory
            int_t1_td = float(np.trapz(np.maximum(hist_td["t1"][sl] * active_td, 0.0), steps))
            int_t2_td = float(np.trapz(hist_td["t2_raw"][sl] * active_td, steps))
            raw_cancel = int_t2_td / max(int_t1_td, 1e-9)

            int_t1_g = float(np.trapz(np.maximum(t1, 0.0), steps))
            int_t2_g = float(np.trapz(t2_gated, steps))
            gated_cancel = int_t2_g / max(int_t1_g, 1e-9)
            non_contract_g = float(np.trapz(np.maximum(net_gated, 0.0), steps))

            mean_r = float(np.mean(hist_g["r_rot"][sl][active_g])) if np.any(active_g) else float(np.mean(hist_g["r_rot"][sl]))
            mean_w = float(np.mean(hist_g["omega"][sl][active_g])) if np.any(active_g) else float(np.mean(hist_g["omega"][sl]))

            print(
                f"{p_fwd:<6.2f} | {col_titles[c_i]:<22} | {mean_r:>10.3f} | {mean_w:>10.3f} | "
                f"{raw_cancel:>11.1%} | {gated_cancel:>13.1%} | {non_contract_g:>15.3f}"
            )

            ax = axes[r_i, c_i]
            ax.plot(steps, t1, color="#1f77b4", lw=2.0, label="Term 1: $\\tilde{T}_1$ ($S$-term, Grad Diff)")
            ax.plot(steps, t2_raw, color="#d62728", lw=1.4, ls=":", alpha=0.85, label="Ungated Term 2: $\\tilde{T}_2$ (Raw $K$-term)")
            ax.plot(steps, t2_gated, color="#2ca02c", lw=2.1, label="Gated Term 2: $\\frac{1-\\omega_t}{1+\\omega_t}\\tilde{T}_2$ (Reduced $K$)")
            ax.fill_between(
                steps, t2_gated, t2_raw,
                where=(t2_raw > t2_gated),
                color="#2ca02c", alpha=0.18,
                label="Cancelled-Out Portion of $K$-term",
            )
            ax.plot(steps, net_gated, color="#111111", lw=1.5, ls="--", label="Net Gated Rate: $-\\tilde{T}_1 + \\frac{1-\\omega_t}{1+\\omega_t}\\tilde{T}_2$")
            ax.fill_between(steps, 0, net_gated, where=(net_gated > 0), color="#d62728", alpha=0.35)
            ax.axhline(0.0, color="#666666", lw=0.8, ls=":")

            if r_i == 0:
                ax.set_title(col_titles[c_i], fontweight="bold")
            if c_i == 0:
                ax.set_ylabel(f"{row_titles[r_i]}\nNormalized Rate", fontweight="bold")
            if r_i == 2:
                ax.set_xlabel("Training Step")

            box_txt = (
                f"$\\bar{{r}}_{{\\mathrm{{rot}}}} = {mean_r:.2f}$, $\\bar{{\\omega}} = {mean_w:.2f}$\n"
                f"TD Cancel: {raw_cancel:.1%}\n"
                f"Gated Cancel: {gated_cancel:.1%}"
            )
            ax.text(
                0.97, 0.95, box_txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=8.1,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#aaaaaa", alpha=0.92),
            )
            ax.grid(True, alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9.2, frameon=True, bbox_to_anchor=(0.5, 1.045))
    fig.suptitle(
        "Cancelling the Contraction Cancellation: Rotation-Gated $E$ ($\\omega_t = \\max(0, (\\bar{r}_{\\mathrm{rot}} - \\tau)/(\\bar{r}_{\\mathrm{rot}} + \\tau))$, $\\tau = 0.25$)",
        fontsize=12.5, fontweight="bold", y=1.075,
    )

    out_dir = os.path.join(REPO_ROOT, "investigation", "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "rotation_gated_3x3_term1_vs_term2.png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "rotation_gated_3x3_term1_vs_term2.pdf"), bbox_inches="tight")
    plt.close(fig)

    # --- Companion 2x3 figure on 1-Hot + CNN (AvgPool): Rel MSE + Gate omega_t ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(13.8, 6.8), constrained_layout=True)
    for c_i, p_fwd in enumerate(p_fwds):
        h_td = run_chain_gated_trajectory(p_fwd, "onehot_cnn_pool", mode="td0", tau=tau, lr=3e-3, steps=720, eval_every=4)
        h_g = run_chain_gated_trajectory(p_fwd, "onehot_cnn_pool", mode="adaptive", tau=tau, lr=3e-3, steps=720, eval_every=4)
        h_e = run_chain_gated_trajectory(p_fwd, "onehot_cnn_pool", mode="e0", tau=tau, lr=3e-3, steps=720, eval_every=4)

        ax_top = axes2[0, c_i]
        ax_top.plot(h_td["step"], h_td["rel_mse"], color="#d62728", lw=2.0, ls="--", label="TD(0) / Forward-Only ($\\omega = 0$)")
        ax_top.plot(h_g["step"], h_g["rel_mse"], color="#2ca02c", lw=2.4, label="Rotation-Gated $E$ ($\\omega_t^*$, $\\tau=0.25$)")
        ax_top.plot(h_e["step"], h_e["rel_mse"], color="#111111", lw=1.9, ls=":", label="Full $E(0)$ ($\\omega = 1$)")
        ax_top.set_yscale("log")
        ax_top.set_ylim(1e-4, 2.0)
        ax_top.set_title(row_titles[c_i], fontweight="bold")
        if c_i == 0:
            ax_top.set_ylabel("Relative Value MSE", fontweight="bold")
        ax_top.grid(True, which="both", alpha=0.22)

        ax_bot = axes2[1, c_i]
        ax_bot.plot(h_g["step"], h_g["omega"], color="#2ca02c", lw=2.2, label="Gate $\\omega_t \\in [0, 1]$")
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.set_xlabel("Training Step")
        if c_i == 0:
            ax_bot.set_ylabel("Backward Gate $\\omega_t$", fontweight="bold", color="#2ca02c")
        ax_bot.grid(True, alpha=0.22)

        ax_r = ax_bot.twinx()
        ax_r.plot(h_g["step"], h_g["r_rot"], color="#1f77b4", lw=1.6, ls="-.", alpha=0.85, label="Online $\\bar{r}_{\\mathrm{rot}}(t)$")
        ax_r.axhline(tau, color="#1f77b4", lw=1.0, ls=":", alpha=0.7, label=f"Threshold $\\tau = {tau}$")
        ax_r.set_ylim(0.0, max(2.5, float(np.max(h_g["r_rot"])) * 1.15))
        if c_i == 2:
            ax_r.set_ylabel("Rotation Ratio $\\bar{r}_{\\mathrm{rot}}$", fontweight="bold", color="#1f77b4")

    h1, l1 = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(h1, l1, loc="upper center", ncol=3, fontsize=9.2, frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig2.suptitle("1-Hot + CNN (AvgPool): Learning Curves and Online Gate $\\omega_t$ Across Reversibility Regimes", fontsize=11.8, fontweight="bold", y=1.095)
    fig2.savefig(os.path.join(out_dir, "rotation_gated_chain_summary.png"), dpi=160, bbox_inches="tight")
    fig2.savefig(os.path.join(out_dir, "rotation_gated_chain_summary.pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"\nSaved figures to {out_dir}")


if __name__ == "__main__":
    main()
