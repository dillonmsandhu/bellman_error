"""
Diagnostic plot on Space Invaders (SpaceInvadersExactValue, near-optimal policy, Grid CNN, gamma=0.99):
Compares exact TD(0), exact Full E(0), and exact Rotation-Gated E(0) (tau=0.25, ema=0.95):
  - Panel 1: Relative Value MSE learning curves
  - Panel 2: Term 1 (T1), Ungated Term 2 (T2), Gated Term 2 ((1-omega)/(1+omega)*T2), and Net Rate
  - Panel 3: Online r_rot_ema(t) vs threshold tau=0.25 and resulting backward gate omega_t

Saves:
  - investigation/figures/space_invaders_gated_diagnostic.png (.pdf)
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

from investigation.scripts.env_vs_entk_rotation import GridCNN2D
from investigation.scripts.plot_forward_alignment import load_env_with_V


def run_space_invaders(obs_arr, mu, P_pi, V, mode="adaptive", tau=0.25, ema=0.95, gamma=0.99, lr=3e-3, steps=400, eval_every=4):
    obs_j = jnp.asarray(obs_arr, dtype=jnp.float32)
    mu_np = np.asarray(mu, dtype=np.float64)
    P_np = np.asarray(P_pi, dtype=np.float64)
    V_np = np.asarray(V, dtype=np.float64)

    D_np = np.diag(mu_np)
    A_np = D_np @ (np.eye(len(mu_np)) - gamma * P_np)
    S_np = 0.5 * (A_np + A_np.T)
    K_np = 0.5 * (A_np - A_np.T)
    V_norm = float(np.sum(mu_np * (V_np ** 2)))

    mu_j = jnp.asarray(mu_np, dtype=jnp.float32)
    A_j = jnp.asarray(A_np, dtype=jnp.float32)
    S_j = jnp.asarray(S_np, dtype=jnp.float32)
    K_j = jnp.asarray(K_np, dtype=jnp.float32)
    V_j = jnp.asarray(V_np, dtype=jnp.float32)

    net = GridCNN2D()
    params = net.init(jax.random.PRNGKey(0), obs_j)
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)

    def v_fn(p):
        return net.apply(p, obs_j)

    @jax.jit
    def probe_step(p):
        v = v_fn(p)
        e = V_j - v
        e_s_sq = jnp.maximum(jnp.dot(e, S_j @ e), 1e-12)
        rel_mse = jnp.sum(mu_j * (e ** 2)) / V_norm

        _, vjp_v = jax.vjp(v_fn, p)
        jse_flat, _ = ravel_pytree(vjp_v(S_j @ e)[0])
        jke_flat, _ = ravel_pytree(vjp_v(K_j @ e)[0])
        t1 = jnp.dot(jse_flat, jse_flat) / e_s_sq
        t2 = -jnp.dot(jse_flat, jke_flat) / e_s_sq

        # Centered JVP / VJP: J_c^T w = J^T (w - (1^T w) mu), j_c = j - (mu^T j) 1
        w_ae = -(A_j @ e)
        w_ae_c = w_ae - jnp.sum(w_ae) * mu_j
        g_td_flat, unravel = ravel_pytree(vjp_v(w_ae_c)[0])
        u_flat = g_td_flat / jnp.maximum(jnp.linalg.norm(g_td_flat), 1e-12)
        u_tree = unravel(u_flat)

        _, j_u = jax.jvp(v_fn, (p,), (u_tree,))
        j_c = j_u - jnp.dot(mu_j, j_u)
        alpha_eff = jnp.dot(j_c, S_j @ j_c)
        w_k = K_j @ j_c
        w_k_c = w_k - jnp.sum(w_k) * mu_j
        ku_flat, _ = ravel_pytree(vjp_v(w_k_c)[0])
        beta_eff = jnp.linalg.norm(ku_flat)
        return rel_mse, t1, t2, alpha_eff, beta_eff, jnp.sqrt(e_s_sq)

    @jax.jit
    def update_step(p, os_, omega_val):
        def surrogate_loss(pp):
            v = v_fn(pp)
            e = jax.lax.stop_gradient(V_j - v)
            w_s = (1.0 + omega_val) * (S_j @ e) + (1.0 - omega_val) * (K_j @ e)
            return -jnp.dot(v, w_s) / (1.0 + omega_val)

        g = jax.grad(surrogate_loss)(p)
        u, os_ = tx.update(g, os_, p)
        return optax.apply_updates(p, u), os_

    hist = {
        "step": [], "rel_mse": [], "t1": [], "t2_raw": [], "t2_gated": [],
        "net_gated": [], "r_rot": [], "omega": [], "err": [],
    }

    alpha_ema, beta_ema = 0.0, 0.0
    omega = 0.0 if mode == "adaptive" else (1.0 if mode == "e0" else 0.0)

    for s in range(steps):
        rel_mse, t1, t2, a_eff, b_eff, err = probe_step(params)
        a_f, b_f = float(a_eff), float(b_eff)
        if s == 0:
            alpha_ema, beta_ema = a_f, b_f
        else:
            alpha_ema = ema * alpha_ema + (1.0 - ema) * a_f
            beta_ema = ema * beta_ema + (1.0 - ema) * b_f
        r_rot = beta_ema / max(alpha_ema, 1e-6)

        if mode == "adaptive":
            omega_target = max(0.0, (r_rot - tau) / (r_rot + tau))
            omega = ema * omega + (1.0 - ema) * omega_target
        elif mode == "td0":
            omega = 0.0
        elif mode == "e0":
            omega = 1.0

        if s % eval_every == 0:
            t1_f, t2_f = float(t1), float(t2)
            t2_g = ((1.0 - omega) / (1.0 + omega)) * t2_f
            hist["step"].append(s)
            hist["rel_mse"].append(float(rel_mse))
            hist["t1"].append(t1_f)
            hist["t2_raw"].append(t2_f)
            hist["t2_gated"].append(t2_g)
            hist["net_gated"].append(-t1_f + t2_g)
            hist["r_rot"].append(r_rot)
            hist["omega"].append(omega)
            hist["err"].append(float(err))

        params, opt_state = update_step(params, opt_state, jnp.asarray(omega, dtype=jnp.float32))

    return {k: np.asarray(v) for k, v in hist.items()}


def main():
    obs_stack, mu, P_pi, V = load_env_with_V("SpaceInvadersExactValue", gamma=0.99, policy="optimal")
    h_td = run_space_invaders(obs_stack, mu, P_pi, V, mode="td0")
    h_e0 = run_space_invaders(obs_stack, mu, P_pi, V, mode="e0")
    h_gt = run_space_invaders(obs_stack, mu, P_pi, V, mode="adaptive", tau=0.25, ema=0.95)

    print(f"Space Invaders (Grid CNN, optimal policy):")
    print(f"  TD(0)           | AUC={h_td['rel_mse'].mean():.5f} | Fin={h_td['rel_mse'][-1]:.5f}")
    print(f"  Full E(0)       | AUC={h_e0['rel_mse'].mean():.5f} | Fin={h_e0['rel_mse'][-1]:.5f}")
    print(f"  Rot-Gated E(0)  | AUC={h_gt['rel_mse'].mean():.5f} | Fin={h_gt['rel_mse'][-1]:.5f} | mean r_rot={h_gt['r_rot'].mean():.3f} | max r_rot={h_gt['r_rot'].max():.3f} | mean omega={h_gt['omega'].mean():.4f}")

    sl = slice(2, None)
    int_t1 = float(np.trapz(np.maximum(h_td["t1"][sl], 0.0), h_td["step"][sl]))
    int_t2 = float(np.trapz(h_td["t2_raw"][sl], h_td["step"][sl]))
    print(f"  TD(0) Avg Contraction Cancelled (int T2 / int T1) = {int_t2 / max(int_t1, 1e-12):.2%}")

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)

    # Panel 1: Relative MSE
    ax1.plot(h_td["step"], h_td["rel_mse"], color="#d62728", lw=2.2, ls="--", label="Exact TD(0) ($\\omega=0$)")
    ax1.plot(h_gt["step"], h_gt["rel_mse"], color="#2ca02c", lw=2.4, alpha=0.9, label="Rot-Gated E(0) ($\\tau=0.25$)")
    ax1.plot(h_e0["step"], h_e0["rel_mse"], color="#111111", lw=1.9, ls=":", label="Exact Full E(0) ($\\omega=1$)")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-4, 2.0)
    ax1.set_title("Value Learning Curves", fontweight="bold")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Relative Value MSE")
    ax1.grid(True, which="both", alpha=0.22)
    ax1.legend(fontsize=8.5, loc="upper right")

    # Panel 2: T1 vs T2 (raw & gated)
    steps = h_gt["step"][sl]
    ax2.plot(steps, h_gt["t1"][sl], color="#1f77b4", lw=2.1, label="Term 1: $\\tilde{T}_1$ ($S$-term)")
    ax2.plot(steps, h_gt["t2_raw"][sl], color="#d62728", lw=1.6, ls=":", label="Term 2: $\\tilde{T}_2$ (Raw $K$-term)")
    ax2.plot(steps, h_gt["t2_gated"][sl], color="#2ca02c", lw=2.0, label="Gated Term 2: $\\frac{1-\\omega_t}{1+\\omega_t}\\tilde{T}_2$")
    ax2.plot(steps, h_gt["net_gated"][sl], color="#111111", lw=1.5, ls="--", label="Net Rate: $-\\tilde{T}_1 + \\tilde{T}_{2,\\mathrm{gated}}$")
    ax2.axhline(0.0, color="#666666", lw=0.8, ls=":")
    ax2.set_title(f"Contraction Terms (Cancel = {int_t2 / max(int_t1, 1e-12):.1%})", fontweight="bold")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Normalized Rate")
    ax2.grid(True, alpha=0.22)
    ax2.legend(fontsize=8.0, loc="upper right")

    # Panel 3: centered r_rot_c and omega_t
    l1, = ax3.plot(h_gt["step"], h_gt["r_rot"], color="#1f77b4", lw=2.1, label="Centered $\\bar{r}_{\\mathrm{rot},c}(t)$")
    l2 = ax3.axhline(0.25, color="#444444", lw=1.3, ls="--", label="Threshold $\\tau = 0.25$")
    ax3.set_ylim(0.0, max(0.85, float(h_gt["r_rot"].max()) * 1.25))
    ax3.set_title("Centered $\\bar{r}_{\\mathrm{rot},c}(t)$ and Backward Gate $\\omega_t$", fontweight="bold")
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Centered Rotation Ratio $\\bar{r}_{\\mathrm{rot},c}(t)$", color="#1f77b4", fontweight="bold")
    ax3.grid(True, alpha=0.22)

    ax3_r = ax3.twinx()
    l3, = ax3_r.plot(h_gt["step"], h_gt["omega"], color="#2ca02c", lw=2.2, ls="-.", label="Backward Gate $\\omega_t$")
    ax3_r.set_ylim(-0.04, 1.04)
    ax3_r.set_ylabel("Backward Gate $\\omega_t$", color="#2ca02c", fontweight="bold")
    ax3.legend([l1, l2, l3], [l.get_label() for l in (l1, l2, l3)], fontsize=8.3, loc="upper right")

    fig.suptitle("Exact Space Invaders (Near-Optimal Policy, Grid CNN, Centered JVP, $\\gamma=0.99$): TD(0) vs. E(0) vs. Rotation-Gated E(0)", fontsize=11.3, fontweight="bold")

    out_dir = os.path.join(REPO_ROOT, "investigation", "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "space_invaders_gated_diagnostic.png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "space_invaders_gated_diagnostic.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_dir}/space_invaders_gated_diagnostic.png")


if __name__ == "__main__":
    main()
