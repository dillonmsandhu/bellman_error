"""
investigation/scripts/measure_tabular_contraction_diagnostics.py

Measures all four canonical contraction-loss & return-variance diagnostics across
the exact Tabular benchmark suite (FourRooms, FourRooms+Mines, EightRooms,
Whirlpool, MountainCar, Space Invaders) for TD(lambda) vs. E(lambda):
  1. diag_rho_c:         Centered gradient aliasing rho_c
  2. diag_r_rot_c:       Centered rotation ratio r_{rot,c}
  3. diag_ret_var_ratio: Return-to-TD error-variance ratio E[(G - v)^2] / E[delta^2]
  4. diag_aux_unexp_var: Unexplained value variance 1 - R^2

Outputs (under investigation/figures/tabular_diagnostics/):
  - learning_curves_diag_*.{png,pdf}
  - diagnostic_auc_per_env.{png,pdf}
  - diagnostic_benchmark_avg_auc.{png,pdf}
  - scatter_2x2_vs_rel_perf_lam*.{png,pdf}
  - tabular_diagnostics_summary.csv / .md
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from investigation.scripts.env_vs_entk_rotation import GridCNN2D, MLPNet
from investigation.scripts.plot_forward_alignment import extract_grid_xy, load_env_with_V, mountaincar_to_grid


def _build_lambda_operator(P_np: np.ndarray, gamma: float, lmbda: float) -> tuple[np.ndarray, float]:
    """Returns effective transition matrix P_lam and gamma_eff such that (I - A_lam) = gamma_eff * P_lam."""
    n = P_np.shape[0]
    I = np.eye(n, dtype=np.float64)
    if lmbda <= 1e-8:
        return P_np.copy(), float(gamma)
    # (1 - lmbda) * sum_{k>=0} lmbda^k (gamma P)^{k+1} = (1 - lmbda) * gamma P (I - gamma lmbda P)^{-1}
    gamma_P_lam = (1.0 - lmbda) * gamma * P_np @ np.linalg.inv(I - gamma * lmbda * P_np)
    gamma_eff = float((gamma * (1.0 - lmbda)) / max(1.0 - gamma * lmbda, 1e-8))
    return gamma_P_lam / max(gamma_eff, 1e-12), gamma_eff


def _run_exact_critic(
    net,
    obs_arr: np.ndarray,
    mu: np.ndarray,
    P_pi: np.ndarray,
    V: np.ndarray,
    algo: str,
    gamma: float = 0.99,
    lmbda: float = 0.8,
    lr: float = 3e-3,
    steps: int = 300,
    eval_every: int = 5,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    obs_j = jnp.asarray(obs_arr, dtype=jnp.float32)
    mu_np = np.asarray(mu, dtype=np.float64)
    P_np = np.asarray(P_pi, dtype=np.float64)
    V_np = np.asarray(V, dtype=np.float64)

    P_lam_np, gamma_eff = _build_lambda_operator(P_np, gamma, lmbda)
    D_np = np.diag(mu_np)
    A_lam_np = D_np @ (np.eye(len(mu_np)) - gamma_eff * P_lam_np)
    S_lam_np = 0.5 * (A_lam_np + A_lam_np.T)
    K_lam_np = 0.5 * (A_lam_np - A_lam_np.T)
    I_minus_gP_np = np.eye(len(mu_np)) - gamma * P_np

    V_var = max(float(np.sum(mu_np * (V_np - np.sum(mu_np * V_np)) ** 2)), 1e-8)
    V_norm = max(float(np.sum(mu_np * (V_np ** 2))), 1e-8)

    mu_j = jnp.asarray(mu_np, dtype=jnp.float32)
    P_lam_j = jnp.asarray(P_lam_np, dtype=jnp.float32)
    A_lam_j = jnp.asarray(A_lam_np, dtype=jnp.float32)
    S_lam_j = jnp.asarray(S_lam_np, dtype=jnp.float32)
    K_lam_j = jnp.asarray(K_lam_np, dtype=jnp.float32)
    I_minus_gP_j = jnp.asarray(I_minus_gP_np, dtype=jnp.float32)
    V_j = jnp.asarray(V_np, dtype=jnp.float32)

    params = net.init(jax.random.PRNGKey(seed), obs_j)
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)

    def v_fn(p):
        out = net.apply(p, obs_j)
        return out.squeeze(-1) if out.ndim > 1 else out

    @jax.jit
    def _measure_and_step(p, opt_st):
        v = v_fn(p)
        err = v - V_j
        rel_mse = jnp.sum(mu_j * (err ** 2)) / V_norm

        # 1. Centered TD(lambda) probe direction u_c
        w_td_c = A_lam_j @ err
        w_td_c = w_td_c - jnp.sum(mu_j * w_td_c)
        _, vjp_v = jax.vjp(v_fn, p)
        g_td_c = vjp_v(w_td_c)[0]
        g_flat, unravel = ravel_pytree(g_td_c)
        u_tree = unravel(g_flat / jnp.maximum(jnp.linalg.norm(g_flat), 1e-12))

        _, j_raw = jax.jvp(v_fn, (p,), (u_tree,))
        j = j_raw - jnp.sum(mu_j * j_raw)
        j_fwd = gamma_eff * (P_lam_j @ j)

        j_sq = jnp.sum(mu_j * j * j)
        alpha_eff = j_sq - jnp.sum(mu_j * j * j_fwd)
        rho_c = jnp.sum(mu_j * j * j_fwd) / jnp.maximum(gamma_eff * j_sq, 1e-12)

        k_vec = K_lam_j @ j
        k_vec_c = k_vec - jnp.sum(mu_j * k_vec)
        ku_flat, _ = ravel_pytree(vjp_v(k_vec_c)[0])
        beta_eff = jnp.linalg.norm(ku_flat)
        r_rot_c = beta_eff / jnp.maximum(alpha_eff, 1e-6)

        # 3. Return-to-TD error ratio E[(V - v)^2] / E[((I - gamma P)(V - v))^2]
        td_err = I_minus_gP_j @ err
        ret_var_ratio = jnp.sum(mu_j * (err ** 2)) / jnp.maximum(jnp.sum(mu_j * (td_err ** 2)), 1e-8)

        # 4. Unexplained value variance 1 - R^2
        unexp_var = jnp.sum(mu_j * (err ** 2)) / V_var

        # Gradient update: TD(lambda) uses A_lam @ err; E(lambda) uses 2 * S_lam @ err
        if algo == "td":
            grad_vec = A_lam_j @ err
        else:
            grad_vec = 2.0 * (S_lam_j @ err)
        grads = vjp_v(grad_vec)[0]
        updates, new_opt_st = tx.update(grads, opt_st, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_st, (rel_mse, rho_c, r_rot_c, ret_var_ratio, unexp_var)

    hist = {"step": [], "rel_mse": [], "diag_rho_c": [], "diag_r_rot_c": [], "diag_ret_var_ratio": [], "diag_aux_unexp_var": []}
    for step in range(steps):
        params, opt_state, vals = _measure_and_step(params, opt_state)
        if step % eval_every == 0:
            rm, rc, rr, vr, uv = [float(x) for x in vals]
            hist["step"].append(step)
            hist["rel_mse"].append(rm)
            hist["diag_rho_c"].append(rc)
            hist["diag_r_rot_c"].append(rr)
            hist["diag_ret_var_ratio"].append(vr)
            hist["diag_aux_unexp_var"].append(uv)

    return {k: np.asarray(v, dtype=np.float64) for k, v in hist.items()}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank().to_numpy(dtype=np.float64)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(description="Tabular Contraction-Loss & Variance Diagnostics")
    parser.add_argument("--lmbda", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--policy", choices=["random", "optimal"], default="optimal")
    parser.add_argument("--out_dir", type=str, default=os.path.join(REPO_ROOT, "investigation", "figures", "tabular_diagnostics"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env_specs = [
        ("FourRooms-misc", "FourRooms (xy)", "xy"),
        ("fourrooms-mines", "FourRooms+Mines (xy)", "xy"),
        ("eightrooms-misc", "EightRooms (xy)", "xy"),
        ("whirlpool-misc", "Whirlpool (xy)", "xy"),
        ("mountaincar-dense", "MountainCar (xy)", "xy"),
        ("SpaceInvadersExactValue", "SpaceInvaders (CNN)", "cnn"),
    ]

    results = {}
    rows = []
    for env_name, label, rep in env_specs:
        obs_stack, mu, P_pi, V = load_env_with_V(env_name, gamma=args.gamma, policy=args.policy)
        if rep == "cnn":
            obs_arr = mountaincar_to_grid(obs_stack) if "mountaincar" in env_name.lower() else obs_stack
            net = GridCNN2D()
        else:
            if "mountaincar" in env_name.lower():
                lo, hi = obs_stack.min(axis=0, keepdims=True), obs_stack.max(axis=0, keepdims=True)
                obs_arr = ((obs_stack - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)
            else:
                obs_arr = extract_grid_xy(obs_stack)
            net = MLPNet()

        td_res = _run_exact_critic(net, obs_arr, mu, P_pi, V, algo="td", gamma=args.gamma, lmbda=args.lmbda, lr=args.lr, steps=args.steps)
        e_res = _run_exact_critic(net, obs_arr, mu, P_pi, V, algo="e", gamma=args.gamma, lmbda=args.lmbda, lr=args.lr, steps=args.steps)
        results[label] = {"td": td_res, "e": e_res}

        td_mse_auc = float(np.mean(td_res["rel_mse"]))
        e_mse_auc = float(np.mean(e_res["rel_mse"]))
        rel_gain_pct = 100.0 * (td_mse_auc - e_mse_auc) / max(td_mse_auc, 1e-8)

        row = {
            "Environment": label,
            "TD_MSE_AUC": td_mse_auc,
            "E_MSE_AUC": e_mse_auc,
            "Rel_Gain_Pct": rel_gain_pct,
            "diag_rho_c": float(np.mean(td_res["diag_rho_c"])),
            "diag_r_rot_c": float(np.mean(td_res["diag_r_rot_c"])),
            "diag_ret_var_ratio": float(np.mean(td_res["diag_ret_var_ratio"])),
            "diag_aux_unexp_var": float(np.mean(e_res["diag_aux_unexp_var"])),
        }
        rows.append(row)
        print(
            f"{label:22s} | Gain={rel_gain_pct:+6.1f}% | rho_c={row['diag_rho_c']:.3f} "
            f"r_rot={row['diag_r_rot_c']:.3f} vr={row['diag_ret_var_ratio']:.1f} uv={row['diag_aux_unexp_var']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "tabular_diagnostics_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "tabular_diagnostics_summary.md"), "w") as f:
        f.write(f"### Tabular Contraction & Variance Diagnostics (λ={args.lmbda:g}, policy={args.policy})\n\n")
        f.write(df.to_markdown(index=False) + "\n")

    # 2x2 Scatter Grid vs Relative E(lambda) / TD(lambda) Gain
    diag_meta = [
        ("diag_rho_c", "Centered Gradient Aliasing (ρ_c)", 0.85),
        ("diag_r_rot_c", "Centered Rotation Ratio (r_rot,c)", 0.25),
        ("diag_ret_var_ratio", "Return-to-TD Error Ratio (VR_ret)", 1.0),
        ("diag_aux_unexp_var", "Unexplained Value Variance (1 - R²)", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8))
    ys = df["Rel_Gain_Pct"].to_numpy(dtype=np.float64)
    for idx, (col, title, ref_v) in enumerate(diag_meta):
        ax = axes[idx // 2][idx % 2]
        xs = df[col].to_numpy(dtype=np.float64)
        colors = ["#16a34a" if y > 1.0 else ("#dc2626" if y < -1.0 else "#64748b") for y in ys]
        ax.axhline(0.0, color="#334155", ls="--", lw=1.1, alpha=0.8)
        ax.axvline(ref_v, color="#94a3b8", ls=":", lw=1.0, alpha=0.8)
        ax.scatter(xs, ys, s=75, c=colors, edgecolors="#111827", linewidths=0.7, zorder=4)
        for xi, yi, txt in zip(xs, ys, df["Environment"]):
            ax.annotate(txt, (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=8.0)
        if len(xs) >= 3 and np.std(xs) > 1e-12:
            m, b = np.polyfit(xs, ys, 1)
            xl = np.linspace(float(xs.min()), float(xs.max()), 50)
            ax.plot(xl, m * xl + b, color="#2563eb", lw=1.4, alpha=0.7)
        r_p = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) > 1e-12 and np.std(ys) > 1e-12 else float("nan")
        r_s = _spearman(xs, ys)
        ax.text(
            0.03, 0.96, f"Pearson r = {r_p:+.2f}\nSpearman ρ = {r_s:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.92),
        )
        ax.set_xlabel(f"AUC of {title}", fontsize=9.2)
        ax.set_ylabel("Relative MSE Reduction: E(λ) vs. TD(λ) (%)", fontsize=8.8)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Tabular Suite: Diagnostic AUCs vs. E(λ)/TD(λ) Gain (λ={args.lmbda:g})", fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    base = os.path.join(args.out_dir, f"scatter_2x2_vs_rel_perf_lam{str(args.lmbda).replace('.', 'p')}")
    plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tabular diagnostic outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
