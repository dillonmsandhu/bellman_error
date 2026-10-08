"""
scripts/sampling_E_test/compare_e_variants.py

Head-to-head comparison of the 3 Sampled E formulations against Exact Matrix E:
1. Sept 14 (Uncorrected):
   L = (1-gamma) E[e_i^2] + (gamma/2) E[(e_i - e_j)^2]
2. Oct 5 (Corrected / High Variance):
   L = (1-gamma) E[e_i^2] + (gamma/2) E[(e_i - e_j)^2] + (gamma/2) (E[e_i^2] - E[e_j^2])
3. Oct 8 (Start-State Anchored / Exact Non-Negative):
   L = (1-gamma) E[e_i^2] + (gamma/2) E[(e_i - e_j)^2] + (gamma/2) nu E_{s0}[e(s0)^2]

Evaluates:
- Loss value alignment against Exact Matrix E
- Cosine similarity of sample gradients to Exact E gradient
- Empirical gradient variance across independent rollout batches
- Gradient MSE to Exact E
- Full PPO policy learning curves across the 3 variants
"""

from __future__ import annotations
import os
import sys
import argparse
import time
from typing import Dict, Any, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from core.config import config as base_cfg
from core.helpers import make_env, create_evaluator
from core.networks import initialize_network
from core.gradient_tracking import extract_critic_flat_grads, cos_similarity
from ppo.sampled_E import make_train as make_e_train


def analyze_gradient_variance(
    evaluator,
    network,
    params,
    gamma: float = 0.99,
    num_batches: int = 50,
    num_trajectories: int = 64,
    num_steps: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes exact gradient and evaluates sample gradient distribution across
    num_batches independent sample rollout batches for all 3 variants.
    """
    S_states = evaluator.obs_stack
    N = evaluator.num_states
    start_idx = evaluator.start_idx
    P = evaluator.P

    # Policy and occupancy
    old_pi = jnp.ones((N, 4)) / 4.0
    terminal_policy = jnp.ones([1, 4]) / 4.0
    old_pi_full = jnp.vstack([old_pi, terminal_policy])

    mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
    mu = jnp.append(mu, 0.0)
    D = jnp.diag(mu)
    P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
    I = jnp.eye(N + 1)
    A_mat = D @ (I - gamma * P_pi)
    S_mat = 0.5 * (A_mat + A_mat.T)
    V_true = evaluator.compute_true_values_raw(old_pi_full)

    def get_e(p):
        v = network.apply(p, S_states, method=network.value).squeeze()
        return V_true - jnp.append(v, 0.0)

    # 1. Exact Matrix Loss & Gradient
    def loss_exact(p):
        e = get_e(p)
        return e.T @ S_mat @ e

    val_exact = float(loss_exact(params))
    g_exact = extract_critic_flat_grads(jax.grad(loss_exact)(params))
    norm_g_exact = float(jnp.linalg.norm(g_exact))

    # 2. Expected Losses & Gradients
    flux = mu - (mu @ P_pi)

    def loss_exp_sep14(p):
        e = get_e(p)
        mc = (1.0 - gamma) * jnp.sum(mu * (e ** 2))
        dirichlet = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e[:, None] - e[None, :]) ** 2))
        return mc + dirichlet

    def loss_exp_corr(p):
        e = get_e(p)
        mc = (1.0 - gamma) * jnp.sum(mu * (e ** 2))
        dirichlet = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e[:, None] - e[None, :]) ** 2))
        corr = 0.5 * gamma * (jnp.sum(mu * (e ** 2)) - jnp.sum((mu @ P_pi) * (e ** 2)))
        return mc + dirichlet + corr

    def loss_exp_oct8(p):
        e = get_e(p)
        mc = (1.0 - gamma) * jnp.sum(mu * (e ** 2))
        dirichlet = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e[:, None] - e[None, :]) ** 2))
        start_anchor = 0.5 * gamma * jnp.sum(flux * (e ** 2))
        return mc + dirichlet + start_anchor

    g_exp_sep14 = extract_critic_flat_grads(jax.grad(loss_exp_sep14)(params))
    g_exp_corr = extract_critic_flat_grads(jax.grad(loss_exp_corr)(params))
    g_exp_oct8 = extract_critic_flat_grads(jax.grad(loss_exp_oct8)(params))

    # 3. Sample over num_batches independent rollout draws
    rng = jax.random.PRNGKey(seed)
    mu_trans = mu[:N] / jnp.maximum(jnp.sum(mu[:N]), 1e-12)

    def sample_rollout(roll_rng):
        r_init, r_steps = jax.random.split(roll_rng)
        s0 = jax.random.choice(r_init, N, shape=(num_trajectories,), p=mu_trans)

        def step_fn(carry_s, step_rng):
            step_rngs = jax.random.split(step_rng, num_trajectories)
            logits = jnp.log(jnp.maximum(P_pi[carry_s], 1e-12))
            next_s_raw = jax.vmap(jax.random.categorical)(step_rngs, logits)
            next_s_carry = jnp.where(next_s_raw == N, start_idx, next_s_raw)
            return next_s_carry, (carry_s, next_s_raw)

        step_rngs = jax.random.split(r_steps, num_steps)
        _, (traj_s, traj_next_s) = jax.lax.scan(step_fn, s0, step_rngs)
        return traj_s, traj_next_s

    # Define per-batch sample loss functions
    def make_batch_loss(traj_s, traj_next_s):
        def l_sep14(p):
            e = get_e(p)
            es = e[traj_s]
            en = e[traj_next_s]
            return (1.0 - gamma) * jnp.mean(es ** 2) + 0.5 * gamma * jnp.mean((es - en) ** 2)

        def l_corr(p):
            e = get_e(p)
            es = e[traj_s]
            en = e[traj_next_s]
            return (
                (1.0 - gamma) * jnp.mean(es ** 2)
                + 0.5 * gamma * jnp.mean((es - en) ** 2)
                + 0.5 * gamma * (jnp.mean(es ** 2) - jnp.mean(en ** 2))
            )

        def l_oct8(p):
            e = get_e(p)
            es = e[traj_s]
            en = e[traj_next_s]
            nu = jnp.mean(traj_next_s == N)
            return (
                (1.0 - gamma) * jnp.mean(es ** 2)
                + 0.5 * gamma * jnp.mean((es - en) ** 2)
                + 0.5 * gamma * nu * (e[start_idx] ** 2)
            )

        return l_sep14, l_corr, l_oct8

    grads_sep14 = []
    grads_corr = []
    grads_oct8 = []

    sub_keys = jax.random.split(rng, num_batches)
    for k in sub_keys:
        traj_s, traj_next_s = sample_rollout(k)
        l_sep14, l_corr, l_oct8 = make_batch_loss(traj_s, traj_next_s)

        g_s14 = extract_critic_flat_grads(jax.grad(l_sep14)(params))
        g_c = extract_critic_flat_grads(jax.grad(l_corr)(params))
        g_o8 = extract_critic_flat_grads(jax.grad(l_oct8)(params))

        grads_sep14.append(np.array(g_s14))
        grads_corr.append(np.array(g_c))
        grads_oct8.append(np.array(g_o8))

    grads_sep14 = np.stack(grads_sep14, axis=0)  # (B, D)
    grads_corr = np.stack(grads_corr, axis=0)
    grads_oct8 = np.stack(grads_oct8, axis=0)
    g_exact_np = np.array(g_exact)

    def stats(G):
        mean_g = np.mean(G, axis=0)
        # Cosine similarity of mean gradient to exact
        cos_mean = float(np.dot(mean_g, g_exact_np) / (np.linalg.norm(mean_g) * np.linalg.norm(g_exact_np) + 1e-12))
        # Per-batch cosine similarities
        norms = np.linalg.norm(G, axis=1) * np.linalg.norm(g_exact_np) + 1e-12
        cos_samples = np.sum(G * g_exact_np[None, :], axis=1) / norms
        # Gradient variance: Tr(Var(g)) = E[||g - E[g]||^2]
        var_tr = float(np.mean(np.sum((G - mean_g[None, :]) ** 2, axis=1)))
        # Mean Squared Error: E[||g - g_exact||^2]
        mse = float(np.mean(np.sum((G - g_exact_np[None, :]) ** 2, axis=1)))
        return {
            "mean_cosine": cos_mean,
            "sample_cosine_mean": float(np.mean(cos_samples)),
            "sample_cosine_std": float(np.std(cos_samples)),
            "variance": var_tr,
            "mse": mse,
            "mean_norm": float(np.mean(np.linalg.norm(G, axis=1))),
        }

    return {
        "val_exact": val_exact,
        "val_exp_sep14": float(loss_exp_sep14(params)),
        "val_exp_corr": float(loss_exp_corr(params)),
        "val_exp_oct8": float(loss_exp_oct8(params)),
        "norm_g_exact": norm_g_exact,
        "stats_sep14": stats(grads_sep14),
        "stats_corr": stats(grads_corr),
        "stats_oct8": stats(grads_oct8),
    }


def run_training_comparison(env_name: str, total_timesteps: int = 200000, n_seeds: int = 3, seed: int = 42):
    """
    Runs PPO training head-to-head across the 3 critic loss formulations.
    """
    is_mc = "mountaincar" in env_name.lower()
    base_env_cfg = base_cfg.copy()
    base_env_cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": 256,
        "NUM_ENVS": 64,
        "TOTAL_TIMESTEPS": total_timesteps,
        "MINIBATCH_SIZE": 1024,
        "NUM_EPOCHS_ACTOR": 4,
        "NUM_EPOCHS_CRITIC": 4,
        "N_SEEDS": n_seeds,
        "SEED": seed,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
        "LR": 0.001 if is_mc else 0.0003,
        "ACTOR_LR": 0.001 if is_mc else 0.0003,
        "RETURN_LAMBDA": 0.0,
        "GAE_LAMBDA": 0.8,
    })

    variants = [
        {"name": "sep14", "label": "Sept 14 (Uncorrected)", "color": "#1f77b4"},
        {"name": "corrected", "label": "Oct 5 (Corrected / High Var)", "color": "#d62728"},
        {"name": "oct8", "label": "Oct 8 (Start Anchored / Exact)", "color": "#2ca02c"},
    ]

    results = {}
    for var in variants:
        v_name = var["name"]
        cfg = base_env_cfg.copy()
        cfg["E_VARIANT"] = v_name
        print(f"Training variant: {var['label']}...", end="", flush=True)
        t0 = time.time()
        train_fn = make_e_train(cfg)
        vmapped_train = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))
        rngs = jax.random.split(jax.random.PRNGKey(seed), n_seeds)
        out = vmapped_train(rngs, {"RETURN_LAMBDA": 0.0})
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
        elapsed = time.time() - t0
        print(f" Done ({elapsed:.1f}s)")
        metrics = out["metrics"]
        results[v_name] = {
            "V_start": np.array(metrics.get("V_start", 0.0)),
            "returns": np.array(metrics.get("returned_episode_returns", 0.0)),
            "value_loss": np.array(metrics.get("value_loss", 0.0)),
            "label": var["label"],
            "color": var["color"],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Sept 14 vs Corrected vs Oct 8 Sampled E variants")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Benchmark environment name")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of rollout batches for gradient analysis")
    parser.add_argument("--num_trajectories", type=int, default=64, help="Trajectories per rollout batch")
    parser.add_argument("--num_steps", type=int, default=128, help="Steps per trajectory")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total timesteps for training comparison")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds for training comparison")
    parser.add_argument("--out_dir", type=str, default="results/compare_e_variants", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print(f"HEAD-TO-HEAD SAMPLED E COMPARISON: {args.env_name}")
    print("=" * 80)

    # 1. Gradient Variance & Alignment Analysis
    print("\n--- Part 1: Empirical Gradient Variance & Alignment Analysis ---")
    cfg = base_cfg.copy()
    cfg["ENV_NAME"] = args.env_name
    env, env_params = make_env(cfg)
    evaluator = create_evaluator(cfg)
    is_mc = "mountaincar" in args.env_name.lower()
    network, network_params = initialize_network(
        jax.random.PRNGKey(42), evaluator.obs_stack.shape[1:], env, env_params, 16, 2, False
    )

    grad_analysis = analyze_gradient_variance(
        evaluator=evaluator,
        network=network,
        params=network_params,
        gamma=cfg["GAMMA"],
        num_batches=args.num_batches,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        seed=42,
    )

    print(f"Exact E Value:               {grad_analysis['val_exact']:.6f}")
    print(f"Expected Sept 14 (Uncorr):   {grad_analysis['val_exp_sep14']:.6f}")
    print(f"Expected Oct 5 (Corrected):  {grad_analysis['val_exp_corr']:.6f}")
    print(f"Expected Oct 8 (Start-Anc):  {grad_analysis['val_exp_oct8']:.6f}")
    print(f"Exact vs Oct 8 Diff:         {grad_analysis['val_exact'] - grad_analysis['val_exp_oct8']:.2e}")

    print("\nGradient Statistical Properties:")
    print(f"{'Variant':<25} | {'Mean Cosine':<12} | {'Sample Cosine':<18} | {'Variance (Tr)':<14} | {'MSE to Exact':<14}")
    print("-" * 90)
    for name, key in [
        ("Sept 14 (Uncorrected)", "stats_sep14"),
        ("Oct 5 (Corrected)", "stats_corr"),
        ("Oct 8 (Start-Anchored)", "stats_oct8"),
    ]:
        st = grad_analysis[key]
        cos_str = f"{st['sample_cosine_mean']:.4f} +/- {st['sample_cosine_std']:.4f}"
        print(f"{name:<25} | {st['mean_cosine']:<12.4f} | {cos_str:<18} | {st['variance']:<14.4e} | {st['mse']:<14.4e}")

    # 2. PPO Policy Training Comparison
    print("\n--- Part 2: PPO Policy Training Head-to-Head ---")
    train_results = run_training_comparison(
        env_name=args.env_name,
        total_timesteps=args.total_timesteps,
        n_seeds=args.n_seeds,
        seed=42,
    )

    # 3. Plotting Consolidated Dashboard
    print("\nGenerating Consolidated Comparison Figures...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    ax_val, ax_ret, ax_var = axes[0], axes[1], axes[2]

    # Panel 1: V(s0) Learning Curves
    for var_key, data in train_results.items():
        arr = data["V_start"]
        x = np.arange(arr.shape[1])
        m = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
        ax_val.plot(x, m, label=data["label"], color=data["color"], lw=2.0)
        ax_val.fill_between(x, m - sem, m + sem, color=data["color"], alpha=0.18)
    ax_val.set_title(f"{args.env_name}: True Value $V^\\pi(s_0)$", fontsize=11, fontweight="bold")
    ax_val.set_xlabel("PPO Updates", fontsize=10)
    ax_val.set_ylabel("$V^\\pi(s_0)$", fontsize=10)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(frameon=True, fontsize=9)

    # Panel 2: Episode Returns
    for var_key, data in train_results.items():
        arr = data["returns"]
        x = np.arange(arr.shape[1])
        m = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
        ax_ret.plot(x, m, label=data["label"], color=data["color"], lw=2.0)
        ax_ret.fill_between(x, m - sem, m + sem, color=data["color"], alpha=0.18)
    ax_ret.set_title(f"{args.env_name}: Episode Returns", fontsize=11, fontweight="bold")
    ax_ret.set_xlabel("PPO Updates", fontsize=10)
    ax_ret.set_ylabel("Return", fontsize=10)
    ax_ret.grid(True, alpha=0.3)
    ax_ret.legend(frameon=True, fontsize=9)

    # Panel 3: Gradient Variance & MSE Bar Chart
    var_labels = ["Sept 14\n(Uncorr)", "Oct 5\n(Corr)", "Oct 8\n(Start-Anc)"]
    variances = [
        grad_analysis["stats_sep14"]["variance"],
        grad_analysis["stats_corr"]["variance"],
        grad_analysis["stats_oct8"]["variance"],
    ]
    mses = [
        grad_analysis["stats_sep14"]["mse"],
        grad_analysis["stats_corr"]["mse"],
        grad_analysis["stats_oct8"]["mse"],
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    x_idx = np.arange(len(var_labels))
    width = 0.35
    ax_var.bar(x_idx - width/2, variances, width, label="Sample Gradient Variance", color=colors, alpha=0.85)
    ax_var.bar(x_idx + width/2, mses, width, label="Gradient MSE to Exact", color=colors, hatch="//", alpha=0.55)
    ax_var.set_xticks(x_idx)
    ax_var.set_xticklabels(var_labels, fontsize=9)
    ax_var.set_title("Gradient Variance & MSE to Exact E", fontsize=11, fontweight="bold")
    ax_var.set_ylabel("Magnitude (Log Scale)", fontsize=10)
    ax_var.set_yscale("log")
    ax_var.grid(True, alpha=0.3, axis="y")
    ax_var.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, f"compare_e_variants_{args.env_name}.png")
    pdf_path = os.path.join(args.out_dir, f"compare_e_variants_{args.env_name}.pdf")
    plt.savefig(plot_path, bbox_inches="tight", dpi=200)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved dashboard to: {plot_path}")
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
