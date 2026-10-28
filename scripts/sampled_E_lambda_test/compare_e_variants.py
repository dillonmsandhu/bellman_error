"""
scripts/sampled_E_lambda_test/compare_e_variants.py

Head-to-head comparison of Sampled E(lambda) formulations against Exact Matrix E(lambda):
1. Naive Sampling (Oct 8 Analog):
   L = w_mag * E[e^2] + w_dir * E[(e_t - e_j)^2] + w_start * E_{mu_0}[e_t^2]
   Differentiates both source state s_t and destination state s_j.
2. Forward Only:
   L = w_mag * E[e^2] + w_dir * E[(e_t - sg(e_j))^2] + w_start * E_{mu_0}[e_t^2]
   Drops backward gradient branch.
3. V_aux (Aux Version):
   L = w_mag * E[e^2] + w_dir * E[(e_t - sg(e_j))^2 + (sg(e_tilde_t) - e_tilde_j)^2] + w_start * E_{mu_0}[e_t^2]
   Surgically debiases the backward branch using auxiliary critic v_aux.

Evaluates:
- Loss value alignment against Exact Matrix E(lambda)
- Cosine similarity of sample gradients to Exact E(lambda) gradient
- Empirical gradient variance across independent rollout batches
- Gradient MSE to Exact E(lambda)
- Full PPO policy learning curves across the 3 variants
"""

from __future__ import annotations
import os
import sys
import argparse
import time
from typing import Dict, Any, List
from functools import partial

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
from core.networks import initialize_network, initialize_flax_train_state
from core.gradient_tracking import extract_critic_flat_grads, cos_similarity
from ppo.E_lambda_geometric import make_train as make_e_lambda_train


def analyze_gradient_variance(
    evaluator,
    network,
    params,
    env,
    env_params,
    gamma: float = 0.99,
    lmbda: float = 0.8,
    num_batches: int = 50,
    num_trajectories: int = 64,
    num_steps: int = 128,
    seed: int = 42,
    runway_length: int = 16,
) -> Dict[str, Any]:
    """
    Computes exact E(lambda) gradient and evaluates sample gradient distributions across
    num_batches independent rollout batches for Naive (Oct 8 analog), Forward Only, and V_aux.
    """
    S_states = evaluator.obs_stack
    N = evaluator.num_states
    start_idx = evaluator.start_idx
    P = evaluator.P

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

    # Exact Resolvent and S_lambda
    L_lmbda = jnp.linalg.inv(I - gamma * lmbda * P_pi)
    A_lmbda = D @ (I - gamma * P_pi) @ L_lmbda
    S_lmbda = 0.5 * (A_lmbda + A_lmbda.T)

    def get_e(p):
        v = network.apply(p, S_states, method=network.value).squeeze()
        return V_true - jnp.append(v, 0.0)

    # 1. Exact Matrix Loss & Gradient
    def loss_exact(p):
        e = get_e(p)
        return 0.5 * (e.T @ S_lmbda @ e)

    val_exact = float(loss_exact(params))
    g_exact = extract_critic_flat_grads(jax.grad(loss_exact)(params))
    norm_g_exact = float(jnp.linalg.norm(g_exact))
    g_exact_np = np.array(g_exact)

    # 2. Auxiliary network for v_aux
    rng = jax.random.PRNGKey(seed)
    rng, rng_aux = jax.random.split(rng)
    obs_shape = evaluator.obs_stack.shape[1:]
    aux_network, aux_params = initialize_network(
        rng_aux, obs_shape, env, env_params, 16, n_heads=1, layer_norm=False
    )
    aux_cfg = base_cfg.copy()
    aux_cfg["NUM_UPDATES"] = num_batches * 10
    aux_train_state = initialize_flax_train_state(aux_cfg, aux_network, aux_params)

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

    # Pre-train auxiliary critic on Monte Carlo rollout returns
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

    v_aux_all = aux_network.apply(aux_train_state.params, S_states, method=aux_network.value).squeeze()
    v_aux_full = jnp.append(v_aux_all, 0.0)
    v_aux_sg = jax.lax.stop_gradient(v_aux_full)

    T, B = num_steps, num_trajectories
    T_start = min(T, max(1, T - runway_length))
    sg = jax.lax.stop_gradient

    # Single-batch gradient computation for all 3 variants
    @partial(jax.jit, static_argnums=(4,))
    def compute_batch_grads(p, traj_s, returns, traj_done, lmbda_val, r_jump):
        returns_sg = sg(returns)
        gl = gamma * lmbda_val
        safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)

        u = jax.random.uniform(r_jump, shape=(T_start, B))
        jumps = jnp.where(
            gl < 1e-6,
            jnp.zeros((T_start, B), dtype=jnp.int32),
            jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
        )
        m = jnp.maximum(jumps, 0) + 1
        t_arr = jnp.arange(T_start)[:, None]
        j = jnp.minimum(t_arr + m, T)

        term_cumsum = jnp.concatenate(
            [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(traj_done.astype(jnp.int32), axis=0)],
            axis=0,
        )
        hit_true_terminal = (jnp.take_along_axis(term_cumsum, j, axis=0) - term_cumsum[:T_start]) > 0

        def scan_steps(k, term):
            k_next = jnp.where(term, 0, k + 1)
            return k_next, k

        init_k = jnp.full((B,), 10_000, dtype=jnp.int32)
        _, all_steps = jax.lax.scan(scan_steps, init_k, traj_done)

        tilde_gamma = (gamma * (1.0 - lmbda_val)) / jnp.maximum(1.0 - gl, 1e-8)
        start_trace = jnp.where(
            gl < 1e-6,
            jnp.where(all_steps[:T_start] == 0, 1.0, 0.0),
            safe_gl ** all_steps[:T_start],
        )
        start_weight = 0.25 * tilde_gamma * start_trace

        traj_s_ext = jnp.concatenate([traj_s, traj_s[-1][None]], axis=0)
        target_ext = jnp.concatenate([returns_sg, returns_sg[-1][None]], axis=0)

        dest_s = jnp.take_along_axis(traj_s_ext, j, axis=0)
        dest_target = jnp.take_along_axis(target_ext, j, axis=0)

        w_mag = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
        w_dir = 0.5 * tilde_gamma

        s_t = traj_s[:T_start]
        ret_t = returns_sg[:T_start]

        # 1. Naive Sampling Loss (Oct 8 Analog: differentiates both ends of jump)
        def l_naive(p_curr):
            v = jnp.append(network.apply(p_curr, S_states, method=network.value).squeeze(), 0.0)
            es = ret_t - v[s_t]
            ej = jnp.where(hit_true_terminal, 0.0, dest_target - v[dest_s])
            diff_sq = jnp.where(hit_true_terminal, es ** 2, (es - ej) ** 2)
            return w_mag * jnp.mean(es ** 2) + w_dir * jnp.mean(diff_sq) + jnp.mean(start_weight * (es ** 2))

        # 2. Aux Version Loss (debiased surrogate via v_aux)
        def l_aux(p_curr):
            v = jnp.append(network.apply(p_curr, S_states, method=network.value).squeeze(), 0.0)
            es = ret_t - v[s_t]
            ej = jnp.where(hit_true_terminal, 0.0, dest_target - v[dest_s])
            e_tilde_s = v_aux_sg[s_t] - v[s_t]
            e_tilde_j = jnp.where(hit_true_terminal, 0.0, v_aux_sg[dest_s] - v[dest_s])
            diff_sq = jnp.where(
                hit_true_terminal,
                es ** 2,
                (es - sg(ej)) ** 2 + (sg(e_tilde_s) - e_tilde_j) ** 2,
            )
            return w_mag * jnp.mean(es ** 2) + w_dir * jnp.mean(diff_sq) + jnp.mean(start_weight * (es ** 2))

        # 3. Forward Only Loss
        def l_fwd(p_curr):
            v = jnp.append(network.apply(p_curr, S_states, method=network.value).squeeze(), 0.0)
            es = ret_t - v[s_t]
            ej = jnp.where(hit_true_terminal, 0.0, dest_target - v[dest_s])
            diff_sq = jnp.where(hit_true_terminal, es ** 2, (es - sg(ej)) ** 2)
            return w_mag * jnp.mean(es ** 2) + w_dir * jnp.mean(diff_sq) + jnp.mean(start_weight * (es ** 2))

        g_n = extract_critic_flat_grads(jax.grad(l_naive)(p))
        g_a = extract_critic_flat_grads(jax.grad(l_aux)(p))
        g_f = extract_critic_flat_grads(jax.grad(l_fwd)(p))
        return g_n, g_a, g_f

    # Sample over num_batches independent rollout draws
    grads_naive = []
    grads_aux = []
    grads_fwd = []

    batch_keys = jax.random.split(rng, num_batches)
    for bk in batch_keys:
        r_roll, r_jump = jax.random.split(bk)
        traj_s, _, traj_r, traj_done, final_s = sample_rollout(r_roll)
        returns = compute_returns(traj_r, traj_done, final_s)

        gn, ga, gf = compute_batch_grads(params, traj_s, returns, traj_done, lmbda, r_jump)
        grads_naive.append(np.array(gn))
        grads_aux.append(np.array(ga))
        grads_fwd.append(np.array(gf))

    G_n = np.stack(grads_naive, axis=0)
    G_a = np.stack(grads_aux, axis=0)
    G_f = np.stack(grads_fwd, axis=0)

    mean_n = np.mean(G_n, axis=0)
    mean_a = np.mean(G_a, axis=0)
    mean_f = np.mean(G_f, axis=0)

    def compute_stats(G, mean_g):
        cos_sims = [float(cos_similarity(g, g_exact_np)) for g in G]
        return {
            "mean_cosine": float(cos_similarity(mean_g, g_exact_np)),
            "sample_cosine_mean": float(np.mean(cos_sims)),
            "sample_cosine_std": float(np.std(cos_sims)),
            "variance": float(np.trace(np.cov(G, rowvar=False))),
            "mse": float(np.mean((mean_g - g_exact_np) ** 2)),
            "norm_ratio": float(np.linalg.norm(mean_g)) / max(1e-8, norm_g_exact),
        }

    return {
        "val_exact": val_exact,
        "norm_g_exact": norm_g_exact,
        "stats_naive": compute_stats(G_n, mean_n),
        "stats_aux": compute_stats(G_a, mean_a),
        "stats_fwd": compute_stats(G_f, mean_f),
    }


def run_training_comparison(
    env_name: str = "FourRooms-misc",
    lmbda: float = 0.8,
    total_timesteps: int = 100000,
    n_seeds: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes full PPO policy learning across:
    1. Naive (Oct 8 Analog): E_LAMBDA_VARIANT='naive'
    2. Aux Version (v_aux):  E_LAMBDA_VARIANT='aux'
    3. Forward Only:         E_LAMBDA_VARIANT='fwd'
    """
    is_mc = "mountaincar" in env_name.lower()
    base_env_cfg = base_cfg.copy()
    base_env_cfg.update({
        "ENV_NAME": env_name,
        "TOTAL_TIMESTEPS": total_timesteps,
        "NUM_STEPS": 64,
        "NUM_ENVS": 8,
        "MINIBATCH_SIZE": 64,
        "NUM_EPOCHS": 4,
        "GAMMA": 0.99,
        "VALUE_LAMBDA": lmbda,
        "RETURN_LAMBDA": 1.0,
        "GAE_LAMBDA": 0.8,
        "CRITIC_LR": 0.001 if is_mc else 0.0003,
        "ACTOR_LR": 0.001 if is_mc else 0.0003,
        "RUNWAY_LENGTH": 16,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
    })

    variants = [
        {"name": "naive", "label": "Naive (Oct 8 Analog)", "color": "#2ca02c"},
        {"name": "fwd", "label": "Forward Only", "color": "#00bcd4"},
        {"name": "aux", "label": "V_aux (Aux Version)", "color": "#9467bd"},
    ]

    results = {}
    for var in variants:
        v_name = var["name"]
        cfg = base_env_cfg.copy()
        cfg["E_LAMBDA_VARIANT"] = v_name
        print(f"Training variant: {var['label']} (lambda={lmbda})...", end="", flush=True)
        t0 = time.time()
        train_fn = make_e_lambda_train(cfg)
        vmapped_train = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))
        rngs = jax.random.split(jax.random.PRNGKey(seed), n_seeds)
        out = vmapped_train(rngs, None)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
        elapsed = time.time() - t0
        print(f" Done ({elapsed:.1f}s)")
        metrics = out["metrics"]
        results[var["label"]] = {
            "V_start": np.array(metrics.get("V_start", 0.0)),
            "returns": np.array(metrics.get("returned_episode_returns", 0.0)),
            "value_loss": np.array(metrics.get("value_loss", 0.0)),
            "label": var["label"],
            "color": var["color"],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Naive (Oct 8) vs Forward Only vs V_aux Sampled E(lambda) variants")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Benchmark environment name")
    parser.add_argument("--value_lambda", type=float, default=0.8, help="Trace parameter lambda for E(lambda)")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of rollout batches for gradient analysis")
    parser.add_argument("--num_trajectories", type=int, default=64, help="Trajectories per rollout batch")
    parser.add_argument("--num_steps", type=int, default=128, help="Steps per trajectory")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total timesteps for training comparison")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds for training comparison")
    parser.add_argument("--out_dir", type=str, default="results/sampled_E_lambda_test", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print(f"HEAD-TO-HEAD SAMPLED E(lambda={args.value_lambda}) COMPARISON: {args.env_name}")
    print("=" * 80)

    # 1. Gradient Variance & Alignment Analysis
    print("\n--- Part 1: Empirical Gradient Variance & Alignment Analysis ---")
    cfg = base_cfg.copy()
    cfg["ENV_NAME"] = args.env_name
    env, env_params = make_env(cfg)
    evaluator = create_evaluator(cfg, env, env_params)
    network, network_params = initialize_network(
        jax.random.PRNGKey(42), evaluator.obs_stack.shape[1:], env, env_params, 16, 2, False
    )

    grad_analysis = analyze_gradient_variance(
        evaluator=evaluator,
        network=network,
        params=network_params,
        env=env,
        env_params=env_params,
        gamma=cfg["GAMMA"],
        lmbda=args.value_lambda,
        num_batches=args.num_batches,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        seed=42,
    )

    print(f"Exact E(lambda) Value:       {grad_analysis['val_exact']:.6f}")
    print(f"Exact Gradient Norm:         {grad_analysis['norm_g_exact']:.6f}")

    print("\nGradient Statistical Properties:")
    print(f"{'Variant':<24} | {'Mean Cosine':<12} | {'Sample Cosine':<18} | {'Variance (Tr)':<14} | {'MSE to Exact':<14}")
    print("-" * 90)
    for name, key in [
        ("Naive (Oct 8 Analog)", "stats_naive"),
        ("Forward Only", "stats_fwd"),
        ("V_aux (Aux Version)", "stats_aux"),
    ]:
        st = grad_analysis[key]
        cos_str = f"{st['sample_cosine_mean']:.4f} +/- {st['sample_cosine_std']:.4f}"
        print(f"{name:<24} | {st['mean_cosine']:<12.4f} | {cos_str:<18} | {st['variance']:<14.4e} | {st['mse']:<14.4e}")

    # 2. PPO Policy Training Comparison
    print("\n--- Part 2: PPO Policy Training Head-to-Head ---")
    train_results = run_training_comparison(
        env_name=args.env_name,
        lmbda=args.value_lambda,
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
    ax_val.set_title(f"{args.env_name}: True Value $V^\\pi(s_0)$ ($\\lambda={args.value_lambda}$)", fontsize=11, fontweight="bold")
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
    ax_ret.set_title(f"{args.env_name}: Episode Returns ($\\lambda={args.value_lambda}$)", fontsize=11, fontweight="bold")
    ax_ret.set_xlabel("PPO Updates", fontsize=10)
    ax_ret.set_ylabel("Return", fontsize=10)
    ax_ret.grid(True, alpha=0.3)
    ax_ret.legend(frameon=True, fontsize=9)

    # Panel 3: Gradient Variance & MSE Bar Chart
    var_labels = ["Naive (Oct 8)", "Forward Only", "V_aux"]
    variances = [
        grad_analysis["stats_naive"]["variance"],
        grad_analysis["stats_fwd"]["variance"],
        grad_analysis["stats_aux"]["variance"],
    ]
    mses = [
        grad_analysis["stats_naive"]["mse"],
        grad_analysis["stats_fwd"]["mse"],
        grad_analysis["stats_aux"]["mse"],
    ]
    colors = ["#2ca02c", "#00bcd4", "#9467bd"]
    x_idx = np.arange(len(var_labels))
    width = 0.35
    ax_var.bar(x_idx - width/2, variances, width, label="Sample Gradient Variance", color=colors, alpha=0.85)
    ax_var.bar(x_idx + width/2, mses, width, label="Gradient MSE to Exact", color=colors, hatch="//", alpha=0.55)
    ax_var.set_xticks(x_idx)
    ax_var.set_xticklabels(var_labels, fontsize=10, fontweight="bold")
    ax_var.set_title(f"Gradient Variance & MSE to Exact E($\\lambda={args.value_lambda}$)", fontsize=11, fontweight="bold")
    ax_var.set_ylabel("Magnitude (Log Scale)", fontsize=10)
    ax_var.set_yscale("log")
    ax_var.grid(True, alpha=0.3, axis="y")
    ax_var.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, f"compare_e_lambda_{args.env_name}.png")
    pdf_path = os.path.join(args.out_dir, f"compare_e_lambda_{args.env_name}.pdf")
    plt.savefig(plot_path, bbox_inches="tight", dpi=200)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved dashboard to: {plot_path}")
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
