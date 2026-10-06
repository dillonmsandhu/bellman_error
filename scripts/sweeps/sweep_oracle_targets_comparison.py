#!/usr/bin/env python3
"""
scripts/sweeps/sweep_oracle_targets_comparison.py

Validates the sampled E-loss formulation via Semi-Sampled "Oracle Targets" (Approach 2):
At each policy update step, computes and compares three gradients:
1. g^*: Exact expected Bellman error gradient (evaluated analytically across all states).
2. g_hat_oracle: Sampled transition gradient with ORACLE targets V^pi(s) (zero return truncation bias).
3. g_hat_empirical: Standard sampled transition gradient with empirical rollout targets G^T.

If cos(g_hat_oracle, g^*) ≈ 1.0, the sampled transition loss code is mathematically verified.
Any discrepancy in g_hat_empirical is proven to stem purely from return bootstrapping G^T != V^pi.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"

import core.config as default_cfg
import core.helpers as helpers
import core.networks as networks
from core.gradient_tracking import (
    compute_all_exact_critic_gradients,
    extract_critic_flat_grads,
    cos_similarity,
)
from ppo.sampled_E import Transition


ALL_ENVS = [
    "fourrooms-dense",
    "FourRooms-misc",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify sampled E accuracy via semi-sampled Oracle Targets"
    )
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Target environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-6)")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout steps per env T")
    parser.add_argument("--num_envs", type=int, default=128, help="Parallel rollout environments B")
    parser.add_argument("--num_updates", type=int, default=40, help="Number of policy update steps")
    parser.add_argument("--num_seeds", type=int, default=4, help="Random seeds to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--layer_norm", action="store_true", help="Enable LayerNorm on critic")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for plots and NPZ")
    return parser.parse_args()


def run_oracle_comparison_single_seed(
    seed: int,
    env_name: str,
    num_steps: int,
    num_envs: int,
    num_updates: int,
    layer_norm: bool,
) -> Dict[str, np.ndarray]:
    """Runs training while computing g^*, g_hat_oracle, and g_hat_empirical at every step."""
    cfg = default_cfg.config.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": num_steps,
        "NUM_ENVS": num_envs,
        "TOTAL_TIMESTEPS": num_steps * num_envs * num_updates,
        "NUM_UPDATES": num_updates,
        "LAYER_NORM": layer_norm,
        "LR": 0.001,
        "ACTOR_LR": 0.001,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
    })

    env, env_params = helpers.make_env(cfg)
    evaluator = helpers.initialize_evaluator(cfg, env, env_params)
    obs_shape = env.observation_space(env_params).shape

    gamma = cfg["GAMMA"]
    k = cfg["k"]

    rng = jax.random.PRNGKey(seed)
    rng, net_rng = jax.random.split(rng)
    network, network_params = networks.initialize_network(
        net_rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=layer_norm
    )
    train_state = networks.initialize_flax_train_state(cfg, network, network_params)

    # Precompute state lookup structures for oracle targets
    n_states = len(evaluator.obs_stack)
    flat_stack = evaluator.obs_stack.reshape(n_states, -1)
    num_actions = evaluator.num_actions

    # Tracking arrays across updates
    history = {
        "rho_oracle_exact": [],
        "rho_empirical_exact": [],
        "rho_empirical_oracle": [],
        "sq_err_total": [],
        "sq_err_oracle": [],
        "sq_err_target_bias": [],
        "policy_return": [],
    }

    # Initial environment reset
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)

    for update_idx in range(num_updates):
        # -------------------------------------------------------------
        # 1. Exact Expected Gradient g* (DIRICHLET FORM)
        # -------------------------------------------------------------
        g_exact_E, _, _, v_start, _, _ = compute_all_exact_critic_gradients(
            train_state, evaluator, network, gamma
        )

        # -------------------------------------------------------------
        # 2. Collect 1 Rollout Batch
        # -------------------------------------------------------------
        def _step(state, unused):
            t_state, e_state, obs, r = state
            r, _r1, _r2 = jax.random.split(r, 3)
            pi, val = network.apply(t_state.params, obs)
            act = pi.sample(seed=_r1)
            r_step = jax.random.split(_r2, num_envs)
            next_obs, next_e_state, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                r_step, e_state, act, env_params
            )
            real_next = info["real_next_obs"]
            next_val = network.apply(t_state.params, real_next, method=network.value)
            trans = Transition(done, act, val, next_val, rew, pi.log_prob(act), obs, real_next, 0.0, info)
            return (t_state, next_e_state, next_obs, r), trans

        rng, step_rng = jax.random.split(rng)
        (train_state, env_state, obsv, _), traj_batch = jax.lax.scan(
            _step, (train_state, env_state, obsv, step_rng), None, num_steps
        )

        # -------------------------------------------------------------
        # 3. Empirical Return Targets G_t
        # -------------------------------------------------------------
        adv, targets = helpers.calculate_gae(traj_batch, gamma, cfg["RETURN_LAMBDA"])
        is_timeout = traj_batch.info["is_timeout"]
        true_terminal = traj_batch.done & ~is_timeout

        next_targets = jnp.roll(targets, shift=-1, axis=0).at[-1].set(traj_batch.next_value[-1])
        next_targets = jnp.where(is_timeout, traj_batch.next_value, next_targets)
        next_targets = jnp.where(true_terminal, 0.0, next_targets)

        # -------------------------------------------------------------
        # 4. Ground-Truth Oracle Targets V^pi(s)
        # -------------------------------------------------------------
        pi_dist, _ = network.apply(train_state.params, evaluator.obs_stack)
        if hasattr(pi_dist, "probs"):
            old_pi = pi_dist.probs
        else:
            action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
            log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
            old_pi = jax.nn.softmax(log_probs, axis=-1)

        term_pi = jnp.ones([1, num_actions], dtype=old_pi.dtype) / num_actions
        old_pi_full = jnp.vstack([old_pi, term_pi])
        V_true = evaluator.compute_true_values_raw(old_pi_full)
        V_active = V_true[:n_states]

        # Fast observation-to-state lookup
        flat_obs = traj_batch.obs.reshape(-1, flat_stack.shape[-1])
        dists = jnp.sum((flat_obs[:, None, :] - flat_stack[None, :, :]) ** 2, axis=-1)
        curr_indices = jnp.argmin(dists, axis=-1)
        v_oracle_curr = V_active[curr_indices].reshape(traj_batch.obs.shape[:-len(evaluator.obs_stack.shape) + 1])

        flat_next_obs = traj_batch.next_obs.reshape(-1, flat_stack.shape[-1])
        next_dists = jnp.sum((flat_next_obs[:, None, :] - flat_stack[None, :, :]) ** 2, axis=-1)
        next_indices = jnp.argmin(next_dists, axis=-1)
        v_oracle_next = jnp.where(
            true_terminal,
            0.0,
            V_active[next_indices].reshape(traj_batch.next_obs.shape[:-len(evaluator.obs_stack.shape) + 1]),
        )

        # -------------------------------------------------------------
        # 5. Compute Empirical vs. Oracle Gradients
        # -------------------------------------------------------------
        def empirical_critic_loss(params):
            v_i = network.apply(params, traj_batch.obs, method=network.value)
            v_j = network.apply(params, traj_batch.next_obs, method=network.value)
            loss_val, _, _ = helpers.e_critic_loss(
                v_i, targets, v_j, next_targets, true_terminal, gamma
            )
            return loss_val

        def oracle_critic_loss(params):
            v_i = network.apply(params, traj_batch.obs, method=network.value)
            v_j = network.apply(params, traj_batch.next_obs, method=network.value)
            loss_val, _, _ = helpers.e_critic_loss(
                v_i, v_oracle_curr, v_j, v_oracle_next, true_terminal, gamma
            )
            return loss_val

        g_empirical = extract_critic_flat_grads(jax.grad(empirical_critic_loss)(train_state.params))
        g_oracle = extract_critic_flat_grads(jax.grad(oracle_critic_loss)(train_state.params))

        # Metrics
        rho_ora_exact = float(cos_similarity(g_oracle, g_exact_E))
        rho_emp_exact = float(cos_similarity(g_empirical, g_exact_E))
        rho_emp_ora = float(cos_similarity(g_empirical, g_oracle))

        sq_err_tot = float(jnp.sum((g_empirical - g_exact_E) ** 2))
        sq_err_ora = float(jnp.sum((g_oracle - g_exact_E) ** 2))
        sq_err_tgt = float(jnp.sum((g_empirical - g_oracle) ** 2))

        history["rho_oracle_exact"].append(rho_ora_exact)
        history["rho_empirical_exact"].append(rho_emp_exact)
        history["rho_empirical_oracle"].append(rho_emp_ora)
        history["sq_err_total"].append(sq_err_tot)
        history["sq_err_oracle"].append(sq_err_ora)
        history["sq_err_target_bias"].append(sq_err_tgt)
        history["policy_return"].append(float(v_start))

        # -------------------------------------------------------------
        # 6. Apply Standard PPO Update Step
        # -------------------------------------------------------------
        def full_loss(params):
            pi_m = network.apply(params, traj_batch.obs, method=network.policy)
            ratio = jnp.exp(pi_m.log_prob(traj_batch.action) - traj_batch.log_prob)
            adv_norm = helpers.post_process_advantage(adv, cfg)
            surr1 = ratio * adv_norm
            surr2 = jnp.clip(ratio, 1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"]) * adv_norm
            actor_loss = -jnp.minimum(surr1, surr2).mean()

            v_loss = empirical_critic_loss(params)
            entropy = pi_m.entropy().mean()
            return actor_loss + cfg["VF_COEF"] * v_loss - cfg["ENT_COEF"] * entropy

        grads = jax.grad(full_loss)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)

    return {k: np.array(v) for k, v in history.items()}


def plot_comparison(
    env_name: str,
    results: Dict[str, np.ndarray],
    num_updates: int,
    num_seeds: int,
    out_dir: str,
):
    """Renders alignment and error decomposition figures comparing Oracle vs Empirical targets."""
    x = np.arange(1, num_updates + 1)

    # -------------------------------------------------------------
    # Figure 1: Cosine Similarity Alignment (Oracle vs Empirical vs Exact)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)

    m_ora = np.mean(results["rho_oracle_exact"], axis=0)
    s_ora = np.std(results["rho_oracle_exact"], axis=0) / np.sqrt(num_seeds)

    m_emp = np.mean(results["rho_empirical_exact"], axis=0)
    s_emp = np.std(results["rho_empirical_exact"], axis=0) / np.sqrt(num_seeds)

    m_emp_ora = np.mean(results["rho_empirical_oracle"], axis=0)
    s_emp_ora = np.std(results["rho_empirical_oracle"], axis=0) / np.sqrt(num_seeds)

    # Oracle Target Alignment (The Key Test of Loss Code Correctness)
    ax.plot(
        x,
        m_ora,
        color="#27ae60",
        lw=2.5,
        label=r"$\rho(\hat{\mathbf{g}}_{\mathrm{oracle}}, \mathbf{g}^*)$ [Oracle Targets $V^\pi$]",
        zorder=5,
    )
    ax.fill_between(x, m_ora - s_ora, m_ora + s_ora, color="#27ae60", alpha=0.18, zorder=4)

    # Empirical Target Alignment
    ax.plot(
        x,
        m_emp,
        color="#2980b9",
        lw=2.2,
        label=r"$\rho(\hat{\mathbf{g}}_{\mathrm{empirical}}, \mathbf{g}^*)$ [Empirical Targets $G^T$]",
        zorder=3,
    )
    ax.fill_between(x, m_emp - s_emp, m_emp + s_emp, color="#2980b9", alpha=0.15, zorder=2)

    # Empirical vs Oracle Alignment
    ax.plot(
        x,
        m_emp_ora,
        color="#8e44ad",
        lw=1.8,
        ls="--",
        label=r"$\rho(\hat{\mathbf{g}}_{\mathrm{empirical}}, \hat{\mathbf{g}}_{\mathrm{oracle}})$ [Target Agreement]",
        zorder=3,
    )

    ax.axhline(1.0, color="#27ae60", ls=":", lw=1.2, alpha=0.7)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.4)

    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Validation of Sampled E: Oracle vs. Empirical Targets ({env_name})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right")

    fig.tight_layout()
    p1_pdf = os.path.join(out_dir, f"{env_name}_oracle_vs_empirical_alignment.pdf")
    p1_png = os.path.join(out_dir, f"{env_name}_oracle_vs_empirical_alignment.png")
    fig.savefig(p1_pdf, bbox_inches="tight")
    fig.savefig(p1_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {p1_pdf}")

    # -------------------------------------------------------------
    # Figure 2: Error Decomposition (Target Truncation vs Transition Sampling)
    # -------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.0, 5.0), dpi=300)

    err_tot = np.mean(results["sq_err_total"], axis=0)
    err_ora = np.mean(results["sq_err_oracle"], axis=0)
    err_tgt = np.mean(results["sq_err_target_bias"], axis=0)

    ax2.plot(x, err_tot, color="#e74c3c", lw=2.2, label=r"Total Gradient MSE $\|\hat{\mathbf{g}}_{\mathrm{emp}} - \mathbf{g}^*\|^2$")
    ax2.plot(x, err_tgt, color="#f39c12", lw=2.0, ls="--", label=r"Target Truncation Error $\|\hat{\mathbf{g}}_{\mathrm{emp}} - \hat{\mathbf{g}}_{\mathrm{ora}}\|^2$")
    ax2.plot(x, err_ora, color="#2ecc71", lw=2.0, ls="-.", label=r"Pure Transition Sampling Error $\|\hat{\mathbf{g}}_{\mathrm{ora}} - \mathbf{g}^*\|^2$")

    ax2.set_yscale("log")
    ax2.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Squared Parameter Gradient Error (Log)", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Disentangled Gradient Error Decomposition ({env_name})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9.5, loc="upper right")

    fig2.tight_layout()
    p2_pdf = os.path.join(out_dir, f"{env_name}_oracle_error_decomposition.pdf")
    p2_png = os.path.join(out_dir, f"{env_name}_oracle_error_decomposition.png")
    fig2.savefig(p2_pdf, bbox_inches="tight")
    fig2.savefig(p2_png, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [Saved] {p2_pdf}")


def main():
    args = parse_args()

    if args.env_idx is not None:
        env_name = ALL_ENVS[args.env_idx]
    else:
        env_name = args.env_name

    out_dir = args.out_dir or os.path.join("results", "sweeps", "oracle_targets_comparison", env_name)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{env_name}_oracle_comparison.npz")

    print("=" * 70)
    print("SEMI-SAMPLED ORACLE TARGETS VALIDATION SWEEP (APPROACH 2)")
    print(f"Environment:    {env_name}")
    print(f"Rollout Shape:  B={args.num_envs} x T={args.num_steps} (N={args.num_envs * args.num_steps})")
    print(f"Updates:        {args.num_updates}")
    print(f"Seeds:          {args.num_seeds} (base: {args.seed})")
    print(f"LayerNorm:      {args.layer_norm}")
    print(f"Output Dir:     {out_dir}")
    print("=" * 70)

    # Collect multi-seed runs
    seed_runs = []
    t_start = time.time()
    for s_idx in range(args.num_seeds):
        curr_seed = args.seed + s_idx * 100
        print(f"\n--> Running Seed {s_idx + 1}/{args.num_seeds} (seed={curr_seed})...")
        res = run_oracle_comparison_single_seed(
            seed=curr_seed,
            env_name=env_name,
            num_steps=args.num_steps,
            num_envs=args.num_envs,
            num_updates=args.num_updates,
            layer_norm=args.layer_norm,
        )
        print(
            f"    Mean rho(g_oracle, g*):    {res['rho_oracle_exact'].mean():.4f}\n"
            f"    Mean rho(g_emp, g*):       {res['rho_empirical_exact'].mean():.4f}\n"
            f"    Mean rho(g_emp, g_oracle): {res['rho_empirical_oracle'].mean():.4f}"
        )
        seed_runs.append(res)

    elapsed = time.time() - t_start
    print(f"\nAll seeds completed in {elapsed:.1f}s.")

    # Stack results across seeds: (num_seeds, num_updates)
    stacked = {}
    for k in seed_runs[0].keys():
        stacked[k] = np.stack([run[k] for run in seed_runs], axis=0)

    # Save to NPZ
    np.savez_compressed(npz_path, **stacked)
    print(f"[NPZ Saved] {npz_path}")

    # Generate plots
    plot_comparison(
        env_name=env_name,
        results=stacked,
        num_updates=args.num_updates,
        num_seeds=args.num_seeds,
        out_dir=out_dir,
    )

    print("\n" + "=" * 70)
    print("ORACLE TARGET VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
