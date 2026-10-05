"""
run_sampled_vs_exact_e.py

Compares the empirical sampled E critic update gradient to the exact theoretical
closed-form expected E gradient on actual PPO training trajectories.

Key Methodological Design:
1. In on-policy RL, an update collects a rollout of size N = B * T (num_envs * num_steps),
   partitioned into K minibatches of size M = N / K.
2. Individual minibatch gradients g_mb^(m) have variance dependent on M = (B * T) / K.
3. However, the average gradient across minibatches:
       g_bar_update = (1/K) * sum_{m=1}^K g_mb^(m)
   is identically the full-batch gradient over all N = B * T transitions in that update!
4. Comparing g_bar_update to g_exact evaluates the actual parameter update step,
   and its variance depends purely on the total batch size N = B * T and rollout length T,
   independent of the arbitrary choice of minibatch split K.

Supports:
- Tabular environments: FourRooms-misc, fourrooms-dense, eightrooms-misc, eightrooms-dense, whirlpool-misc
- MountainCar-v0 (discretized 32x32 exact model)
- Trajectory tracking and 2D parameter sweeps over (num_steps, num_envs)
- Iso-batch variance scaling curves (Var vs. N = B * T)
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax import struct

from core import helpers, networks
from ppo.sampled_E import Transition


def extract_critic_flat_grads(grads) -> jnp.ndarray:
    """Extracts and flattens only the critic parameter gradients from a PyTree."""
    if not isinstance(grads, dict) or "params" not in grads:
        leaves = jax.tree_util.tree_leaves(grads)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

    critic_subtrees = {
        k: v for k, v in grads["params"].items() if "critic" in k
    }
    if not critic_subtrees:
        leaves = jax.tree_util.tree_leaves(grads["params"])
    else:
        leaves = jax.tree_util.tree_leaves(critic_subtrees)
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])


def compute_exact_e_loss_and_grad(
    train_state: TrainState,
    evaluator,
    network,
    gamma: float,
) -> Tuple[float, jnp.ndarray, float]:
    """
    Computes exact closed-form E-loss, its true parameter gradient g_exact,
    and current policy expected return V_pi(s_0) using exact tabular dynamics.
    """
    S_states = evaluator.obs_stack
    n_actions = evaluator.num_actions

    # 1. Extract current policy matrix pi(s, a) over all states
    pi_dist, _ = network.apply(train_state.params, S_states)
    if hasattr(pi_dist, "probs"):
        old_pi = pi_dist.probs
    else:
        action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
        old_log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
        old_pi = jax.nn.softmax(old_log_probs, axis=-1)

    terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
    old_pi_full = jnp.vstack([old_pi, terminal_policy])

    # 2. True values and stationary distribution
    V_true = evaluator.compute_true_values_raw(old_pi_full)
    start_val = float(V_true[evaluator.start_idx])

    mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
    mu_full = jnp.append(mu, 0.0)
    D = jnp.diag(mu_full)

    P = evaluator.P
    I = jnp.eye(evaluator.num_total_states)
    P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
    A_mat = D @ (I - gamma * P_pi)
    S_mat = 0.5 * (A_mat + A_mat.T)

    # 3. Exact critic loss function on parameters
    def exact_loss_fn(params):
        v = network.apply(params, S_states, method=network.value).squeeze()
        v_full = jnp.append(v, 0.0)
        diff = V_true - v_full
        return diff.T @ S_mat @ diff

    loss_exact, grads_exact = jax.value_and_grad(exact_loss_fn)(train_state.params)
    g_exact_critic = extract_critic_flat_grads(grads_exact)

    return float(loss_exact), g_exact_critic, start_val


def run_comparison_single_setting(
    env_name: str = "FourRooms-misc",
    num_updates: int = 40,
    num_envs: int = 32,
    num_steps: int = 64,
    num_minibatches: int = 4,
    lr: float = 3e-4,
    seed: int = 42,
    use_visual_obs: bool = False,
    out_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Runs PPO with sampled E critic and logs exact vs. sampled critic gradients at every update.
    Uses JIT compilation for rollout and update routines for high execution efficiency.
    """
    config = {
        "ENV_NAME": env_name,
        "NETWORK_TYPE": "mlp",
        "LAYER_NORM": False,
        "GAMMA": 0.99,
        "LR": lr,
        "ACTOR_LR": lr,
        "NUM_EPOCHS": 4,
        "NUM_MINIBATCHES": num_minibatches,
        "NUM_UPDATES": num_updates,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "GAE_LAMBDA": 0.95,
        "RETURN_LAMBDA": 1.0,  # Pure MC returns for faithful E(0) error evaluation
        "CALC_TRUE_VALUES": True,
        "USE_VISUAL_OBS": use_visual_obs,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,
        "k": 32,
        "MAX_STEPS_IN_EPISODE": 1000,
        "USE_TABULAR_SIMULATOR": True,
    }

    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    if evaluator is None:
        evaluator = helpers.create_evaluator(config)

    obs_shape = env.observation_space(env_params).shape
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    return_lambda = config["RETURN_LAMBDA"]

    rng = jax.random.PRNGKey(seed)
    rng_net, rng_run = jax.random.split(rng)

    network, network_params = networks.initialize_network(
        rng_net, obs_shape, env, env_params, k=config["k"], n_heads=2, layer_norm=config["LAYER_NORM"]
    )
    train_state = networks.initialize_flax_train_state(config, network, network_params)

    # Initialize parallel environments
    rng_run, _rng = jax.random.split(rng_run)
    reset_rng = jax.random.split(_rng, num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    # -------------------------------------------------------------------------
    # JIT-compiled step functions
    # -------------------------------------------------------------------------
    @jax.jit
    def rollout_and_batch_fn(t_state, e_state, o, r):
        def _env_step(carry, _):
            es, cur_o, cur_r = carry
            cur_r, r_act = jax.random.split(cur_r)
            pi, val = network.apply(t_state.params, cur_o)
            act = pi.sample(seed=r_act)
            log_p = pi.log_prob(act)
            cur_r, r_step_base = jax.random.split(cur_r)
            r_step = jax.random.split(r_step_base, num_envs)
            next_o, next_es, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                r_step, es, act, env_params
            )
            real_next_o = info.get("real_next_obs", next_o)
            next_v = network.apply(t_state.params, real_next_o, method=network.value).squeeze()
            return (next_es, next_o, cur_r), (cur_o, real_next_o, rew, done, val, next_v, act, log_p, info)

        r, r_scan = jax.random.split(r)
        (e_state, o, _), (
            obs_seq, next_obs_seq, rew_seq, done_seq, val_seq, next_val_seq, act_seq, log_p_seq, info_seq
        ) = jax.lax.scan(_env_step, (e_state, o, r_scan), None, num_steps)

        traj_batch = Transition(
            done=done_seq,
            action=act_seq,
            value=val_seq,
            next_value=next_val_seq,
            reward=rew_seq,
            log_prob=log_p_seq,
            obs=obs_seq,
            next_obs=next_obs_seq,
            next_target=jnp.zeros_like(val_seq),
            info=info_seq,
        )

        advantages, _ = helpers.calculate_gae(traj_batch, gamma, gae_lambda)
        _, targets = helpers.calculate_gae(traj_batch, gamma, return_lambda)

        is_timeout = traj_batch.info["is_timeout"]
        true_terminal = traj_batch.done & ~is_timeout

        next_targets = jnp.roll(targets, shift=-1, axis=0)
        next_targets = next_targets.at[-1].set(traj_batch.next_value[-1])
        next_targets = jnp.where(is_timeout, traj_batch.next_value, next_targets)
        next_targets = jnp.where(true_terminal, 0.0, next_targets)

        batch = (
            traj_batch.obs,
            traj_batch.action,
            traj_batch.log_prob,
            traj_batch.next_obs,
            true_terminal,
            next_targets,
            advantages,
            targets,
        )
        r, r_batch = jax.random.split(r)
        mbs = helpers.shuffle_and_batch(r_batch, batch, num_minibatches)
        return e_state, o, mbs, r

    @jax.jit
    def ppo_update_fn(t_state, mbs):
        def _update_epoch(update_state, unused):
            def _update_minbatch(cur_t_state, mb_info):
                obs_mb, action_mb, log_prob_mb, next_obs_mb, true_term_mb, next_tgt_mb, advantages_mb, targets_mb = mb_info

                def loss_fn(params):
                    pi = network.apply(params, obs_mb, method=network.policy)
                    log_prob = pi.log_prob(action_mb)
                    entropy = pi.entropy().mean()
                    ratio = jnp.exp(log_prob - log_prob_mb)
                    adv_norm = helpers.post_process_advantage(advantages_mb, config)
                    surr1 = ratio * adv_norm
                    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
                    actor_loss = -jnp.minimum(surr1, surr2).mean()

                    v_i = network.apply(params, obs_mb, method=network.value)
                    v_j = network.apply(params, next_obs_mb, method=network.value)
                    value_loss, _, _ = helpers.e_critic_loss(
                        v_i, targets_mb, v_j, next_tgt_mb, true_term_mb, gamma
                    )
                    return actor_loss + config["VF_COEF"] * value_loss - entropy * config["ENT_COEF"]

                grads = jax.grad(loss_fn)(cur_t_state.params)
                cur_t_state = cur_t_state.apply_gradients(grads=grads)
                return cur_t_state, None

            cur_t_state, cur_mbs = update_state
            cur_t_state, _ = jax.lax.scan(_update_minbatch, cur_t_state, cur_mbs)
            return (cur_t_state, cur_mbs), None

        (t_state, _), _ = jax.lax.scan(_update_epoch, (t_state, mbs), None, config["NUM_EPOCHS"])
        return t_state

    # Logging storage
    history = {
        "update": [],
        "policy_return": [],
        "exact_loss": [],
        "sampled_loss": [],
        "exact_grad_norm": [],
        "sampled_update_grad_norm": [],
        "cosine_sim_update": [],
        "rel_err_update": [],
        "update_variance": [],
        "update_sq_err": [],
        "mb_variance": [],
        "norm_ratio_update": [],
        # Minibatch scatter points
        "mb_updates": [],
        "mb_grad_norms": [],
        "mb_cosine_sims": [],
    }

    start_time = time.time()

    for update_idx in range(num_updates):
        # 1. Exact Theoretical E Gradient at Current Parameters
        exact_loss_val, g_exact_critic, start_val = compute_exact_e_loss_and_grad(
            train_state, evaluator, network, gamma
        )
        norm_exact = float(jnp.linalg.norm(g_exact_critic))

        # 2. Collect Rollouts and build Minibatches
        env_state, obsv, minibatches, rng_run = rollout_and_batch_fn(
            train_state, env_state, obsv, rng_run
        )

        # 3. Measure Minibatch Gradients & Compute Average Update Gradient
        def critic_loss_fn(params, mb_info):
            obs_mb, _, _, next_obs_mb, true_term_mb, next_tgt_mb, _, targets_mb = mb_info
            v_i = network.apply(params, obs_mb, method=network.value)
            v_j = network.apply(params, next_obs_mb, method=network.value)
            value_loss, _, _ = helpers.e_critic_loss(
                v_i, targets_mb, v_j, next_tgt_mb, true_term_mb, gamma
            )
            return value_loss

        mb_grads_list = []
        mb_losses_list = []
        num_mbs = config["NUM_MINIBATCHES"]
        for mb_idx in range(num_mbs):
            mb_item = jax.tree_util.tree_map(lambda x: x[mb_idx], minibatches)
            l_val, g_mb = jax.value_and_grad(critic_loss_fn)(train_state.params, mb_item)
            g_mb_flat = extract_critic_flat_grads(g_mb)
            norm_g_mb = float(jnp.linalg.norm(g_mb_flat))

            if norm_g_mb > 1e-12 and norm_exact > 1e-12:
                cos_mb = float(jnp.dot(g_mb_flat, g_exact_critic) / (norm_g_mb * norm_exact))
            else:
                cos_mb = 0.0

            mb_grads_list.append(g_mb_flat)
            mb_losses_list.append(float(l_val))

            history["mb_updates"].append(update_idx)
            history["mb_grad_norms"].append(norm_g_mb)
            history["mb_cosine_sims"].append(cos_mb)

        # Average update gradient: g_bar_update = (1/K) * sum g_mb
        # Identically the full-batch update gradient over N = B * T transitions
        mb_grads_arr = jnp.stack(mb_grads_list, axis=0)  # [K, D]
        g_avg_update = jnp.mean(mb_grads_arr, axis=0)
        norm_avg_update = float(jnp.linalg.norm(g_avg_update))

        if norm_avg_update > 1e-12 and norm_exact > 1e-12:
            cos_sim_update = float(jnp.dot(g_avg_update, g_exact_critic) / (norm_avg_update * norm_exact))
        else:
            cos_sim_update = 0.0

        rel_err_update = float(
            jnp.linalg.norm(g_avg_update - g_exact_critic) / max(norm_exact, 1e-12)
        )
        sq_err_update = float(jnp.sum((g_avg_update - g_exact_critic) ** 2))
        norm_ratio_update = float(norm_avg_update / max(norm_exact, 1e-12))
        avg_sampled_loss = float(np.mean(mb_losses_list))

        # Variance of the average gradient: Var(g_bar) = (1 / K(K-1)) * sum ||g_m - g_bar||^2
        diff_from_avg = mb_grads_arr - g_avg_update
        mb_variance = float(jnp.mean(jnp.sum(diff_from_avg ** 2, axis=-1)))
        if num_mbs > 1:
            var_avg_update = float(jnp.sum(diff_from_avg ** 2) / (num_mbs * (num_mbs - 1)))
        else:
            var_avg_update = 0.0

        history["update"].append(update_idx)
        history["policy_return"].append(start_val)
        history["exact_loss"].append(exact_loss_val)
        history["sampled_loss"].append(avg_sampled_loss)
        history["exact_grad_norm"].append(norm_exact)
        history["sampled_update_grad_norm"].append(norm_avg_update)
        history["cosine_sim_update"].append(cos_sim_update)
        history["rel_err_update"].append(rel_err_update)
        history["update_variance"].append(var_avg_update)
        history["update_sq_err"].append(sq_err_update)
        history["mb_variance"].append(mb_variance)
        history["norm_ratio_update"].append(norm_ratio_update)

        # 4. Apply PPO update step
        train_state = ppo_update_fn(train_state, minibatches)

    elapsed = time.time() - start_time
    history_np = {k: np.array(v) for k, v in history.items()}
    history_np["runtime_seconds"] = np.array(elapsed)
    return history_np


def plot_single_trajectory_comparison(history: Dict[str, np.ndarray], env_name: str, out_path: str):
    """
    Plots the comparison trajectory:
    1. Top: Exact gradient norm (thick line) vs. sampled update average and minibatch scatter dots.
    2. Middle: Directional cosine similarity rho of the update average gradient.
    3. Bottom: Policy return V_pi(s_0) and relative error of the update average gradient.
    """
    updates = history["update"]
    mb_updates = history["mb_updates"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 1. Gradient Norms (Exact Thick Line vs Sampled Update vs Minibatch Scatter)
    ax1 = axes[0]
    ax1.plot(updates, history["exact_grad_norm"], color="#111111", lw=3.2, label=r"Exact Expected Gradient $\|g_{\mathrm{exact}}\|$ (Ground Truth)", zorder=4)
    ax1.plot(updates, history["sampled_update_grad_norm"], color="#d62728", lw=1.8, ls="--", label=r"Average Update Gradient $\|\bar{g}_{\mathrm{update}}\|$ (Full Batch)", zorder=3)
    ax1.scatter(mb_updates, history["mb_grad_norms"], color="#ff7f0e", alpha=0.35, s=14, label="Minibatch Gradients (Per-batch Scatter)", zorder=2)
    ax1.set_ylabel(r"Gradient Norm $\|\nabla_\theta \mathcal{L}\|$", fontsize=11, fontweight="bold")
    ax1.set_title(f"Expected vs. Sampled E Critic Updates ({env_name})", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)

    # 2. Directional Cosine Similarity (rho)
    ax2 = axes[1]
    ax2.axhline(1.0, color="#2ca02c", ls=":", lw=2.0, label=r"Perfect Directional Alignment ($\rho = 1.0$)")
    ax2.plot(updates, history["cosine_sim_update"], color="#1f77b4", lw=2.4, label=r"Update Cosine Similarity $\rho(\bar{g}_{\mathrm{update}}, g_{\mathrm{exact}})$")
    ax2.scatter(mb_updates, history["mb_cosine_sims"], color="#17becf", alpha=0.25, s=12, label="Minibatch Alignments")
    ax2.set_ylabel(r"Cosine Similarity $\rho$", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.2, 1.05)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Policy Performance & Relative Error
    ax3 = axes[2]
    color_ret = "#222222"
    color_err = "#9467bd"

    ax3.plot(updates, history["policy_return"], color=color_ret, lw=2.2, label=r"Policy Return $V^\pi(s_0)$")
    ax3.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Expected Return", color=color_ret, fontsize=11, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor=color_ret)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(updates, history["rel_err_update"], color=color_err, lw=1.8, ls="-.", label=r"Update Relative Error $\|\bar{g}_{\mathrm{update}} - g_{\mathrm{exact}}\| / \|g_{\mathrm{exact}}\|$")
    ax3_twin.set_ylabel("Relative Error", color=color_err, fontsize=11, fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor=color_err)
    ax3_twin.grid(False)

    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved trajectory comparison plot to: {out_path}")


def run_sweep(
    env_name: str,
    num_steps_list: List[int],
    num_envs_list: List[int],
    num_updates: int = 35,
    seed: int = 42,
    out_dir: str = "results/sampled_vs_exact_e",
) -> Dict[str, np.ndarray]:
    """
    Sweeps over num_steps (T) and num_envs (B) to evaluate the accuracy and variance
    of the average update gradient.
    Generates 2D heatmaps and iso-batch variance scaling curves.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_envs = len(num_envs_list)
    n_steps = len(num_steps_list)

    grid_rho = np.zeros((n_envs, n_steps))
    grid_rel_err = np.zeros((n_envs, n_steps))
    grid_var_update = np.zeros((n_envs, n_steps))
    grid_norm_ratio = np.zeros((n_envs, n_steps))
    grid_batch_size = np.zeros((n_envs, n_steps))

    print("=" * 75)
    print(f"Starting 2D Parameter Sweep on {env_name}")
    print(f"Rollout Lengths (T): {num_steps_list}")
    print(f"Parallel Envs   (B): {num_envs_list}")
    print(f"Updates/Run: {num_updates} | Seed: {seed}")
    print("=" * 75)

    for i, b in enumerate(num_envs_list):
        for j, t in enumerate(num_steps_list):
            batch_size = b * t
            grid_batch_size[i, j] = batch_size
            print(f"--> (B={b:2d}, T={t:3d}, N={batch_size:5d})...", end="", flush=True)

            res = run_comparison_single_setting(
                env_name=env_name,
                num_updates=num_updates,
                num_envs=b,
                num_steps=t,
                seed=seed,
            )

            # Evaluate over updates where sampled gradient was non-zero
            sampled_norms = res["sampled_update_grad_norm"]
            nonzero_mask = sampled_norms > 1e-6
            if np.sum(nonzero_mask) >= 3:
                eval_slice = np.where(nonzero_mask)[0]
                # Take second half of active updates for stable regime
                mid = len(eval_slice) // 2
                eval_slice = eval_slice[mid:]
            else:
                eval_slice = slice(max(num_updates // 2, 1), None)

            mean_rho = float(np.mean(res["cosine_sim_update"][eval_slice]))
            mean_rel_err = float(np.mean(res["rel_err_update"][eval_slice]))
            mean_var = float(np.mean(res["update_variance"][eval_slice]))
            mean_ratio = float(np.mean(res["norm_ratio_update"][eval_slice]))

            grid_rho[i, j] = mean_rho
            grid_rel_err[i, j] = mean_rel_err
            grid_var_update[i, j] = mean_var
            grid_norm_ratio[i, j] = mean_ratio

            print(f" Done! CosSim: {mean_rho:.4f} | RelErr: {mean_rel_err:.4f} | Var(g_bar): {mean_var:.2e}")

    sweep_data = {
        "env_name": env_name,
        "num_steps_list": np.array(num_steps_list),
        "num_envs_list": np.array(num_envs_list),
        "grid_batch_size": grid_batch_size,
        "grid_rho": grid_rho,
        "grid_rel_err": grid_rel_err,
        "grid_var_update": grid_var_update,
        "grid_norm_ratio": grid_norm_ratio,
    }

    # Save sweep arrays
    np.savez(os.path.join(out_dir, f"{env_name}_sweep_data.npz"), **sweep_data)

    # Plot 2D Heatmaps
    plot_sweep_heatmaps(
        grid_rho, grid_rel_err, grid_var_update, num_steps_list, num_envs_list, env_name,
        os.path.join(out_dir, f"{env_name}_sweep_heatmaps.png")
    )

    # Plot Iso-Batch Variance Scaling Curves
    plot_iso_batch_scaling(
        sweep_data, os.path.join(out_dir, f"{env_name}_iso_batch_scaling.png")
    )

    return sweep_data


def plot_sweep_heatmaps(
    grid_rho: np.ndarray,
    grid_rel_err: np.ndarray,
    grid_var_update: np.ndarray,
    num_steps_list: List[int],
    num_envs_list: List[int],
    env_name: str,
    out_path: str,
):
    """Plots 3-panel 2D heatmaps of Cosine Similarity, Relative Error, and Update Variance."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Heatmap 1: Cosine Similarity
    im1 = axes[0].imshow(grid_rho, cmap="viridis", aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(range(len(num_steps_list)))
    axes[0].set_xticklabels(num_steps_list, fontsize=10)
    axes[0].set_yticks(range(len(num_envs_list)))
    axes[0].set_yticklabels(num_envs_list, fontsize=10)
    axes[0].set_xlabel("Rollout Length T (num_steps)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Parallel Envs B (num_envs)", fontsize=11, fontweight="bold")
    axes[0].set_title(r"Directional Cosine Similarity $\rho(\bar{g}_{\mathrm{update}}, g_{\mathrm{exact}})$", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(len(num_envs_list)):
        for j in range(len(num_steps_list)):
            axes[0].text(j, i, f"{grid_rho[i, j]:.2f}", ha="center", va="center",
                         color="white" if grid_rho[i, j] < 0.6 else "black", fontsize=9, fontweight="bold")

    # Heatmap 2: Relative Error
    im2 = axes[1].imshow(grid_rel_err, cmap="magma_r", aspect="auto", origin="lower")
    axes[1].set_xticks(range(len(num_steps_list)))
    axes[1].set_xticklabels(num_steps_list, fontsize=10)
    axes[1].set_yticks(range(len(num_envs_list)))
    axes[1].set_yticklabels(num_envs_list, fontsize=10)
    axes[1].set_xlabel("Rollout Length T (num_steps)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Parallel Envs B (num_envs)", fontsize=11, fontweight="bold")
    axes[1].set_title(r"Relative Error $\|\bar{g}_{\mathrm{update}} - g_{\mathrm{exact}}\| / \|g_{\mathrm{exact}}\|$", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    for i in range(len(num_envs_list)):
        for j in range(len(num_steps_list)):
            axes[1].text(j, i, f"{grid_rel_err[i, j]:.2f}", ha="center", va="center",
                         color="white" if grid_rel_err[i, j] > np.median(grid_rel_err) else "black", fontsize=9, fontweight="bold")

    # Heatmap 3: Update Gradient Variance (log scale)
    log_var = np.log10(np.maximum(grid_var_update, 1e-12))
    im3 = axes[2].imshow(log_var, cmap="coolwarm_r", aspect="auto", origin="lower")
    axes[2].set_xticks(range(len(num_steps_list)))
    axes[2].set_xticklabels(num_steps_list, fontsize=10)
    axes[2].set_yticks(range(len(num_envs_list)))
    axes[2].set_yticklabels(num_envs_list, fontsize=10)
    axes[2].set_xlabel("Rollout Length T (num_steps)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Parallel Envs B (num_envs)", fontsize=11, fontweight="bold")
    axes[2].set_title(r"Update Gradient Variance $\log_{10} \widehat{\mathrm{Var}}(\bar{g}_{\mathrm{update}})$", fontsize=11, fontweight="bold")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    for i in range(len(num_envs_list)):
        for j in range(len(num_steps_list)):
            axes[2].text(j, i, f"{log_var[i, j]:.1f}", ha="center", va="center",
                         color="white" if abs(log_var[i, j] - np.mean(log_var)) > 1.0 else "black", fontsize=9, fontweight="bold")

    fig.suptitle(f"Sampled vs. Exact E Metrics Grid ({env_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved sweep heatmap figure to: {out_path}")


def plot_iso_batch_scaling(sweep_data: Dict[str, np.ndarray], out_path: str):
    """
    Plots Update Gradient Variance and Relative Error vs. Total Batch Size N = B * T.
    Demonstrates that averaging the minibatch gradients yields variance that scales
    purely with total batch size N.
    """
    env_name = sweep_data["env_name"]
    num_steps_list = sweep_data["num_steps_list"]
    num_envs_list = sweep_data["num_envs_list"]
    grid_var = sweep_data["grid_var_update"]
    grid_rel_err = sweep_data["grid_rel_err"]
    grid_rho = sweep_data["grid_rho"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(num_steps_list)))

    # Panel 1: Update Gradient Variance vs Total Batch Size N = B * T
    for j, t in enumerate(num_steps_list):
        batch_sizes = [b * t for b in num_envs_list]
        vars_t = grid_var[:, j]
        ax1.plot(batch_sizes, vars_t, marker="o", lw=2.0, color=colors[j], label=f"T = {t}")

    # Theoretical 1/N reference slope
    all_batches = np.sort(np.unique([[b * t for b in num_envs_list] for t in num_steps_list]))
    median_var = np.median(grid_var[grid_var > 0]) if np.any(grid_var > 0) else 1e-4
    ref_x = np.array([all_batches[0], all_batches[-1]])
    ref_y = median_var * (all_batches[len(all_batches)//2] / ref_x)
    ax1.plot(ref_x, ref_y, "k--", lw=1.8, alpha=0.7, label=r"Ideal $O(1/N)$ Scaling")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Total Batch Size $N = B \\times T$", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Update Gradient Variance $\widehat{\mathrm{Var}}(\bar{g}_{\mathrm{update}})$", fontsize=11, fontweight="bold")
    ax1.set_title("Variance Scaling of Average Update Gradient", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)

    # Panel 2: Relative Error vs Total Batch Size N = B * T
    for j, t in enumerate(num_steps_list):
        batch_sizes = [b * t for b in num_envs_list]
        errs_t = grid_rel_err[:, j]
        ax2.plot(batch_sizes, errs_t, marker="s", lw=2.0, color=colors[j], label=f"T = {t}")

    ax2.set_xscale("log")
    ax2.set_xlabel("Total Batch Size $N = B \\times T$", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Relative Error $\|\bar{g}_{\mathrm{update}} - g_{\mathrm{exact}}\| / \|g_{\mathrm{exact}}\|$", fontsize=11, fontweight="bold")
    ax2.set_title("Relative Error vs. Total Batch Size", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Iso-Batch Scaling Analysis ({env_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved iso-batch scaling plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sampled vs. Exact E Critic Updates")
    parser.add_argument(
        "--env_name",
        type=str,
        default="FourRooms-misc",
        choices=[
            "FourRooms-misc",
            "fourrooms-dense",
            "eightrooms-misc",
            "eightrooms-dense",
            "whirlpool-misc",
            "MountainCar-v0",
        ],
        help="Target environment with exact tabular evaluator dynamics",
    )
    parser.add_argument("--mode", type=str, default="single", choices=["single", "sweep"], help="Mode: single trajectory or 2D parameter sweep")
    parser.add_argument("--num_updates", type=int, default=40, help="Number of policy update steps")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments (for single mode)")
    parser.add_argument("--num_steps", type=int, default=64, help="Rollout length per environment (for single mode)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, f"results/sampled_vs_exact_e/{args.env_name}")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "single":
        print(f"Running single comparison on {args.env_name} (envs={args.num_envs}, steps={args.num_steps})...")
        history = run_comparison_single_setting(
            env_name=args.env_name,
            num_updates=args.num_updates,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            lr=args.lr,
            seed=args.seed,
            out_dir=args.out_dir,
        )
        plot_path = os.path.join(args.out_dir, f"{args.env_name}_trajectory_comparison.png")
        plot_single_trajectory_comparison(history, args.env_name, plot_path)
        np.savez(os.path.join(args.out_dir, f"{args.env_name}_metrics.npz"), **history)
        print("Completed successfully!")

    elif args.mode == "sweep":
        if args.env_name == "MountainCar-v0":
            steps_list = [16, 32, 64, 100, 200]
        else:
            steps_list = [16, 32, 64, 128, 256]
        envs_list = [4, 8, 16, 32, 64]

        run_sweep(
            env_name=args.env_name,
            num_steps_list=steps_list,
            num_envs_list=envs_list,
            num_updates=args.num_updates,
            seed=args.seed,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
