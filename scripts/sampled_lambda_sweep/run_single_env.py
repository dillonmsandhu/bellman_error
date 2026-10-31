"""
scripts/sampled_lambda_sweep/run_single_env.py

Executes sampled PPO across 2 algorithms and multiple values of lambda
on a single benchmark environment:
  1. td_lambda: PPO with TD(lambda) critic (ppo/sampled_td_lambda.py)
  2. e_lambda:  PPO with Geometric Jump Sampled E(lambda) critic using v_aux debiasing (ppo/E_lambda_geometric.py)

Sweeps:
  - Lambdas: [0.5, 0.9, 0.95] (matching head-to-head comparisons)
  - Critic LRs: [0.0005, 0.001, 0.003] (default, CLI configurable)
  - Actor LR: Fixed at 0.0003

Condition selection:
  - Selects winning critic LR per (algorithm, lambda) condition by standardized
    policy return AUC: V_norm = (V_start - V_rand) / (V_star - V_rand).

Outputs:
  - out.pkl
  - summary.csv (winning run per condition)
  - summary_all_lrs.csv (all evaluated learning rates)
  - lr_selection_summary.json
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import pickle
import json
from typing import Dict, Any, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from core.config import config as base_cfg
import core.helpers as helpers
from ppo.sampled_td_lambda import make_train as make_td_lambda_train
from ppo.E_lambda_geometric import make_train as make_e_lambda_train

trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

ALL_ENVS = [
    "mountaincar",
    "FourRooms-misc",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]

ALGORITHMS = {
    "td_lambda": {
        "display_name": "TD(λ)",
        "cond_prefix": "td",
        "train_builder": make_td_lambda_train,
        "base_hparams": {},
        "color_base": "#ff7f0e",  # Orange base
    },
    "e_lambda": {
        "display_name": "E(λ) [v_aux]",
        "cond_prefix": "e",
        "train_builder": make_e_lambda_train,
        "base_hparams": {"E_LAMBDA_VARIANT": "aux", "RETURN_LAMBDA": 1.0},
        "color_base": "#2ca02c",  # Green base
    },
    "e_lambda_fwd": {
        "display_name": "E(λ) [Forward-Only]",
        "cond_prefix": "e_fwd",
        "train_builder": make_e_lambda_train,
        "base_hparams": {"E_LAMBDA_VARIANT": "forward", "RETURN_LAMBDA": 1.0},
        "color_base": "#1f77b4",  # Blue base
    },
}

LAMBDA_COLORS = {
    # TD(lambda) - Orange / Red shades
    "td_0.40": "#fdd0a2",
    "td_0.80": "#fdae6b",
    "td_0.95": "#e6550d",
    "td_0.99": "#8c2d04",
    # E(lambda) [v_aux] - Green shades
    "e_0.40": "#c7e9c0",
    "e_0.80": "#74c476",
    "e_0.95": "#238b45",
    "e_0.99": "#00441b",
    # E(lambda) [Forward-Only] - Blue shades
    "e_fwd_0.40": "#c6dbef",
    "e_fwd_0.80": "#6baed6",
    "e_fwd_0.95": "#2171b5",
    "e_fwd_0.99": "#08306b",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sampled PPO Lambda Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.4, 0.8, 0.95, 0.99], help="Lambda values to evaluate (default: 0.4 0.8 0.95 0.99)")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of rollout updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k", type=int, default=32, help="Critic feature dimension (default: 32)")
    parser.add_argument("--actor_lr", type=float, default=0.0003, help="Actor learning rate (default: 0.0003)")
    parser.add_argument("--critic_lrs", type=float, nargs="+", default=[0.0005, 0.001, 0.003], help="Critic learning rates to sweep (default: 0.0005 0.001 0.003)")
    parser.add_argument("--num_envs", type=int, default=32, help="Parallel environments (default: 32)")
    parser.add_argument("--num_steps", type=int, default=256, help="Steps per rollout (default: 256)")
    parser.add_argument("--minibatch_size", type=int, default=256, help="Minibatch size (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Epochs per update (default: 4)")
    parser.add_argument("--gae_lambda", type=float, default=0.8, help="Actor GAE lambda (default: 0.8)")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon (default: 0.2)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--runway_length", type=str, default="adaptive", help="Runway window length for geometric jumps (integer or 'adaptive', default: 'adaptive')")
    parser.add_argument("--select_by", type=str, default="std_perf_auc", choices=["std_perf_auc", "std_perf_final", "ve_auc"], help="Metric to select best critic condition")
    parser.add_argument("--out_dir", type=str, default="results/sampled_lambda_sweep", help="Root output directory")
    parser.add_argument("--algos", type=str, nargs="+", default=["td_lambda", "e_lambda", "e_lambda_fwd"], choices=["td_lambda", "e_lambda", "e_lambda_fwd"], help="Algorithms to evaluate")
    parser.add_argument("--eval_grad_fidelity", action="store_true", default=True, help="Compute exact gradient MSE, variance, and cossim (default: True)")
    return parser.parse_args()


def load_env_baseline(env_name: str) -> tuple[float, float, float]:
    """Loads v_star, v_rand, and delta from reference/env_performance_baselines.json."""
    json_path = os.path.join(REPO_ROOT, "reference", "env_performance_baselines.json")
    if not os.path.exists(json_path):
        return 1.0, 0.0, 1.0
    with open(json_path, "r") as f:
        data = json.load(f)
    if env_name in data:
        v_star = float(data[env_name]["v_star"])
        v_rand = float(data[env_name]["v_rand"])
        delta = float(data[env_name]["delta"])
        return v_star, v_rand, delta
    return 1.0, 0.0, 1.0


def build_env_config(env_name: str, args: argparse.Namespace, critic_lr: float) -> Dict[str, Any]:
    """Configures environment parameters matching sampled PPO standards."""
    is_mc = "mountaincar" in env_name.lower()
    batch_size = args.num_envs * args.num_steps
    runway_val = None if str(args.runway_length).lower() in ["none", "adaptive", "auto"] else int(args.runway_length)
    cfg = base_cfg.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": args.num_steps,
        "NUM_ENVS": args.num_envs,
        "MINIBATCH_SIZE": min(args.minibatch_size, batch_size),
        "NUM_EPOCHS": args.num_epochs,
        "NUM_EPOCHS_ACTOR": args.num_epochs,
        "NUM_EPOCHS_CRITIC": args.num_epochs,
        "TOTAL_TIMESTEPS": args.num_updates * batch_size,
        "NUM_UPDATES": args.num_updates,
        "N_SEEDS": args.n_seeds,
        "SEED": args.seed,
        "GAMMA": args.gamma,
        "k": args.k,
        "ACTOR_LR": args.actor_lr,
        "ACTOR_LR_END": args.actor_lr,
        "CRITIC_LR": critic_lr,
        "CRITIC_LR_END": critic_lr,
        "LR": critic_lr,
        "LR_END": critic_lr,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
        "RUNWAY_LENGTH": runway_val,
    })
    return cfg


def compute_normalized_auc(trajectory_1d: np.ndarray) -> float:
    """Computes trapezoidal Area Under the Curve normalized by steps."""
    if len(trajectory_1d) < 2:
        return float(trajectory_1d[0])
    x = np.arange(len(trajectory_1d))
    return float(trapz_fn(trajectory_1d, x) / (len(trajectory_1d) - 1))


def compute_gradient_fidelity(
    env_name: str,
    algo_key: str,
    lam: float,
    gamma: float = 0.99,
    k: int = 32,
    num_eval_batches: int = 20,
    num_trajectories: int = 32,
    num_steps: int = 128,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluates empirical sample gradient MSE, empirical trace variance,
    and cosine similarity against the exact matrix gradient.
    NOTE:
      - For E(lambda) [v_aux] AND E(lambda) [Forward-Only], g_exact is the exact E(lambda) gradient (- J^T S_lambda e).
      - For TD(lambda), g_exact is the exact TD(lambda) update (- J^T A_lambda e).
    """
    try:
        from core.networks import initialize_network, initialize_flax_train_state
        from core.gradient_tracking import extract_critic_flat_grads, cos_similarity
        from core.helpers import e_lambda_geometric_critic_loss

        cfg = base_cfg.copy()
        cfg["ENV_NAME"] = env_name
        cfg["k"] = k
        env, env_params = helpers.make_env(cfg)
        evaluator = helpers.create_evaluator(cfg, env, env_params)

        S_states = evaluator.obs_stack
        N = evaluator.num_states
        start_idx = evaluator.start_idx
        P = evaluator.P

        old_pi = jnp.ones((N, 4)) / 4.0
        terminal_policy = jnp.ones([1, 4]) / 4.0
        old_pi_full = jnp.vstack([old_pi, terminal_policy])

        mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
        mu = jnp.append(mu, 0.0)
        D = jnp.diag(mu)
        P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
        I = jnp.eye(N + 1)
        V_true = evaluator.compute_true_values_raw(old_pi_full)

        mu_trans = mu[:N] / jnp.maximum(jnp.sum(mu[:N]), 1e-12)

        rng = jax.random.PRNGKey(seed)
        rng, rng_net, rng_aux = jax.random.split(rng, 3)
        obs_shape = evaluator.obs_stack.shape[1:]

        network, params = initialize_network(rng_net, obs_shape, env, env_params, k, n_heads=1, layer_norm=False)
        aux_network, aux_params = initialize_network(rng_aux, obs_shape, env, env_params, k, n_heads=1, layer_norm=False)
        aux_cfg = cfg.copy()
        aux_cfg["NUM_UPDATES"] = 100
        aux_cfg["NUM_MINIBATCHES"] = 1
        aux_cfg["NUM_EPOCHS"] = 1
        aux_train_state = initialize_flax_train_state(aux_cfg, aux_network, aux_params)

        v_cur = network.apply(params, S_states, method=network.value).squeeze()
        v_full_cur = jnp.append(v_cur, 0.0)

        # 1. Exact Compound Matrix
        L_lam = jnp.linalg.inv(I - gamma * lam * P_pi)
        A_lam = D @ (I - gamma * P_pi) @ L_lam
        S_lam = 0.5 * (A_lam + A_lam.T)

        def get_e(p):
            v = network.apply(p, S_states, method=network.value).squeeze()
            return V_true - jnp.append(v, 0.0)

        if algo_key == "td_lambda":
            # TD update direction: grad_p (v^T A_lam e_sg) = J^T A_lam e
            e_val = get_e(params)
            def td_exact_loss(p):
                v_p = network.apply(p, S_states, method=network.value).squeeze()
                v_full = jnp.append(v_p, 0.0)
                return jnp.dot(v_full, A_lam @ jax.lax.stop_gradient(e_val))
            g_exact = extract_critic_flat_grads(jax.grad(td_exact_loss)(params))
        else:
            # E(lambda) exact gradient: grad_p (0.5 * e^T S_lam e)
            # Both v_aux and Forward-Only estimators are evaluated against this exact objective!
            def e_exact_loss(p):
                e = get_e(p)
                return 0.5 * (e.T @ S_lam @ e)
            g_exact = extract_critic_flat_grads(jax.grad(e_exact_loss)(params))

        # 2. Rollout sampling
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
        def compute_mc_returns(traj_r, traj_done, final_s):
            v_final = v_full_cur[final_s]
            def rev_scan_fn(g_next, step_data):
                r_t, d_t = step_data
                g_cur = r_t + gamma * jnp.where(d_t, 0.0, g_next)
                return g_cur, g_cur
            _, returns = jax.lax.scan(rev_scan_fn, v_final, (traj_r, traj_done), reverse=True)
            return returns

        if algo_key == "e_lambda":
            pretrain_keys = jax.random.split(rng, 10)
            for pk in pretrain_keys:
                traj_s, _, traj_r, traj_done, final_s = sample_rollout(pk)
                returns = compute_mc_returns(traj_r, traj_done, final_s)
                def aux_step(ts, _):
                    def loss(p_a):
                        v_pred = aux_network.apply(p_a, S_states).squeeze()
                        v_full = jnp.append(v_pred, 0.0)
                        return jnp.mean((returns - v_full[traj_s]) ** 2)
                    grads = jax.grad(loss)(ts.params)
                    return ts.apply_gradients(grads=grads), None
                aux_train_state, _ = jax.lax.scan(aux_step, aux_train_state, None, length=3)

        batch_keys = jax.random.split(rng, num_eval_batches)
        sample_grads = []

        for bk in batch_keys:
            r_roll, r_jump = jax.random.split(bk)
            traj_s, _, traj_r, traj_done, final_s = sample_rollout(r_roll)
            returns = compute_mc_returns(traj_r, traj_done, final_s)

            if algo_key == "td_lambda":
                v_s = network.apply(params, S_states, method=network.value).squeeze()
                v_full = jnp.append(v_s, 0.0)
                v_traj = v_full[traj_s]
                v_final = v_full[final_s]

                def gae_scan(gae_next, step_data):
                    r_t, d_t, v_t, v_nxt = step_data
                    mask = 1.0 - d_t.astype(jnp.float32)
                    delta = r_t + gamma * v_nxt * mask - v_t
                    gae_cur = delta + gamma * lam * mask * gae_next
                    return gae_cur, gae_cur + v_t

                v_nxt_all = jnp.concatenate([v_traj[1:], v_final[None]], axis=0)
                _, targets = jax.lax.scan(gae_scan, jnp.zeros_like(v_traj[0]), (traj_r, traj_done, v_traj, v_nxt_all), reverse=True)

                def td_loss(p_c):
                    v_curr = network.apply(p_c, S_states, method=network.value).squeeze()
                    v_all = jnp.append(v_curr, 0.0)[traj_s]
                    return 0.5 * jnp.mean((v_all - jax.lax.stop_gradient(targets)) ** 2)

                g_samp = extract_critic_flat_grads(jax.grad(td_loss)(params))
            else:
                if algo_key == "e_lambda":
                    v_aux_all = aux_network.apply(aux_train_state.params, S_states, method=aux_network.value).squeeze()
                    v_aux_full = jnp.append(v_aux_all, 0.0)
                    aux_vals = v_aux_full[traj_s]
                else:
                    aux_vals = None  # Forward-Only

                def e_loss(p_c):
                    v_a = network.apply(p_c, S_states, method=network.value).squeeze()
                    vals = jnp.append(v_a, 0.0)[traj_s]
                    val_loss, _, _ = e_lambda_geometric_critic_loss(
                        values=vals,
                        targets=returns,
                        dones=traj_done,
                        gamma=gamma,
                        lmbda=lam,
                        rng=r_jump,
                        aux_values=aux_vals,
                        true_terminals=traj_done,
                        return_details=False,
                    )
                    return val_loss

                g_samp = extract_critic_flat_grads(jax.grad(e_loss)(params))

            sample_grads.append(np.array(g_samp))

        sample_grads = np.array(sample_grads)
        g_exact_np = np.array(g_exact)

        grad_mse = float(np.mean(np.linalg.norm(sample_grads - g_exact_np, axis=1) ** 2))
        grad_var = float(np.sum(np.var(sample_grads, axis=0)))
        grad_cossim = float(np.mean([cos_similarity(g, g_exact_np) for g in sample_grads]))

        return {"grad_mse": grad_mse, "grad_var": grad_var, "grad_cossim": grad_cossim}

    except Exception as e:
        print(f"    [Warning: could not evaluate gradient fidelity: {e}]")
        return {"grad_mse": float("nan"), "grad_var": float("nan"), "grad_cossim": float("nan")}


def run_algo_condition(
    train_builder,
    config: Dict[str, Any],
    algo_hparams: Dict[str, Any],
    n_seeds: int,
    base_seed: int,
) -> Dict[str, np.ndarray]:
    """Runs a single algorithm across random seeds using jax.vmap."""
    run_cfg = config.copy()
    run_cfg.update(algo_hparams)

    train_fn = train_builder(run_cfg)
    vmapped_train = jax.vmap(train_fn)

    rng = jax.random.PRNGKey(base_seed)
    rngs = jax.random.split(rng, n_seeds)

    t0 = time.time()
    out = vmapped_train(rngs)
    dt = time.time() - t0

    metrics = out.get("metrics", {})
    converted_metrics = {}
    for key, val in metrics.items():
        if isinstance(val, (jax.Array, jnp.ndarray)):
            converted_metrics[key] = np.array(val)
        else:
            converted_metrics[key] = val

    converted_metrics["_runtime_sec"] = dt
    return converted_metrics


def main():
    args = parse_args()

    if args.env_idx is not None:
        if 0 <= args.env_idx < len(ALL_ENVS):
            args.env_name = ALL_ENVS[args.env_idx]
        else:
            raise ValueError(f"env_idx {args.env_idx} out of range [0, {len(ALL_ENVS)-1}]")

    v_star, v_rand, delta = load_env_baseline(args.env_name)

    print("=" * 80)
    print("Sampled PPO Lambda Sweep on Single Environment")
    print(f"Environment:       {args.env_name}")
    print(f"Num Updates:       {args.num_updates}")
    print(f"Rollout Config:    {args.num_envs} envs x {args.num_steps} steps (batch={args.num_envs * args.num_steps})")
    print(f"Num Seeds:         {args.n_seeds} (base seed: {args.seed})")
    print(f"Critic Feature k:  {args.k}")
    print(f"Actor LR:          {args.actor_lr} (fixed)")
    print(f"Critic LRs:        {args.critic_lrs}")
    print(f"Lambdas:           {args.lambdas}")
    print(f"Actor GAE Lambda:  {args.gae_lambda}")
    print(f"PPO Clip Epsilon:  {args.clip_eps}")
    print(f"Selection Metric:  {args.select_by}")
    print(f"Algorithms:        {args.algos}")
    print(f"Baselines:         V* = {v_star:.4f} | V_rand = {v_rand:.4f} | Delta = {delta:.4f}")
    out_env_dir = os.path.join(os.path.abspath(args.out_dir), args.env_name)
    os.makedirs(out_env_dir, exist_ok=True)
    print(f"Output Directory:  {out_env_dir}")
    print("=" * 80)

    full_results = {
        "env_name": args.env_name,
        "k": args.k,
        "args": vars(args),
        "baselines": {"v_star": v_star, "v_rand": v_rand, "delta": delta},
        "runs": {},             # Winning run per condition: cond_key -> metrics dict
        "all_runs": {},          # All evaluated runs: cond_key -> lr -> metrics dict
        "best_critic_lrs": {},   # cond_key -> best_lr
        "best_conditions": {},   # cond_key -> selection summary dict
        "selection_metric": args.select_by,
    }

    summary_rows = []
    all_lrs_rows = []

    for algo_key in args.algos:
        algo_info = ALGORITHMS[algo_key]
        cond_prefix = algo_info.get("cond_prefix", "td" if algo_key == "td_lambda" else "e")
        for lam in args.lambdas:
            lam_str = f"{lam:.2f}"
            cond_key = f"{cond_prefix}_{lam_str}"
            display_name = f"{algo_info['display_name']}({lam_str})"

            print(f"\n========================================================")
            print(f"Evaluating Condition: {cond_key} ({display_name})")
            print(f"========================================================")

            # Evaluate exact gradient fidelity metrics
            if args.eval_grad_fidelity:
                print(f"  --> Measuring Critic Gradient Fidelity (MSE & Var) ...", end="", flush=True)
                grad_fid = compute_gradient_fidelity(
                    env_name=args.env_name,
                    algo_key=algo_key,
                    lam=float(lam),
                    gamma=args.gamma,
                    k=args.k,
                    seed=args.seed,
                )
                print(f" MSE={grad_fid['grad_mse']:.6e} | Var={grad_fid['grad_var']:.6e} | CosSim={grad_fid['grad_cossim']:+.4f}")
            else:
                grad_fid = {"grad_mse": float("nan"), "grad_var": float("nan"), "grad_cossim": float("nan")}

            full_results["all_runs"][cond_key] = {}
            lr_candidates = []

            for c_lr in args.critic_lrs:
                print(f"  --> Testing Critic LR = {c_lr} ...", end="", flush=True)
                env_cfg = build_env_config(args.env_name, args, critic_lr=c_lr)

                hparams = dict(algo_info["base_hparams"])
                hparams["VALUE_LAMBDA"] = float(lam)

                res = run_algo_condition(
                    train_builder=algo_info["train_builder"],
                    config=env_cfg,
                    algo_hparams=hparams,
                    n_seeds=args.n_seeds,
                    base_seed=args.seed,
                )

                dt = res["_runtime_sec"]
                v_start_seeds = res.get("V_start")  # shape: (n_seeds, num_updates)
                ve_seeds = res.get("nn_weighted_VE")
                cossim_seeds = res.get("nn_advantage_cossim")

                # Compute normalized performance trajectory per seed: (V_start - V_rand) / delta
                std_perf_seeds = (v_start_seeds - v_rand) / delta
                std_perf_mean = np.mean(std_perf_seeds, axis=0)
                std_perf_sem = np.std(std_perf_seeds, axis=0) / np.sqrt(args.n_seeds)

                # Summary scalar metrics across updates
                auc_perf = compute_normalized_auc(std_perf_mean)
                final_perf = float(std_perf_mean[-1])
                final_perf_sem = float(std_perf_sem[-1])

                final_ve = float(np.mean(ve_seeds[:, -1])) if ve_seeds is not None else float("nan")
                final_ve_sem = (
                    float(np.std(ve_seeds[:, -1]) / np.sqrt(args.n_seeds))
                    if ve_seeds is not None
                    else float("nan")
                )
                auc_ve = (
                    compute_normalized_auc(np.mean(ve_seeds, axis=0))
                    if ve_seeds is not None
                    else float("nan")
                )

                final_cossim = (
                    float(np.mean(cossim_seeds[:, -1])) if cossim_seeds is not None else float("nan")
                )
                final_cossim_sem = (
                    float(np.std(cossim_seeds[:, -1]) / np.sqrt(args.n_seeds))
                    if cossim_seeds is not None
                    else float("nan")
                )
                auc_cossim = (
                    compute_normalized_auc(np.mean(cossim_seeds, axis=0))
                    if cossim_seeds is not None
                    else float("nan")
                )

                print(
                    f" done ({dt:.1f}s) | AUC_Perf={auc_perf:+.4f} | Final_Perf={final_perf:+.4f} +/- {final_perf_sem:.4f} | Final_VE={final_ve:.4f}"
                )

                cand_record = {
                    "env_name": args.env_name,
                    "algo": algo_key,
                    "lambda": lam,
                    "condition": cond_key,
                    "display_name": display_name,
                    "actor_lr": args.actor_lr,
                    "critic_lr": c_lr,
                    "std_perf_auc": auc_perf,
                    "std_perf_final": final_perf,
                    "std_perf_final_sem": final_perf_sem,
                    "ve_auc": auc_ve,
                    "ve_final": final_ve,
                    "ve_final_sem": final_ve_sem,
                    "cossim_auc": auc_cossim,
                    "cossim_final": final_cossim,
                    "cossim_final_sem": final_cossim_sem,
                    "grad_mse": grad_fid["grad_mse"],
                    "grad_var": grad_fid["grad_var"],
                    "grad_cossim": grad_fid["grad_cossim"],
                    "runtime_sec": dt,
                }
                lr_candidates.append((c_lr, cand_record, res))
                all_lrs_rows.append(cand_record)
                full_results["all_runs"][cond_key][str(c_lr)] = res

            # Select winning condition based on args.select_by
            if args.select_by == "std_perf_auc":
                best_tuple = max(lr_candidates, key=lambda item: item[1]["std_perf_auc"])
            elif args.select_by == "std_perf_final":
                best_tuple = max(lr_candidates, key=lambda item: item[1]["std_perf_final"])
            elif args.select_by == "ve_auc":
                best_tuple = min(lr_candidates, key=lambda item: item[1]["ve_auc"])
            else:
                best_tuple = max(lr_candidates, key=lambda item: item[1]["std_perf_auc"])

            best_lr, best_record, best_res = best_tuple
            print(
                f"  --> WINNING CRITIC LR for {cond_key}: {best_lr} (AUC={best_record['std_perf_auc']:+.4f}, Final={best_record['std_perf_final']:+.4f})"
            )

            full_results["runs"][cond_key] = best_res
            full_results["best_critic_lrs"][cond_key] = best_lr
            full_results["best_conditions"][cond_key] = best_record
            summary_rows.append(best_record)

    # Convert summaries to DataFrames
    summary_df = pd.DataFrame(summary_rows)
    all_lrs_df = pd.DataFrame(all_lrs_rows)

    # Save outputs
    pkl_path = os.path.join(out_env_dir, "out.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(full_results, f)
    print(f"\nSaved checkpoint: {pkl_path}")

    csv_summary_path = os.path.join(out_env_dir, "summary.csv")
    summary_df.to_csv(csv_summary_path, index=False)
    print(f"Saved winning summary: {csv_summary_path}")

    csv_all_lrs_path = os.path.join(out_env_dir, "summary_all_lrs.csv")
    all_lrs_df.to_csv(csv_all_lrs_path, index=False)
    print(f"Saved all LRs summary: {csv_all_lrs_path}")

    json_selection_path = os.path.join(out_env_dir, "lr_selection_summary.json")
    with open(json_selection_path, "w") as f:
        json.dump(
            {
                "env_name": args.env_name,
                "selection_metric": args.select_by,
                "best_conditions": full_results["best_conditions"],
                "best_critic_lrs": full_results["best_critic_lrs"],
            },
            f,
            indent=2,
        )
    print(f"Saved LR selection: {json_selection_path}")

    print("\n" + "=" * 80)
    print(f"FINAL WINNING CONDITIONS ON {args.env_name}:")
    print(
        summary_df[
            [
                "condition",
                "display_name",
                "critic_lr",
                "std_perf_auc",
                "std_perf_final",
                "std_perf_final_sem",
                "ve_final",
                "cossim_final",
            ]
        ].to_string(index=False)
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
