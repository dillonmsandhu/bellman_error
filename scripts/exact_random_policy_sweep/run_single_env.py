"""
scripts/exact_random_policy_sweep/run_single_env.py

Executes exact (in-expectation) random policy evaluation across 5 algorithms:
  1. MC:          random_policy.exact_mc
  2. TD(0):       random_policy.exact_td
  3. E(0):        random_policy.exact_E_gd
  4. TD(lambda):  random_policy.exact_td_lambda (VALUE_LAMBDA=0.8)
  5. E(lambda):   random_policy.exact_E_lambda (VALUE_LAMBDA=0.8)

Sweeps representation capacity bottleneck (k=8 vs k=64) and learning rates
(e.g., 0.0003, 0.001, 0.003).

Tracks key metrics:
  - nn_weighted_VE (weighted value error)
  - nn_advantage_cossim (cosine similarity of advantage vector to ground truth)
  - nn_greedy_performance (greedy policy expected return from start state)
  - nn_greedy_correct (fraction of states matching optimal greedy action)
  - total_loss / value_loss

Computes Normalized AUC (trapezoidal integration over steps) for each run,
identifies best LR per (k, algo) condition, and saves:
  - out.pkl (full trajectories for all seeds, lrs, algos, and k values)
  - summary_auc.csv (aggregated AUC table for this environment)
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
from random_policy.exact_mc import make_train as make_mc_train
from random_policy.exact_td import make_train as make_td0_train
from random_policy.exact_E_gd import make_train as make_e0_train
from random_policy.exact_td_lambda import make_train as make_td_lambda_train
from random_policy.exact_E_lambda import make_train as make_e_lambda_train

# Universal trapezoid integration compatible with all numpy versions
trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))

ALL_ENVS = [
    "fourrooms-dense",
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
    "mc": {
        "display_name": "MC (Exact)",
        "train_builder": make_mc_train,
        "hparams": {},
        "color": "#1f77b4",
    },
    "td0": {
        "display_name": "TD(0) [Exact]",
        "train_builder": make_td0_train,
        "hparams": {},
        "color": "#ff7f0e",
    },
    "e0": {
        "display_name": "E(0) [Exact]",
        "train_builder": make_e0_train,
        "hparams": {},
        "color": "#2ca02c",
    },
    "td_lambda": {
        "display_name": "TD(λ=0.8) [Exact]",
        "train_builder": make_td_lambda_train,
        "hparams": {"VALUE_LAMBDA": 0.8},
        "color": "#d62728",
    },
    "e_lambda": {
        "display_name": "E(λ=0.8) [Exact]",
        "train_builder": make_e_lambda_train,
        "hparams": {"VALUE_LAMBDA": 0.8},
        "color": "#9467bd",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Exact Random Policy Evaluation Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of exact gradient updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k_values", type=int, nargs="+", default=[8, 64], help="Feature dimensions to evaluate (default: 8 64)")
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.0003, 0.001, 0.003], help="Learning rates to sweep (default: 0.0003 0.001 0.003)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--val_lambda", type=float, default=0.8, help="Lambda value for TD(lambda) and E(lambda)")
    parser.add_argument("--out_dir", type=str, default="results/exact_random_policy_sweep", help="Root output directory")
    return parser.parse_args()


def build_env_config(env_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Configures environment parameters matching exact tabular matrix simulator standards."""
    is_mc = "mountaincar" in env_name.lower()
    cfg = base_cfg.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": 1,
        "NUM_ENVS": 1,
        "MINIBATCH_SIZE": 1,
        "NUM_EPOCHS": 1,
        "TOTAL_TIMESTEPS": args.num_updates,
        "NUM_UPDATES": args.num_updates,
        "N_SEEDS": args.n_seeds,
        "SEED": args.seed,
        "GAMMA": args.gamma,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
    })
    return cfg


def compute_skew_symmetry_ratio(env_name: str, config: Dict[str, Any]) -> float:
    """Computes Frobenius norm ratio ||K||_F / ||S||_F for the environment under uniform random policy."""
    try:
        env, env_params = helpers.make_env(config)
        evaluator = helpers.initialize_evaluator(config, env, env_params)
        n_states = len(evaluator.obs_stack)
        n_actions = env.action_space(env_params).n
        pi_active = jnp.ones((n_states, n_actions)) / n_actions
        pi_terminal = jnp.ones((1, n_actions)) / n_actions
        Pi = np.array(jnp.vstack([pi_active, pi_terminal]))

        P = np.array(evaluator.P)
        gamma = float(config.get("GAMMA", 0.99))
        n_total = P.shape[0]

        P_pi = np.einsum("sa,sam->sm", Pi[:n_total, :], P)
        mu_raw, _ = evaluator.compute_stationary_distribution_raw(Pi[:n_states, :])
        mu = np.array(mu_raw)
        mu = np.maximum(mu, 0.0)
        if np.sum(mu) > 0:
            mu = mu / np.sum(mu)
        else:
            mu = np.ones(n_states) / n_states

        D = np.diag(mu[:n_states])
        P_pi_cont = P_pi[:n_states, :n_states]
        A = D @ (np.eye(n_states) - gamma * P_pi_cont)
        S = 0.5 * (A + A.T)
        K = 0.5 * (A - A.T)
        norm_s = float(np.linalg.norm(S, ord="fro"))
        norm_k = float(np.linalg.norm(K, ord="fro"))
        return float(norm_k / (norm_s + 1e-12))
    except Exception as e:
        print(f"[Warning] Could not compute random policy skew symmetry: {e}")
        return np.nan


def compute_normalized_auc(trajectory_1d: np.ndarray) -> float:
    """Computes trapezoidal Area Under the Curve normalized by steps (average metric value)."""
    if len(trajectory_1d) < 2:
        return float(trajectory_1d[0])
    x = np.arange(len(trajectory_1d))
    return float(trapz_fn(trajectory_1d, x) / (len(trajectory_1d) - 1))


def run_condition(
    train_builder,
    config: Dict[str, Any],
    algo_hparams: Dict[str, Any],
    k_val: int,
    lr_val: float,
    n_seeds: int,
    base_seed: int,
) -> Dict[str, np.ndarray]:
    """Runs a single (algo, k, lr) condition across random seeds."""
    run_cfg = config.copy()
    run_cfg.update(algo_hparams)
    run_cfg["k"] = k_val
    run_cfg["LR"] = lr_val

    train_fn = train_builder(run_cfg)
    seed_metrics = []

    for s_idx in range(n_seeds):
        rng = jax.random.PRNGKey(base_seed + s_idx)
        t0 = time.time()
        res = train_fn(rng)
        dt = time.time() - t0
        np_metrics = jax.tree.map(lambda x: np.asarray(x), res["metrics"])
        seed_metrics.append(np_metrics)

    # Stack metrics across seeds: (n_seeds, num_updates)
    keys = seed_metrics[0].keys()
    stacked = {}
    for k in keys:
        try:
            stacked[k] = np.stack([sm[k] for sm in seed_metrics], axis=0)
        except Exception:
            pass

    return stacked


def main():
    args = parse_args()

    if args.env_idx is not None and 0 <= args.env_idx < len(ALL_ENVS):
        env_name = ALL_ENVS[args.env_idx]
    else:
        env_name = args.env_name

    out_dir = os.path.abspath(args.out_dir)
    env_out_dir = os.path.join(out_dir, env_name)
    os.makedirs(env_out_dir, exist_ok=True)

    # Update lambda parameter if provided
    ALGORITHMS["td_lambda"]["hparams"]["VALUE_LAMBDA"] = args.val_lambda
    ALGORITHMS["td_lambda"]["display_name"] = f"TD(λ={args.val_lambda}) [Exact]"
    ALGORITHMS["e_lambda"]["hparams"]["VALUE_LAMBDA"] = args.val_lambda
    ALGORITHMS["e_lambda"]["display_name"] = f"E(λ={args.val_lambda}) [Exact]"

    print("=" * 75)
    print("Exact (In-Expectation) Random Policy Evaluation Sweep")
    print(f"Environment:       {env_name}")
    print(f"Num Updates:       {args.num_updates}")
    print(f"Num Seeds:         {args.n_seeds} (base seed: {args.seed})")
    print(f"k Values:          {args.k_values}")
    print(f"Learning Rates:    {args.lrs}")
    print(f"Algorithms:        {list(ALGORITHMS.keys())}")
    print(f"Output Directory:  {env_out_dir}")
    print("=" * 75)

    base_env_cfg = build_env_config(env_name, args)
    k_over_s_ratio = compute_skew_symmetry_ratio(env_name, base_env_cfg)
    print(f"Degree of Skew Symmetry (||K||_F / ||S||_F): {k_over_s_ratio:.4f}")

    # Storage structure:
    # results[k][algo][lr] = dict of stacked metrics (n_seeds, num_updates)
    results: Dict[int, Dict[str, Dict[float, Dict[str, np.ndarray]]]] = {k: {} for k in args.k_values}
    summary_rows = []

    total_conditions = len(args.k_values) * len(ALGORITHMS) * len(args.lrs)
    cond_counter = 0

    for k_val in args.k_values:
        for algo_key, algo_info in ALGORITHMS.items():
            results[k_val][algo_key] = {}
            lr_aucs_ve = []
            lr_metric_data = {}

            for lr_val in args.lrs:
                cond_counter += 1
                disp = f"[{cond_counter}/{total_conditions}] k={k_val} | {algo_key} | lr={lr_val}"
                print(f"--> Running {disp} ...", end="", flush=True)

                t0 = time.time()
                metrics = run_condition(
                    train_builder=algo_info["train_builder"],
                    config=base_env_cfg,
                    algo_hparams=algo_info["hparams"],
                    k_val=k_val,
                    lr_val=lr_val,
                    n_seeds=args.n_seeds,
                    base_seed=args.seed,
                )
                dt = time.time() - t0

                results[k_val][algo_key][lr_val] = metrics
                lr_metric_data[lr_val] = metrics

                # Compute normalized AUC for VE across seeds
                ve_trajectories = metrics.get("nn_weighted_VE", None)
                if ve_trajectories is not None:
                    seed_aucs = [compute_normalized_auc(ve_trajectories[s]) for s in range(args.n_seeds)]
                    mean_auc = float(np.mean(seed_aucs))
                    sem_auc = float(np.std(seed_aucs) / np.sqrt(max(1, args.n_seeds)))
                    final_ve_mean = float(np.mean(ve_trajectories[:, -1]))
                    final_ve_sem = float(np.std(ve_trajectories[:, -1]) / np.sqrt(max(1, args.n_seeds)))
                else:
                    mean_auc, sem_auc, final_ve_mean, final_ve_sem = np.nan, np.nan, np.nan, np.nan

                # Compute normalized AUC for advantage cosine similarity
                cos_trajectories = metrics.get("nn_advantage_cossim", None)
                if cos_trajectories is not None:
                    seed_cos_aucs = [compute_normalized_auc(cos_trajectories[s]) for s in range(args.n_seeds)]
                    mean_cos_auc = float(np.mean(seed_cos_aucs))
                    sem_cos_auc = float(np.std(seed_cos_aucs) / np.sqrt(max(1, args.n_seeds)))
                    final_cos_mean = float(np.mean(cos_trajectories[:, -1]))
                else:
                    mean_cos_auc, sem_cos_auc, final_cos_mean = np.nan, np.nan, np.nan

                # Compute greedy policy performance
                perf_trajectories = metrics.get("nn_greedy_performance", None)
                if perf_trajectories is not None:
                    seed_perf_aucs = [compute_normalized_auc(perf_trajectories[s]) for s in range(args.n_seeds)]
                    mean_perf_auc = float(np.mean(seed_perf_aucs))
                    sem_perf_auc = float(np.std(seed_perf_aucs) / np.sqrt(max(1, args.n_seeds)))
                    final_perf_mean = float(np.mean(perf_trajectories[:, -1]))
                else:
                    mean_perf_auc, sem_perf_auc, final_perf_mean = np.nan, np.nan, np.nan

                lr_aucs_ve.append((lr_val, mean_auc, sem_auc, mean_cos_auc, sem_cos_auc, final_ve_mean, final_ve_sem, mean_perf_auc, sem_perf_auc, final_perf_mean))
                print(f" done in {dt:.1f}s | VE AUC: {mean_auc:.4e} | Final VE: {final_ve_mean:.4e}")

            # Determine best LR based on minimum normalized AUC of Value Error
            valid_aucs = [item for item in lr_aucs_ve if not np.isnan(item[1])]
            if valid_aucs:
                best_item = min(valid_aucs, key=lambda x: x[1])
                best_lr, best_auc_ve, best_sem_ve, best_auc_cos, best_sem_cos, best_fin_ve, best_fin_sem, best_auc_perf, best_sem_perf, best_fin_perf = best_item
                summary_rows.append({
                    "env": env_name,
                    "k": k_val,
                    "algo": algo_key,
                    "display_name": algo_info["display_name"],
                    "k_over_s_ratio": k_over_s_ratio,
                    "best_lr": best_lr,
                    "auc_ve_mean": best_auc_ve,
                    "auc_ve_sem": best_sem_ve,
                    "auc_cossim_mean": best_auc_cos,
                    "auc_cossim_sem": best_sem_cos,
                    "final_ve_mean": best_fin_ve,
                    "final_ve_sem": best_fin_sem,
                    "auc_greedy_perf": best_auc_perf,
                    "auc_greedy_perf_sem": best_sem_perf,
                    "final_greedy_perf": best_fin_perf,
                })

    # Save full results dictionary
    out_pkl = os.path.join(env_out_dir, "out.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "env_name": env_name,
            "k_values": args.k_values,
            "lrs": args.lrs,
            "algorithms": list(ALGORITHMS.keys()),
            "data": results,
            "args": vars(args),
        }, f)
    print(f"\n[Saved] Full trajectory dataset -> {out_pkl}")

    # Save summary dataframe
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(env_out_dir, "summary_auc.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] Summary AUC table -> {summary_csv}")
    print("\nEnvironment Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
