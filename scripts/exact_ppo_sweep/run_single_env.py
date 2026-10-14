"""
scripts/exact_ppo_sweep/run_single_env.py

Runs the 5 PPO exact algorithms on a single environment across multiple random seeds:
- MC (Exact Critic)
- TD(0) [Exact Critic]
- E(0) [Exact Critic]
- TD(λ=0.8) [Exact Critic]
- E(λ=0.8) [Exact Critic]

Key specifications:
- Single critic feature dimension: k=32
- GAE lambda: 0.2 (low GAE lambda ensures advantage relies heavily on learned critic)
- Slow actor learning: ACTOR_LR=0.0001
- Faster critic learning: CRITIC_LR=0.001
- PPO ratio clipping: CLIP_EPS=0.2 (somewhat high clipping to permit policy shifts)
- Standardized performance metric:
    V_bar = (V_start - V_rand) / (V_star - V_rand)
  where V_rand and V_star are loaded from reference/env_performance_baselines.json.
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
from ppo.exact_mc import make_train as make_mc_train
from ppo.exact_td import make_train as make_td0_train
from ppo.exact_E import make_train as make_e0_train
from ppo.exact_td_lambda import make_train as make_td_lambda_train
from ppo.exact_E_lambda import make_train as make_e_lambda_train

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
    "mc": {
        "display_name": "MC (Exact Critic)",
        "train_builder": make_mc_train,
        "hparams": {},
        "color": "#1f77b4",
    },
    "td0": {
        "display_name": "TD(0) [Exact Critic]",
        "train_builder": make_td0_train,
        "hparams": {},
        "color": "#ff7f0e",
    },
    "e0": {
        "display_name": "E(0) [Exact Critic]",
        "train_builder": make_e0_train,
        "hparams": {},
        "color": "#2ca02c",
    },
    "td_lambda": {
        "display_name": "TD(λ=0.8) [Exact Critic]",
        "train_builder": make_td_lambda_train,
        "hparams": {"VALUE_LAMBDA": 0.8},
        "color": "#d62728",
    },
    "e_lambda": {
        "display_name": "E(λ=0.8) [Exact Critic]",
        "train_builder": make_e_lambda_train,
        "hparams": {"VALUE_LAMBDA": 0.8},
        "color": "#9467bd",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Exact PPO Evaluation Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of PPO updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k", type=int, default=32, help="Critic feature dimension (default: 32)")
    parser.add_argument("--actor_lr", type=float, default=0.0001, help="Actor learning rate (default: 0.0001)")
    parser.add_argument("--critic_lrs", type=float, nargs="+", default=[0.001, 0.005, 0.02], help="Critic learning rates to sweep (default: 0.001 0.005 0.02)")
    parser.add_argument("--gae_lambda", type=float, default=0.1, help="GAE lambda (default: 0.1)")
    parser.add_argument("--clip_eps", type=float, default=0.05, help="PPO clip epsilon (default: 0.05)")
    parser.add_argument("--val_lambda", type=float, default=0.8, help="Value lambda for TD(lambda) and E(lambda)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--num_epochs", type=int, default=4, help="PPO epochs per update (default: 4)")
    parser.add_argument("--out_dir", type=str, default="results/exact_ppo_sweep", help="Root output directory")
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
    """Configures environment parameters matching PPO exact evaluation standards."""
    is_mc = "mountaincar" in env_name.lower()
    cfg = base_cfg.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": 1,
        "NUM_ENVS": 1,
        "MINIBATCH_SIZE": 1,
        "NUM_EPOCHS": args.num_epochs,
        "TOTAL_TIMESTEPS": args.num_updates,
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
        "VALUE_LAMBDA": args.val_lambda,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
    })
    return cfg


def compute_normalized_auc(trajectory_1d: np.ndarray) -> float:
    """Computes trapezoidal Area Under the Curve normalized by steps."""
    if len(trajectory_1d) < 2:
        return float(trajectory_1d[0])
    x = np.arange(len(trajectory_1d))
    return float(trapz_fn(trajectory_1d, x) / (len(trajectory_1d) - 1))


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

    print("=" * 75)
    print("Exact PPO Control Sweep on Single Environment")
    print(f"Environment:       {args.env_name}")
    print(f"Num Updates:       {args.num_updates}")
    print(f"Num Seeds:         {args.n_seeds} (base seed: {args.seed})")
    print(f"Critic Feature k:  {args.k}")
    print(f"Actor LR:          {args.actor_lr}")
    print(f"Critic LRs:        {args.critic_lrs}")
    print(f"GAE Lambda:        {args.gae_lambda}")
    print(f"Clip Epsilon:      {args.clip_eps}")
    print(f"Algorithms:        {list(ALGORITHMS.keys())}")
    print(f"Baselines:         V* = {v_star:.4f} | V_rand = {v_rand:.4f} | Delta = {delta:.4f}")
    out_env_dir = os.path.join(os.path.abspath(args.out_dir), args.env_name)
    os.makedirs(out_env_dir, exist_ok=True)
    print(f"Output Directory:  {out_env_dir}")
    print("=" * 75)

    full_results = {
        "env_name": args.env_name,
        "k": args.k,
        "args": vars(args),
        "baselines": {"v_star": v_star, "v_rand": v_rand, "delta": delta},
        "runs": {},          # winning run per algo (for direct plot compatibility)
        "all_runs": {},      # all runs: algo -> lr -> res
        "best_critic_lrs": {},
    }

    summary_rows = []
    all_lrs_rows = []

    for algo_key, algo_info in ALGORITHMS.items():
        print(f"\n========================================================")
        print(f"Algorithm: {algo_key} ({algo_info['display_name']})")
        print(f"========================================================")

        full_results["all_runs"][algo_key] = {}
        lr_candidates = []

        for c_lr in args.critic_lrs:
            print(f"  --> Testing Critic LR = {c_lr} ...", end="", flush=True)
            env_cfg = build_env_config(args.env_name, args, critic_lr=c_lr)

            res = run_algo_condition(
                train_builder=algo_info["train_builder"],
                config=env_cfg,
                algo_hparams=algo_info["hparams"],
                n_seeds=args.n_seeds,
                base_seed=args.seed,
            )

            dt = res["_runtime_sec"]
            v_start_seeds = res.get("V_start")  # shape: (n_seeds, num_updates)

            if v_start_seeds is not None:
                std_perf_seeds = (v_start_seeds - v_rand) / (delta + 1e-12)
                res["standardized_v_start"] = std_perf_seeds

                final_perf_per_seed = std_perf_seeds[:, -1]
                auc_perf_per_seed = np.array([compute_normalized_auc(s) for s in std_perf_seeds])

                mean_final = float(np.mean(final_perf_per_seed))
                sem_final = float(np.std(final_perf_per_seed, ddof=1) / np.sqrt(args.n_seeds)) if args.n_seeds > 1 else 0.0

                mean_auc = float(np.mean(auc_perf_per_seed))
                sem_auc = float(np.std(auc_perf_per_seed, ddof=1) / np.sqrt(args.n_seeds)) if args.n_seeds > 1 else 0.0

                raw_final_mean = float(np.mean(v_start_seeds[:, -1]))
            else:
                mean_final = sem_final = mean_auc = sem_auc = raw_final_mean = np.nan

            full_results["all_runs"][algo_key][c_lr] = res

            row_data = {
                "env": args.env_name,
                "algo": algo_key,
                "display_name": algo_info["display_name"],
                "k": args.k,
                "critic_lr": c_lr,
                "actor_lr": args.actor_lr,
                "gae_lambda": args.gae_lambda,
                "clip_eps": args.clip_eps,
                "runtime_sec": dt,
                "v_start_final_mean": raw_final_mean,
                "std_perf_final_mean": mean_final,
                "std_perf_final_sem": sem_final,
                "std_perf_auc_mean": mean_auc,
                "std_perf_auc_sem": sem_auc,
            }
            all_lrs_rows.append(row_data)
            lr_candidates.append((c_lr, mean_auc, sem_auc, mean_final, sem_final, raw_final_mean, res, dt))

            print(f" done in {dt:.1f}s | AUC Std Perf: {mean_auc:.4f} ± {sem_auc:.4f} | Final: {mean_final:.4f}")

        # Select best critic LR based on maximum AUC of standardized performance
        valid_candidates = [c for c in lr_candidates if not np.isnan(c[1])]
        if valid_candidates:
            best_cand = max(valid_candidates, key=lambda x: x[1])
            best_lr, best_auc, best_auc_sem, best_fin, best_fin_sem, best_raw_fin, best_res, best_dt = best_cand
            print(f"  [Selected Best Critic LR] {best_lr} (AUC: {best_auc:.4f})")
        else:
            best_lr, best_auc, best_auc_sem, best_fin, best_fin_sem, best_raw_fin, best_res, best_dt = lr_candidates[0]

        full_results["runs"][algo_key] = best_res
        full_results["best_critic_lrs"][algo_key] = best_lr

        summary_rows.append({
            "env": args.env_name,
            "algo": algo_key,
            "display_name": algo_info["display_name"],
            "k": args.k,
            "best_critic_lr": best_lr,
            "critic_lr": best_lr,
            "actor_lr": args.actor_lr,
            "gae_lambda": args.gae_lambda,
            "clip_eps": args.clip_eps,
            "runtime_sec": best_dt,
            "v_start_final_mean": best_raw_fin,
            "std_perf_final_mean": best_fin,
            "std_perf_final_sem": best_fin_sem,
            "std_perf_auc_mean": best_auc,
            "std_perf_auc_sem": best_auc_sem,
        })

    # Save summary dataframes
    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_env_dir, "summary.csv")
    df_summary.to_csv(csv_path, index=False)

    df_all_lrs = pd.DataFrame(all_lrs_rows)
    all_lrs_csv = os.path.join(out_env_dir, "summary_all_lrs.csv")
    df_all_lrs.to_csv(all_lrs_csv, index=False)

    # Save out.pkl
    pkl_path = os.path.join(out_env_dir, "out.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(full_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save config.json
    cfg_path = os.path.join(out_env_dir, "config.json")
    base_env_cfg = build_env_config(args.env_name, args, critic_lr=args.critic_lrs[0])
    serializable_cfg = {k: v for k, v in base_env_cfg.items() if isinstance(v, (int, float, str, bool, list, dict)) or v is None}
    with open(cfg_path, "w") as f:
        json.dump(serializable_cfg, f, indent=4)

    print("\n" + "=" * 75)
    print(f"Sweep Completed for '{args.env_name}'")
    print(f"Saved Checkpoint: {pkl_path}")
    print(f"Saved Summary:    {csv_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
