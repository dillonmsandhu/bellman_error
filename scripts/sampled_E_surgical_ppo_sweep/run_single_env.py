"""
scripts/sampled_E_surgical_ppo_sweep/run_single_env.py

Executes full Sampled PPO across 4 critic algorithms:
  1. mc:               Monte Carlo trajectory baseline
  2. surgical_2heads:  Surgical v_aux (2 Heads on Shared Torso)
  3. surgical_2nets:   Surgical v_aux (2 Independent Networks)
  4. oct8_biased:      OG Oct 8 (Biased Start-Anchored)

Sweeps critic learning rates across candidate values, selecting condition by
standardized episodic return AUC (std_perf_auc), breaking ties with final return.

Outputs:
  - out.pkl
  - summary_auc.csv
  - summary.csv
  - summary_all_lrs.csv
  - lr_selection_summary.json
  - config.json
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

# Algorithms
from ppo.sampled_mc import make_train as make_mc_train
from ppo.sampled_td_lambda import make_train as make_td_train
from ppo.sampled_E import make_train as make_sampled_e_train

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
        "display_name": "MC (Sampled)",
        "train_builder": make_mc_train,
        "hparams": {"VALUE_LAMBDA": 1.0},
        "color": "#1f77b4",  # Steel Blue
    },
    "td0": {
        "display_name": "TD(0) [Sampled]",
        "train_builder": make_td_train,
        "hparams": {"VALUE_LAMBDA": 0.0},
        "color": "#ff7f0e",  # Safety Orange
    },
    "oct8_biased": {
        "display_name": "OG Oct 8 (Biased)",
        "train_builder": make_sampled_e_train,
        "hparams": {"LOSS_VARIANT": "oct8", "RETURN_LAMBDA": 1.0},
        "color": "#d62728",  # Crimson / Red
    },
    "forward_e": {
        "display_name": "Forward-Only E",
        "train_builder": make_sampled_e_train,
        "hparams": {"LOSS_VARIANT": "forward_e", "RETURN_LAMBDA": 1.0},
        "color": "#00bcd4",  # Cyan / Teal
    },
    "surgical_v_aux": {
        "display_name": "Surgical V_aux",
        "train_builder": make_sampled_e_train,
        "hparams": {"LOSS_VARIANT": "surgical_v_aux", "SHARED_TORSO": False, "RETURN_LAMBDA": 1.0, "AUX_LAMBDA": 1.0, "AUX_VF_COEF": 1.0},
        "color": "#2ca02c",  # Forest Green
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Full Sampled E-Surgical PPO Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of rollout updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k", type=int, default=32, help="Critic feature dimension (default: 32)")
    parser.add_argument("--actor_lr", type=float, default=0.0003, help="Actor learning rate (default: 0.0003)")
    parser.add_argument("--critic_lrs", type=float, nargs="+", default=[0.0005, 0.001, 0.003], help="Critic learning rates to sweep")
    parser.add_argument("--num_envs", type=int, default=32, help="Parallel environments (default: 32)")
    parser.add_argument("--num_steps", type=int, default=128, help="Steps per rollout (default: 128)")
    parser.add_argument("--minibatch_size", type=int, default=256, help="Minibatch size (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Epochs per update (default: 4)")
    parser.add_argument("--gae_lambda", type=float, default=0.8, help="Actor GAE lambda (default: 0.8)")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon (default: 0.2)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--out_dir", type=str, default="results/sampled_E_surgical_ppo_sweep", help="Root output directory")
    return parser.parse_args()


def compute_normalized_auc(series: np.ndarray) -> float:
    """Computes area under curve normalized by series length."""
    arr = np.asarray(series, dtype=np.float32)
    valid_mask = np.isfinite(arr)
    if not np.any(valid_mask):
        return float("nan")
    if not np.all(valid_mask):
        arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
    n = len(arr)
    if n <= 1:
        return float(arr[0])
    return float(trapz_fn(arr, dx=1.0) / (n - 1))


def is_statistically_tied(mean_a: float, sem_a: float, mean_b: float, sem_b: float, threshold_sems: float = 1.0) -> bool:
    if not np.isfinite(mean_a) or not np.isfinite(mean_b):
        return False
    pooled_sem = np.sqrt(max(0.0, sem_a)**2 + max(0.0, sem_b)**2)
    if pooled_sem <= 1e-12:
        return np.isclose(mean_a, mean_b, atol=1e-6)
    return abs(mean_a - mean_b) <= threshold_sems * pooled_sem


def load_env_baseline(env_name: str) -> tuple[float, float, float]:
    json_path = os.path.join(REPO_ROOT, "reference", "env_performance_baselines.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if env_name in data:
                v_star = float(data[env_name]["v_star"])
                v_rand = float(data[env_name]["v_rand"])
                delta = float(data[env_name]["delta"])
                return v_star, v_rand, delta
        except Exception:
            pass
    return 1.0, 0.0, 1.0


def build_env_config(env_name: str, args, critic_lr: float) -> Dict[str, Any]:
    cfg = base_cfg.copy()
    cfg["ENV_NAME"] = env_name
    cfg["NUM_UPDATES"] = args.num_updates
    cfg["NUM_ENVS"] = args.num_envs
    cfg["NUM_STEPS"] = args.num_steps
    cfg["TOTAL_TIMESTEPS"] = args.num_updates * args.num_envs * args.num_steps
    cfg["MINIBATCH_SIZE"] = args.minibatch_size
    cfg["NUM_MINIBATCHES"] = max(1, (args.num_envs * args.num_steps) // args.minibatch_size)
    cfg["NUM_EPOCHS"] = args.num_epochs
    cfg["ACTOR_LR"] = args.actor_lr
    cfg["CRITIC_LR"] = critic_lr
    cfg["LR"] = critic_lr
    cfg["k"] = args.k
    cfg["GAE_LAMBDA"] = args.gae_lambda
    cfg["CLIP_EPS"] = args.clip_eps
    cfg["GAMMA"] = args.gamma
    cfg["CALC_TRUE_VALUES"] = True
    cfg["LIGHT_METRICS"] = True
    cfg["LOG_FEATURE_METRICS"] = False
    return cfg


def run_experiment_for_lr(
    algo_key: str,
    algo_spec: Dict[str, Any],
    env_name: str,
    args,
    critic_lr: float,
    seeds: List[int],
    v_rand: float,
    delta: float,
) -> Dict[str, Any]:
    t0 = time.time()
    cfg = build_env_config(env_name, args, critic_lr)
    cfg.update(algo_spec["hparams"])

    train_fn = algo_spec["train_builder"](cfg)

    seed_metrics = []
    for s_idx, s in enumerate(seeds):
        rng = jax.random.PRNGKey(s)
        out = train_fn(rng)
        metrics = out["metrics"]
        
        np_metrics = {}
        for m_name, m_val in metrics.items():
            arr = np.asarray(m_val)
            if arr.ndim > 1:
                arr = arr.squeeze()
            np_metrics[m_name] = arr.astype(np.float32)

        # Empirical discounted returns: G_0 = sum gamma^t r_t from rollouts
        disc_ret = np_metrics.get("returned_episode_discounted_returns", np_metrics.get("returned_discounted_episode_returns"))
        if disc_ret is not None:
            np_metrics["returned_episode_discounted_returns"] = disc_ret
            np_metrics["returned_discounted_episode_returns"] = disc_ret
            np_metrics["std_discounted_return"] = (disc_ret - v_rand) / (delta + 1e-12)

        # Standardized policy performance: ONLY computed from true analytical V_start if present
        if "V_start" in np_metrics:
            np_metrics["std_perf"] = (np_metrics["V_start"] - v_rand) / (delta + 1e-12)
            np_metrics["standardized_v_start"] = np_metrics["std_perf"]

        seed_metrics.append(np_metrics)

    elapsed = time.time() - t0
    metric_keys = seed_metrics[0].keys()
    stacked_metrics = {}
    auc_scores = {}
    final_scores = {}

    for k_m in metric_keys:
        stacked = np.stack([sm[k_m] for sm in seed_metrics], axis=0)
        stacked_metrics[k_m] = stacked
        
        aucs = [compute_normalized_auc(stacked[i]) for i in range(len(seeds))]
        auc_scores[k_m] = {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
            "sem": float(np.std(aucs) / np.sqrt(len(seeds))),
            "values": aucs,
        }
        
        finals = stacked[:, -1]
        final_scores[k_m] = {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals)),
            "sem": float(np.std(finals) / np.sqrt(len(seeds))),
            "values": finals.tolist(),
        }

    return {
        "algo": algo_key,
        "k": args.k,
        "critic_lr": critic_lr,
        "stacked_metrics": stacked_metrics,
        "auc_scores": auc_scores,
        "final_scores": final_scores,
        "elapsed_seconds": elapsed,
    }


def select_best_ppo_run(lr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Selects best condition by std_perf AUC (if true V_start present) or returned_episode_discounted_returns AUC."""
    if len(lr_results) == 1:
        return lr_results[0]

    criterion = "std_perf" if "std_perf" in lr_results[0]["auc_scores"] else "returned_episode_discounted_returns"
    sorted_by_auc = sorted(lr_results, key=lambda r: r["auc_scores"].get(criterion, {}).get("mean", 0.0), reverse=True)
    top_auc = sorted_by_auc[0]["auc_scores"].get(criterion, {}).get("mean", 0.0)
    top_sem = sorted_by_auc[0]["auc_scores"].get(criterion, {}).get("sem", 0.0)

    tied = [
        r for r in sorted_by_auc
        if is_statistically_tied(top_auc, top_sem, r["auc_scores"].get(criterion, {}).get("mean", 0.0), r["auc_scores"].get(criterion, {}).get("sem", 0.0))
    ]

    if len(tied) == 1:
        return tied[0]

    # Tie breaker: final score on same criterion
    return max(tied, key=lambda r: r["final_scores"].get(criterion, {}).get("mean", 0.0))


def main():
    args = parse_args()

    if args.env_idx is not None:
        if 0 <= args.env_idx < len(ALL_ENVS):
            args.env_name = ALL_ENVS[args.env_idx]
        else:
            raise ValueError(f"env_idx {args.env_idx} out of range [0, {len(ALL_ENVS)-1}]")

    env_name = args.env_name
    print("=" * 80)
    print(f"FULL SAMPLED E-SURGICAL PPO SWEEP: {env_name}")
    print(f"Algorithms: {list(ALGORITHMS.keys())}")
    print(f"Critic Feature Dimension k: {args.k}")
    print(f"Critic Learning Rates: {args.critic_lrs}")
    print(f"Actor LR: {args.actor_lr} | GAE Lambda: {args.gae_lambda}")
    print(f"Seeds: {args.n_seeds} (base: {args.seed}) | Updates: {args.num_updates}")
    print("=" * 80)

    v_star, v_rand, delta = load_env_baseline(env_name)

    env_out_dir = os.path.join(args.out_dir, env_name)
    os.makedirs(env_out_dir, exist_ok=True)
    figures_dir = os.path.join(args.out_dir, "figures")
    tables_dir = os.path.join(args.out_dir, "analysis_tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]

    summary_rows = []
    lr_tuning_records = []
    results_by_algo = {a: {} for a in ALGORITHMS.keys()}

    total_start_time = time.time()

    for algo_key, algo_spec in ALGORITHMS.items():
        print(f"\n--- Algorithm: {algo_spec['display_name']} ---")
        lr_results = []
        for critic_lr in args.critic_lrs:
            print(f"  Sweeping critic_lr = {critic_lr} across {len(seeds)} seeds...", end="", flush=True)
            run_res = run_experiment_for_lr(
                algo_key=algo_key,
                algo_spec=algo_spec,
                env_name=env_name,
                args=args,
                critic_lr=critic_lr,
                seeds=seeds,
                v_rand=v_rand,
                delta=delta,
            )
            std_auc = run_res["auc_scores"]["std_perf"]["mean"]
            std_final = run_res["final_scores"]["std_perf"]["mean"]
            vstart_auc = run_res["auc_scores"].get("v_pred_start", {}).get("mean", np.nan)
            print(f" Done! Std Perf AUC: {std_auc:.4f} | Final: {std_final:.4f} | V_start AUC: {vstart_auc:.4f} ({run_res['elapsed_seconds']:.1f}s)")
            lr_results.append(run_res)
            
            lr_tuning_records.append({
                "env_name": env_name,
                "algo": algo_key,
                "display_name": algo_spec["display_name"],
                "critic_lr": critic_lr,
                "mean_std_perf_auc": std_auc,
                "sem_std_perf_auc": run_res["auc_scores"]["std_perf"]["sem"],
                "mean_std_perf_final": std_final,
                "sem_std_perf_final": run_res["final_scores"]["std_perf"]["sem"],
                "mean_vstart_auc": vstart_auc,
            })

        best_run = select_best_ppo_run(lr_results)
        best_lr = best_run["critic_lr"]
        print(f"  ==> Selected Best LR for {algo_key}: {best_lr} (Std Perf AUC: {best_run['auc_scores']['std_perf']['mean']:.4f})")

        results_by_algo[algo_key][args.k] = {
            "best_lr": best_lr,
            "metrics": best_run["stacked_metrics"],
            "auc_scores": best_run["auc_scores"],
            "final_scores": best_run["final_scores"],
            "lr_sweep": {
                r["critic_lr"]: {
                    "std_perf_auc_mean": r["auc_scores"]["std_perf"]["mean"],
                    "std_perf_auc_sem": r["auc_scores"]["std_perf"]["sem"],
                    "std_perf_final_mean": r["final_scores"]["std_perf"]["mean"],
                }
                for r in lr_results
            },
        }

        perf_auc = best_run["auc_scores"].get("std_perf", {})
        perf_final = best_run["final_scores"].get("std_perf", {})
        true_vstart_auc = best_run["auc_scores"].get("V_start", {})
        true_vstart_final = best_run["final_scores"].get("V_start", {})
        vpred_auc = best_run["auc_scores"].get("v_pred_start", {})
        vpred_final = best_run["final_scores"].get("v_pred_start", {})
        disc_ret_auc = best_run["auc_scores"].get("returned_episode_discounted_returns", {})
        disc_ret_final = best_run["final_scores"].get("returned_episode_discounted_returns", {})

        summary_rows.append({
            "env_name": env_name,
            "k": args.k,
            "algo": algo_key,
            "display_name": algo_spec["display_name"],
            "best_critic_lr": best_lr,
            "std_perf_auc_mean": perf_auc.get("mean", np.nan),
            "std_perf_auc_sem": perf_auc.get("sem", np.nan),
            "std_perf_final_mean": perf_final.get("mean", np.nan),
            "std_perf_final_sem": perf_final.get("sem", np.nan),
            "true_vstart_auc_mean": true_vstart_auc.get("mean", np.nan),
            "true_vstart_auc_sem": true_vstart_auc.get("sem", np.nan),
            "true_vstart_final_mean": true_vstart_final.get("mean", np.nan),
            "true_vstart_final_sem": true_vstart_final.get("sem", np.nan),
            "disc_return_auc_mean": disc_ret_auc.get("mean", np.nan),
            "disc_return_auc_sem": disc_ret_auc.get("sem", np.nan),
            "disc_return_final_mean": disc_ret_final.get("mean", np.nan),
            "disc_return_final_sem": disc_ret_final.get("sem", np.nan),
            "vpred_start_auc_mean": vpred_auc.get("mean", np.nan),
            "vpred_start_auc_sem": vpred_auc.get("sem", np.nan),
            "vpred_start_final_mean": vpred_final.get("mean", np.nan),
            "vpred_start_final_sem": vpred_final.get("sem", np.nan),
        })

    total_time = time.time() - total_start_time

    # Save out.pkl
    out_pkl_path = os.path.join(env_out_dir, "out.pkl")
    with open(out_pkl_path, "wb") as f:
        pickle.dump(results_by_algo, f)
    print(f"\n[Saved] Full experiment results saved to: {out_pkl_path}")

    # Save summary.csv
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(env_out_dir, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Summary table saved to: {summary_csv_path}")

    # Save lr_tuning_summary.csv
    lr_tuning_df = pd.DataFrame(lr_tuning_records)
    lr_tuning_csv_path = os.path.join(env_out_dir, "summary_all_lrs.csv")
    lr_tuning_df.to_csv(lr_tuning_csv_path, index=False)

    # Save config.json
    cfg_to_save = {
        "env_name": env_name,
        "k": args.k,
        "critic_lrs": args.critic_lrs,
        "actor_lr": args.actor_lr,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "num_updates": args.num_updates,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "minibatch_size": args.minibatch_size,
        "num_epochs": args.num_epochs,
        "gamma": args.gamma,
        "total_elapsed_seconds": total_time,
    }
    with open(os.path.join(env_out_dir, "config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    print(f"\nAll tasks for '{env_name}' completed in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
