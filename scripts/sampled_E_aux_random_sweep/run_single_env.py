"""
scripts/sampled_E_aux_random_sweep/run_single_env.py

Executes sampled (rollout-based) random policy evaluation across 5 algorithms:
  1. mc:             Monte Carlo (random_policy.mc with GAE_LAMBDA=1.0)
  2. oct8_biased:    OG October 8 (random_policy.oct8, start-anchored Dirichlet loss, Baird's RG bias in Term 4)
  3. v_aux_full:     Full v_aux (random_policy.E_sample_v_aux_full, Oct 8 E-loss with e = sg(v_aux) - v)
  4. v_aux_bilinear: Bilinear v_aux (random_policy.E_sample_v_aux_bilinear, direct <e, delta>_D inner product)
  5. v_aux_surgical: Surgical v_aux (random_policy.E_sample_v_aux_surgical, e = G - v except Term 4 which uses v_aux)

Sweeps critic learning rates across candidate values, selecting the condition
minimizing normalized AUC of Value Error (nn_weighted_VE).

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

# Algorithms under comparison
from random_policy.mc import make_train as make_mc_train
from random_policy.oct8 import make_train as make_oct8_train
from random_policy.E_sample_v_aux_full import make_train as make_v_aux_full_train
from random_policy.E_sample_v_aux_bilinear import make_train as make_v_aux_bilinear_train
from random_policy.E_sample_v_aux_surgical import make_train as make_v_aux_surgical_train

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
        "hparams": {"GAE_LAMBDA": 1.0, "VALUE_LAMBDA": 1.0},
        "color": "#1f77b4",  # Steel Blue
    },
    "oct8_biased": {
        "display_name": "OG Oct 8 (Biased)",
        "train_builder": make_oct8_train,
        "hparams": {"RETURN_LAMBDA": 1.0, "E_VARIANT": "oct8"},
        "color": "#d62728",  # Crimson / Red
    },
    "v_aux_full": {
        "display_name": "Full v_aux (Oct 8)",
        "train_builder": make_v_aux_full_train,
        "hparams": {"RETURN_LAMBDA": 1.0, "AUX_VF_COEF": 1.0},
        "color": "#9467bd",  # Purple
    },
    "v_aux_bilinear": {
        "display_name": "Bilinear v_aux",
        "train_builder": make_v_aux_bilinear_train,
        "hparams": {"RETURN_LAMBDA": 1.0, "AUX_VF_COEF": 1.0},
        "color": "#ff7f0e",  # Orange
    },
    "v_aux_surgical": {
        "display_name": "Surgical v_aux",
        "train_builder": make_v_aux_surgical_train,
        "hparams": {"RETURN_LAMBDA": 1.0, "AUX_VF_COEF": 1.0},
        "color": "#2ca02c",  # Forest Green
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sampled E-Aux Random Policy Evaluation Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of rollout updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k_values", type=int, nargs="+", default=[8, 64], help="Feature dimensions to evaluate (default: 8 64)")
    parser.add_argument("--critic_lrs", type=float, nargs="+", default=[0.0005, 0.001, 0.003], help="Critic learning rates to sweep (default: 0.0005 0.001 0.003)")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments (default: 16)")
    parser.add_argument("--num_steps", type=int, default=64, help="Steps per rollout (default: 64)")
    parser.add_argument("--minibatch_size", type=int, default=256, help="Minibatch size (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Epochs per update (default: 4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--val_lambda", type=float, default=1.0, help="Lambda value for Monte Carlo returns")
    parser.add_argument("--out_dir", type=str, default="results/sampled_E_aux_random_sweep", help="Root output directory")
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


def build_env_config(env_name: str, args: argparse.Namespace, critic_lr: float, k: int) -> Dict[str, Any]:
    """Configures environment parameters for sampled random policy evaluation."""
    is_mc = "mountaincar" in env_name.lower()
    batch_size = args.num_envs * args.num_steps
    cfg = base_cfg.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "NUM_STEPS": args.num_steps,
        "NUM_ENVS": args.num_envs,
        "MINIBATCH_SIZE": min(args.minibatch_size, batch_size),
        "NUM_EPOCHS": args.num_epochs,
        "TOTAL_TIMESTEPS": args.num_updates * batch_size,
        "NUM_UPDATES": args.num_updates,
        "N_SEEDS": args.n_seeds,
        "SEED": args.seed,
        "GAMMA": args.gamma,
        "k": k,
        "LR": critic_lr,
        "LR_END": critic_lr,
        "CRITIC_LR": critic_lr,
        "CRITIC_LR_END": critic_lr,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "USE_VISUAL_OBS": False if is_mc else True,
        "LAYER_NORM": False,
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
        "USE_GREEDY_POLICY": False,
        "VALUE_LAMBDA": args.val_lambda,
        "RETURN_LAMBDA": 1.0,
        "RECOMPUTE_TARGETS_EACH_EPOCH": False,
    })
    return cfg


def compute_skew_symmetry_ratio(env_name: str, config: Dict[str, Any]) -> float:
    """Computes Frobenius norm ratio ||K||_F / ||S||_F for the environment under uniform random policy."""
    try:
        env, env_params = helpers.make_env(config)
        evaluator = helpers.initialize_evaluator(config, env, env_params)
        if evaluator is None:
            return np.nan

        n_states = evaluator.num_states
        n_actions = getattr(evaluator, "num_actions", 4)
        P = np.array(evaluator.P)
        gamma = float(config.get("GAMMA", 0.99))
        n_total = P.shape[0]

        # Uniform random policy
        Pi = np.ones((n_total, n_actions)) / n_actions
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
        print(f"[Warning] Could not compute skew symmetry ratio for {env_name}: {e}")
        return np.nan


def compute_normalized_auc(series: np.ndarray) -> float:
    """Computes Normalized Area Under the Curve: (1 / T) * integral_0^T f(t) dt."""
    series = np.asarray(series)
    T = len(series)
    if T <= 1:
        return float(series[0]) if T == 1 else 0.0
    return float(trapz_fn(series, dx=1.0) / (T - 1))


def run_experiment_for_lr(
    algo_key: str,
    algo_spec: Dict[str, Any],
    env_name: str,
    args: argparse.Namespace,
    critic_lr: float,
    k: int,
    seeds: List[int],
) -> Dict[str, Any]:
    """Runs a single (algo, k, lr) configuration across seeds."""
    env_cfg = build_env_config(env_name, args, critic_lr, k)
    
    # Merge algo-specific hyperparameters
    for hk, hv in algo_spec.get("hparams", {}).items():
        env_cfg[hk] = hv

    train_builder = algo_spec["train_builder"]
    train_fn = train_builder(env_cfg)

    seed_metrics = []
    t0 = time.time()

    for s_idx, s in enumerate(seeds):
        rng = jax.random.PRNGKey(s)
        out = train_fn(rng)
        metrics = out["metrics"]
        
        # Convert JAX device arrays to NumPy float32
        np_metrics = {}
        for m_name, m_val in metrics.items():
            arr = np.asarray(m_val)
            if arr.ndim > 1:
                arr = arr.squeeze()
            np_metrics[m_name] = arr.astype(np.float32)
        seed_metrics.append(np_metrics)

    elapsed = time.time() - t0
    metric_keys = seed_metrics[0].keys()
    stacked_metrics = {}
    auc_scores = {}
    final_scores = {}

    for k_m in metric_keys:
        stacked = np.stack([sm[k_m] for sm in seed_metrics], axis=0)  # [n_seeds, num_updates]
        stacked_metrics[k_m] = stacked
        
        # Calculate AUC per seed
        aucs = [compute_normalized_auc(stacked[i]) for i in range(len(seeds))]
        auc_scores[k_m] = {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
            "sem": float(np.std(aucs) / np.sqrt(len(seeds))),
            "values": aucs,
        }
        
        # Calculate final update value per seed
        finals = stacked[:, -1]
        final_scores[k_m] = {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals)),
            "sem": float(np.std(finals) / np.sqrt(len(seeds))),
            "values": finals.tolist(),
        }

    return {
        "algo": algo_key,
        "k": k,
        "critic_lr": critic_lr,
        "stacked_metrics": stacked_metrics,
        "auc_scores": auc_scores,
        "final_scores": final_scores,
        "elapsed_seconds": elapsed,
    }


def main():
    args = parse_args()

    if args.env_idx is not None:
        if 0 <= args.env_idx < len(ALL_ENVS):
            args.env_name = ALL_ENVS[args.env_idx]
        else:
            raise ValueError(f"env_idx {args.env_idx} out of range [0, {len(ALL_ENVS)-1}]")

    env_name = args.env_name
    print("=" * 80)
    print(f"SAMPLED E-AUX RANDOM POLICY EVALUATION SWEEP: {env_name}")
    print(f"Feature Dimensions k: {args.k_values}")
    print(f"Critic Learning Rates: {args.critic_lrs}")
    print(f"Seeds: {args.n_seeds} (base: {args.seed}) | Updates: {args.num_updates}")
    print(f"Rollout: {args.num_envs} envs x {args.num_steps} steps | Minibatch: {args.minibatch_size} | Epochs: {args.num_epochs}")
    print("=" * 80)

    # 1. Environment baseline & skew-symmetry
    v_star, v_rand, delta = load_env_baseline(env_name)
    dummy_cfg = build_env_config(env_name, args, args.critic_lrs[0], args.k_values[0])
    skew_ratio = compute_skew_symmetry_ratio(env_name, dummy_cfg)
    print(f"Baselines -> v_star: {v_star:.4f}, v_rand: {v_rand:.4f}, delta: {delta:.4f}")
    print(f"Skew-Symmetry Ratio (||K|| / ||S||): {skew_ratio:.4f}")

    seeds = [args.seed + i for i in range(args.n_seeds)]
    save_env_dir = os.path.join(args.out_dir, env_name)
    os.makedirs(save_env_dir, exist_ok=True)

    results_by_algo = {algo: {} for algo in ALGORITHMS.keys()}
    lr_tuning_records = []
    summary_rows = []

    # 2. Main execution loop
    total_start_time = time.time()

    for k in args.k_values:
        print(f"\n--- Feature Dimension k = {k} ---")
        for algo_key, algo_spec in ALGORITHMS.items():
            print(f"\nEvaluating Algorithm: {algo_spec['display_name']} (k={k})")
            
            lr_results = []
            for critic_lr in args.critic_lrs:
                print(f"  Sweeping critic_lr = {critic_lr} across {len(seeds)} seeds...", end="", flush=True)
                run_res = run_experiment_for_lr(
                    algo_key=algo_key,
                    algo_spec=algo_spec,
                    env_name=env_name,
                    args=args,
                    critic_lr=critic_lr,
                    k=k,
                    seeds=seeds,
                )
                mean_ve_auc = run_res["auc_scores"]["nn_weighted_VE"]["mean"]
                print(f" Done! Mean VE AUC: {mean_ve_auc:.6f} ({run_res['elapsed_seconds']:.1f}s)")
                lr_results.append(run_res)
                
                lr_tuning_records.append({
                    "env_name": env_name,
                    "k": k,
                    "algo": algo_key,
                    "display_name": algo_spec["display_name"],
                    "critic_lr": critic_lr,
                    "mean_ve_auc": mean_ve_auc,
                    "sem_ve_auc": run_res["auc_scores"]["nn_weighted_VE"]["sem"],
                })

            # Select best learning rate: minimum mean VE AUC
            best_run = min(lr_results, key=lambda r: r["auc_scores"]["nn_weighted_VE"]["mean"])
            best_lr = best_run["critic_lr"]
            print(f"  ==> Selected Best LR for {algo_key} (k={k}): {best_lr} (VE AUC: {best_run['auc_scores']['nn_weighted_VE']['mean']:.6f})")

            results_by_algo[algo_key][k] = {
                "best_lr": best_lr,
                "metrics": best_run["stacked_metrics"],
                "auc_scores": best_run["auc_scores"],
                "final_scores": best_run["final_scores"],
                "lr_sweep": {
                    r["critic_lr"]: {
                        "ve_auc_mean": r["auc_scores"]["nn_weighted_VE"]["mean"],
                        "ve_auc_sem": r["auc_scores"]["nn_weighted_VE"]["sem"],
                    }
                    for r in lr_results
                },
            }

            # Collect summary row for table
            ve_auc = best_run["auc_scores"].get("nn_weighted_VE", {})
            be_auc = best_run["auc_scores"].get("nn_weighted_BE", {})
            cossim_auc = best_run["auc_scores"].get("nn_advantage_cossim", {})
            gperf_auc = best_run["auc_scores"].get("nn_greedy_performance", {})
            vstart_auc = best_run["auc_scores"].get("v_pred_start", {})
            aux_ve_auc = best_run["auc_scores"].get("aux_value_error", {})

            ve_final = best_run["final_scores"].get("nn_weighted_VE", {})
            be_final = best_run["final_scores"].get("nn_weighted_BE", {})
            cossim_final = best_run["final_scores"].get("nn_advantage_cossim", {})
            gperf_final = best_run["final_scores"].get("nn_greedy_performance", {})
            vstart_final = best_run["final_scores"].get("v_pred_start", {})
            aux_ve_final = best_run["final_scores"].get("aux_value_error", {})

            # Standardized greedy score (0=random, 1=optimal)
            gperf_mean = gperf_final.get("mean", 0.0)
            norm_gperf = (gperf_mean - v_rand) / delta if delta > 1e-12 else gperf_mean

            summary_rows.append({
                "env_name": env_name,
                "k": k,
                "algo": algo_key,
                "display_name": algo_spec["display_name"],
                "best_critic_lr": best_lr,
                "skew_symmetry_ratio": skew_ratio,
                "auc_weighted_ve_mean": ve_auc.get("mean", np.nan),
                "auc_weighted_ve_sem": ve_auc.get("sem", np.nan),
                "auc_weighted_be_mean": be_auc.get("mean", np.nan),
                "auc_weighted_be_sem": be_auc.get("sem", np.nan),
                "auc_advantage_cossim_mean": cossim_auc.get("mean", np.nan),
                "auc_advantage_cossim_sem": cossim_auc.get("sem", np.nan),
                "auc_greedy_perf_mean": gperf_auc.get("mean", np.nan),
                "auc_greedy_perf_sem": gperf_auc.get("sem", np.nan),
                "auc_vstart_mean": vstart_auc.get("mean", np.nan),
                "auc_vstart_sem": vstart_auc.get("sem", np.nan),
                "auc_aux_ve_mean": aux_ve_auc.get("mean", np.nan),
                "auc_aux_ve_sem": aux_ve_auc.get("sem", np.nan),
                "final_weighted_ve_mean": ve_final.get("mean", np.nan),
                "final_weighted_ve_sem": ve_final.get("sem", np.nan),
                "final_weighted_be_mean": be_final.get("mean", np.nan),
                "final_weighted_be_sem": be_final.get("sem", np.nan),
                "final_advantage_cossim_mean": cossim_final.get("mean", np.nan),
                "final_advantage_cossim_sem": cossim_final.get("sem", np.nan),
                "final_greedy_perf_mean": gperf_final.get("mean", np.nan),
                "final_greedy_perf_sem": gperf_final.get("sem", np.nan),
                "final_normalized_greedy_perf": norm_gperf,
                "final_vstart_mean": vstart_final.get("mean", np.nan),
                "final_vstart_sem": vstart_final.get("sem", np.nan),
                "final_aux_ve_mean": aux_ve_final.get("mean", np.nan),
                "final_aux_ve_sem": aux_ve_final.get("sem", np.nan),
            })

    total_time = time.time() - total_start_time
    print(f"\nAll experiments for {env_name} completed in {total_time:.1f}s.")

    # 3. Save results package
    out_pkl_path = os.path.join(save_env_dir, "out.pkl")
    save_payload = {
        "env_name": env_name,
        "k_values": args.k_values,
        "skew_symmetry_ratio": skew_ratio,
        "v_star": v_star,
        "v_rand": v_rand,
        "delta": delta,
        "args": vars(args),
        "results_by_algo": results_by_algo,
    }
    with open(out_pkl_path, "wb") as f:
        pickle.dump(save_payload, f)
    print(f"Saved full experiment payload to: {out_pkl_path}")

    # 4. Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_auc_csv_path = os.path.join(save_env_dir, "summary_auc.csv")
    df_summary.to_csv(summary_auc_csv_path, index=False)
    summary_csv_path = os.path.join(save_env_dir, "summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved summary AUC table to: {summary_auc_csv_path} and {summary_csv_path}")

    # 5. Save LR tuning summary JSON and CSV
    lr_json_path = os.path.join(save_env_dir, "lr_selection_summary.json")
    with open(lr_json_path, "w") as f:
        json.dump(lr_tuning_records, f, indent=2)
    print(f"Saved learning rate tuning summary to: {lr_json_path}")

    df_all_lrs = pd.DataFrame(lr_tuning_records)
    all_lrs_csv_path = os.path.join(save_env_dir, "summary_all_lrs.csv")
    df_all_lrs.to_csv(all_lrs_csv_path, index=False)
    print(f"Saved all LRs summary to: {all_lrs_csv_path}")

    # 6. Save config.json
    cfg_path = os.path.join(save_env_dir, "config.json")
    base_env_cfg = build_env_config(
        env_name,
        args,
        critic_lr=args.critic_lrs[0],
        k=args.k_values[0] if args.k_values else 8,
    )
    base_env_cfg["SWEEP_K_VALUES"] = list(args.k_values)
    base_env_cfg["SWEEP_CRITIC_LRS"] = list(args.critic_lrs)
    serializable_cfg = {
        k: v for k, v in base_env_cfg.items()
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None
    }
    with open(cfg_path, "w") as f:
        json.dump(serializable_cfg, f, indent=4)
    print(f"Saved experiment configuration to: {cfg_path}")


if __name__ == "__main__":
    main()
