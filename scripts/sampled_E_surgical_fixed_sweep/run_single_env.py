"""
scripts/sampled_E_surgical_fixed_sweep/run_single_env.py

Executes sampled fixed policy evaluation across 4 algorithms:
  1. mc:               Monte Carlo trajectory baseline
  2. surgical_2heads:  Surgical v_aux (2 Heads on Shared Torso)
  3. surgical_2nets:   Surgical v_aux (2 Independent Networks)
  4. oct8_biased:      OG Oct 8 (Biased Start-Anchored)

Evaluates the exact same fixed policies generated for exact_fixed_policy_sweep
(cached in results/ppo/ground_truth/fixed_policies/<env_name>/out.pkl).

Tuning criterion (Hierarchical Selection per User Specification):
  1. V_start AUC (v_pred_start, higher is better)
  2. If statistical tie (within 1.0 pooled SEM), Advantage Cosine Similarity AUC (nn_advantage_cossim, higher is better)
  3. If still tied, Value Error AUC (nn_weighted_VE, lower is better)

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
from fixed_policy.mc import make_train as make_mc_train
from fixed_policy.td import make_train as make_td_train
from fixed_policy.sampled_E import make_train as make_sampled_e_train
from scripts.exact_fixed_policy_sweep.generate_fixed_policies import generate_policy_for_env, is_checkpoint_valid

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
    "td_lambda": {
        "display_name": "TD(λ=0.9)",
        "train_builder": make_td_train,
        "hparams": {"GAE_LAMBDA": 0.9, "VALUE_LAMBDA": 0.9},
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
        "hparams": {"LOSS_VARIANT": "surgical_v_aux", "SHARED_TORSO": False, "RETURN_LAMBDA": 1.0, "AUX_LAMBDA": 0.9, "AUX_VF_COEF": 1.0},
        "color": "#2ca02c",  # Forest Green
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sampled E-Surgical Fixed Policy Evaluation Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--policy_dir", type=str, default="ground_truth/fixed_policies", help="Directory where fixed policies are cached")
    parser.add_argument("--policy_updates", type=int, default=100, help="Updates used if generating fixed policy")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of rollout updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k_values", type=int, nargs="+", default=[32], help="Feature dimensions to evaluate (default: 32)")
    parser.add_argument("--critic_lrs", type=float, nargs="+", default=[0.0005, 0.001, 0.003], help="Critic learning rates to sweep")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments (default: 16)")
    parser.add_argument("--num_steps", type=int, default=64, help="Steps per rollout (default: 64)")
    parser.add_argument("--minibatch_size", type=int, default=256, help="Minibatch size (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Epochs per update (default: 4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--val_lambda", type=float, default=1.0, help="Lambda value for Monte Carlo returns")
    parser.add_argument("--out_dir", type=str, default="results/sampled_E_surgical_fixed_sweep", help="Root output directory")
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
    """Checks if two mean estimates are within threshold_sems pooled standard errors."""
    if not np.isfinite(mean_a) or not np.isfinite(mean_b):
        return False
    pooled_sem = np.sqrt(max(0.0, sem_a)**2 + max(0.0, sem_b)**2)
    if pooled_sem <= 1e-12:
        return np.isclose(mean_a, mean_b, atol=1e-6)
    return abs(mean_a - mean_b) <= threshold_sems * pooled_sem


def select_best_run_hierarchical(lr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selects best hyperparameter condition using hierarchical tie-breaking:
      1. V_start AUC (v_pred_start, higher is better)
      2. If statistical tie, Advantage Cosine Similarity AUC (nn_advantage_cossim, higher is better)
      3. If still tied, Value Error AUC (nn_weighted_VE, lower is better)
    """
    if len(lr_results) == 1:
        return lr_results[0]

    # Check if v_pred_start exists in all runs
    has_vstart = all("v_pred_start" in r["auc_scores"] for r in lr_results)
    if not has_vstart:
        return min(lr_results, key=lambda r: r["auc_scores"]["nn_weighted_VE"]["mean"])

    # 1. Sort by V_start AUC descending (higher is better)
    sorted_by_vstart = sorted(lr_results, key=lambda r: r["auc_scores"]["v_pred_start"]["mean"], reverse=True)
    top_vstart = sorted_by_vstart[0]
    top_vstart_mean = top_vstart["auc_scores"]["v_pred_start"]["mean"]
    top_vstart_sem = top_vstart["auc_scores"]["v_pred_start"]["sem"]

    # Filter candidates statistically tied with the top V_start
    tied_vstart = [
        r for r in sorted_by_vstart
        if is_statistically_tied(
            top_vstart_mean, top_vstart_sem,
            r["auc_scores"]["v_pred_start"]["mean"], r["auc_scores"]["v_pred_start"]["sem"]
        )
    ]

    if len(tied_vstart) == 1:
        return tied_vstart[0]

    # 2. Tie-break on Advantage Cosine Similarity AUC (higher is better)
    has_asim = all("nn_advantage_cossim" in r["auc_scores"] for r in tied_vstart)
    if not has_asim:
        return min(tied_vstart, key=lambda r: r["auc_scores"]["nn_weighted_VE"]["mean"])

    sorted_by_asim = sorted(tied_vstart, key=lambda r: r["auc_scores"]["nn_advantage_cossim"]["mean"], reverse=True)
    top_asim = sorted_by_asim[0]
    top_asim_mean = top_asim["auc_scores"]["nn_advantage_cossim"]["mean"]
    top_asim_sem = top_asim["auc_scores"]["nn_advantage_cossim"]["sem"]

    tied_asim = [
        r for r in sorted_by_asim
        if is_statistically_tied(
            top_asim_mean, top_asim_sem,
            r["auc_scores"]["nn_advantage_cossim"]["mean"], r["auc_scores"]["nn_advantage_cossim"]["sem"]
        )
    ]

    if len(tied_asim) == 1:
        return tied_asim[0]

    # 3. Final tie-break on Value Error AUC (lower is better)
    best_final = min(tied_asim, key=lambda r: r["auc_scores"]["nn_weighted_VE"]["mean"])
    return best_final


def build_env_config(env_name: str, args, critic_lr: float, k: int) -> Dict[str, Any]:
    cfg = base_cfg.copy()
    cfg["ENV_NAME"] = env_name
    cfg["NUM_UPDATES"] = args.num_updates
    cfg["NUM_ENVS"] = args.num_envs
    cfg["NUM_STEPS"] = args.num_steps
    cfg["TOTAL_TIMESTEPS"] = args.num_updates * args.num_envs * args.num_steps
    cfg["MINIBATCH_SIZE"] = args.minibatch_size
    cfg["NUM_MINIBATCHES"] = max(1, (args.num_envs * args.num_steps) // args.minibatch_size)
    cfg["NUM_EPOCHS"] = args.num_epochs
    cfg["CRITIC_LR"] = critic_lr
    cfg["LR"] = critic_lr
    cfg["k"] = k
    cfg["GAMMA"] = args.gamma
    cfg["RETURN_LAMBDA"] = args.val_lambda
    cfg["VALUE_LAMBDA"] = args.val_lambda
    cfg["LIGHT_METRICS"] = True
    cfg["LOG_FEATURE_METRICS"] = False
    return cfg


def load_env_baseline(env_name: str) -> tuple[float, float, float]:
    """Loads ground truth optimal value, random policy value, and delta."""
    baseline_path = os.path.join(REPO_ROOT, "ground_truth", "optimal_values", f"{env_name}.json")
    v_star, v_rand = 1.0, 0.0
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r") as f:
                data = json.load(f)
                v_star = data.get("v_star", 1.0)
                v_rand = data.get("v_rand", 0.0)
        except Exception:
            pass
    delta = max(v_star - v_rand, 1e-6)
    return v_star, v_rand, delta


def run_experiment_for_lr(
    algo_key: str,
    algo_spec: Dict[str, Any],
    env_name: str,
    args,
    critic_lr: float,
    k: int,
    seeds: List[int],
) -> Dict[str, Any]:
    t0 = time.time()
    cfg = build_env_config(env_name, args, critic_lr, k)
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
        seed_metrics.append(np_metrics)

    elapsed = time.time() - t0
    metric_keys = seed_metrics[0].keys()
    stacked_metrics = {}
    auc_scores = {}
    final_scores = {}

    for k_m in metric_keys:
        stacked = np.stack([sm[k_m] for sm in seed_metrics], axis=0)  # [n_seeds, num_updates]
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
    print(f"SAMPLED E-SURGICAL FIXED POLICY SWEEP: {env_name}")
    print(f"Algorithms: {list(ALGORITHMS.keys())}")
    print(f"Feature Dimensions k: {args.k_values}")
    print(f"Critic Learning Rates: {args.critic_lrs}")
    print(f"Tuning Criterion: V_start AUC > Advantage CosSim AUC > VE AUC")
    print(f"Seeds: {args.n_seeds} (base: {args.seed}) | Updates: {args.num_updates}")
    print("=" * 80)

    # 1. Step 0 Check: Ensure fixed policy checkpoint exists
    policy_env_dir = os.path.join(os.path.abspath(args.policy_dir), env_name)
    print(f"[Step 0] Verifying fixed policy checkpoint in '{policy_env_dir}'...")
    if not is_checkpoint_valid(policy_env_dir):
        print(f"Checkpoint for '{env_name}' missing or incomplete. Generating now...")
        gen_args = argparse.Namespace(
            policy_dir=args.policy_dir,
            num_updates=args.policy_updates,
            actor_lr=0.001,
            seed=args.seed,
            force=False,
        )
        generate_policy_for_env(env_name, gen_args)
    else:
        print(f"Valid fixed policy checkpoint confirmed for '{env_name}'.")

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

    for k in args.k_values:
        print(f"\n>>> Running Capacity k = {k} <<<")
        for algo_key, algo_spec in ALGORITHMS.items():
            print(f"\n--- Algorithm: {algo_spec['display_name']} (k={k}) ---")
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
                vstart_auc_val = run_res["auc_scores"].get("v_pred_start", {}).get("mean", np.nan)
                cossim_auc_val = run_res["auc_scores"].get("nn_advantage_cossim", {}).get("mean", np.nan)
                mean_ve_auc = run_res["auc_scores"]["nn_weighted_VE"]["mean"]
                print(f" Done! V_start: {vstart_auc_val:.4f} | CosSim: {cossim_auc_val:.4f} | VE AUC: {mean_ve_auc:.6f} ({run_res['elapsed_seconds']:.1f}s)")
                lr_results.append(run_res)
                
                lr_tuning_records.append({
                    "env_name": env_name,
                    "k": k,
                    "algo": algo_key,
                    "display_name": algo_spec["display_name"],
                    "critic_lr": critic_lr,
                    "mean_vstart_auc": vstart_auc_val,
                    "mean_cossim_auc": cossim_auc_val,
                    "mean_ve_auc": mean_ve_auc,
                    "sem_ve_auc": run_res["auc_scores"]["nn_weighted_VE"]["sem"],
                })

            # Hierarchical selection
            best_run = select_best_run_hierarchical(lr_results)
            best_lr = best_run["critic_lr"]
            sel_vstart = best_run["auc_scores"].get("v_pred_start", {}).get("mean", np.nan)
            sel_cossim = best_run["auc_scores"].get("nn_advantage_cossim", {}).get("mean", np.nan)
            sel_ve = best_run["auc_scores"]["nn_weighted_VE"]["mean"]
            print(f"  ==> Selected Best LR for {algo_key} (k={k}): {best_lr} (V_start: {sel_vstart:.4f}, CosSim: {sel_cossim:.4f}, VE: {sel_ve:.6f})")

            results_by_algo[algo_key][k] = {
                "best_lr": best_lr,
                "metrics": best_run["stacked_metrics"],
                "auc_scores": best_run["auc_scores"],
                "final_scores": best_run["final_scores"],
                "lr_sweep": {
                    r["critic_lr"]: {
                        "vstart_auc_mean": r["auc_scores"].get("v_pred_start", {}).get("mean", np.nan),
                        "cossim_auc_mean": r["auc_scores"].get("nn_advantage_cossim", {}).get("mean", np.nan),
                        "ve_auc_mean": r["auc_scores"]["nn_weighted_VE"]["mean"],
                        "ve_auc_sem": r["auc_scores"]["nn_weighted_VE"]["sem"],
                    }
                    for r in lr_results
                },
            }

            ve_auc = best_run["auc_scores"].get("nn_weighted_VE", {})
            be_auc = best_run["auc_scores"].get("nn_weighted_BE", {})
            cossim_auc = best_run["auc_scores"].get("nn_advantage_cossim", {})
            vstart_auc = best_run["auc_scores"].get("v_pred_start", {})

            ve_final = best_run["final_scores"].get("nn_weighted_VE", {})
            be_final = best_run["final_scores"].get("nn_weighted_BE", {})
            cossim_final = best_run["final_scores"].get("nn_advantage_cossim", {})
            vstart_final = best_run["final_scores"].get("v_pred_start", {})

            summary_rows.append({
                "env_name": env_name,
                "k": k,
                "algo": algo_key,
                "display_name": algo_spec["display_name"],
                "best_critic_lr": best_lr,
                "auc_vstart_mean": vstart_auc.get("mean", np.nan),
                "auc_vstart_sem": vstart_auc.get("sem", np.nan),
                "final_vstart_mean": vstart_final.get("mean", np.nan),
                "final_vstart_sem": vstart_final.get("sem", np.nan),
                "auc_cossim_mean": cossim_auc.get("mean", np.nan),
                "auc_cossim_sem": cossim_auc.get("sem", np.nan),
                "final_cossim_mean": cossim_final.get("mean", np.nan),
                "final_cossim_sem": cossim_final.get("sem", np.nan),
                "auc_weighted_ve_mean": ve_auc.get("mean", np.nan),
                "auc_weighted_ve_sem": ve_auc.get("sem", np.nan),
                "final_weighted_ve_mean": ve_final.get("mean", np.nan),
                "final_weighted_ve_sem": ve_final.get("sem", np.nan),
                "auc_weighted_be_mean": be_auc.get("mean", np.nan),
                "auc_weighted_be_sem": be_auc.get("sem", np.nan),
                "final_weighted_be_mean": be_final.get("mean", np.nan),
                "final_weighted_be_sem": be_final.get("sem", np.nan),
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
        "k_values": args.k_values,
        "critic_lrs": args.critic_lrs,
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "num_updates": args.num_updates,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "minibatch_size": args.minibatch_size,
        "num_epochs": args.num_epochs,
        "gamma": args.gamma,
        "val_lambda": args.val_lambda,
        "tuning_criterion": "v_pred_start > nn_advantage_cossim > nn_weighted_VE",
        "total_elapsed_seconds": total_time,
    }
    with open(os.path.join(env_out_dir, "config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    print(f"\nAll tasks for '{env_name}' completed in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
