"""
scripts/sampled_fixed_policy_sweep/run_single_env.py

Executes sampled (rollout-based) fixed policy evaluation across 5 algorithms:
  1. MC:          fixed_policy.td (GAE_LAMBDA=1.0)
  2. TD(0):       fixed_policy.td (GAE_LAMBDA=0.0)
  3. E(0) [sep14]: fixed_policy.sampled_E (RETURN_LAMBDA=1.0, E_VARIANT="sep14") [Uncorrected PSD]
  4. E(0) [oct5]:  fixed_policy.sampled_E (RETURN_LAMBDA=1.0, E_VARIANT="oct5") [Empirical Corrected]
  5. E(0) [oct8]:  fixed_policy.sampled_E (RETURN_LAMBDA=1.0, E_VARIANT="oct8") [Start-Anchored]

Evaluates the exact same fixed policies generated for exact_fixed_policy_sweep
(cached in results/ppo/ground_truth/fixed_policies/<env_name>/out.pkl).

Sweeps critic learning rates across candidate values, selecting the condition
minimizing normalized AUC of Value Error (nn_weighted_VE).

Outputs:
  - out.pkl
  - summary_auc.csv
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
from fixed_policy.td import make_train as make_td_train
from fixed_policy.sampled_E import make_train as make_e0_train
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
        "train_builder": make_td_train,
        "hparams": {"GAE_LAMBDA": 1.0},
        "color": "#1f77b4",
    },
    "td0": {
        "display_name": "TD(0) [Sampled]",
        "train_builder": make_td_train,
        "hparams": {"GAE_LAMBDA": 0.0},
        "color": "#ff7f0e",
    },
    "e0_sep14": {
        "display_name": "E(0) [Sept 14 / Uncorrected]",
        "train_builder": make_e0_train,
        "hparams": {"RETURN_LAMBDA": 1.0, "E_VARIANT": "sep14"},
        "color": "#2ca02c",
    },
    "e0_oct5": {
        "display_name": "E(0) [Oct 5 / Corrected]",
        "train_builder": make_e0_train,
        "hparams": {"RETURN_LAMBDA": 1.0, "E_VARIANT": "oct5"},
        "color": "#17becf",
    },
    "e0_oct8": {
        "display_name": "E(0) [Oct 8 / Start-Anchored]",
        "train_builder": make_e0_train,
        "hparams": {"RETURN_LAMBDA": 1.0, "E_VARIANT": "oct8"},
        "color": "#9467bd",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sampled Fixed Policy Evaluation Sweep on a Single Env")
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
    parser.add_argument("--val_lambda", type=float, default=0.8, help="Lambda value for TD(lambda) and E(lambda)")
    parser.add_argument("--policy_dir", type=str, default="results/ppo/ground_truth/fixed_policies", help="Fixed policy model directory")
    parser.add_argument("--policy_updates", type=int, default=50, help="Updates used if generating fixed policy on the fly")
    parser.add_argument("--out_dir", type=str, default="results/sampled_fixed_policy_sweep", help="Root output directory")
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
    """Configures environment parameters for sampled fixed policy evaluation."""
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
        "MODEL_LOAD_DIR": "ground_truth/fixed_policies",
        "USE_GREEDY_POLICY": False,
        "VALUE_LAMBDA": args.val_lambda,
        "RECOMPUTE_TARGETS_EACH_EPOCH": False,
    })
    return cfg


def compute_skew_symmetry_ratio(env_name: str, config: Dict[str, Any]) -> float:
    """Computes Frobenius norm ratio ||K||_F / ||S||_F for the environment under its evaluated fixed policy."""
    try:
        env, env_params = helpers.make_env(config)
        evaluator = helpers.initialize_evaluator(config, env, env_params)
        _, policy_matrix = helpers.get_evaluation_policies(config, evaluator)

        Pi = np.array(policy_matrix)
        P = np.array(evaluator.P)
        gamma = float(config.get("GAMMA", 0.99))
        n_states = evaluator.num_states
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
        print(f"[Warning] Could not compute skew symmetry ratio: {e}")
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
        if hk == "VALUE_LAMBDA":
            env_cfg[hk] = args.val_lambda
        elif hk == "GAE_LAMBDA" and algo_key == "td_lambda":
            env_cfg[hk] = args.val_lambda
        else:
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
    print(f"SAMPLED FIXED POLICY SWEEP: {env_name}")
    print(f"Feature Dimensions k: {args.k_values}")
    print(f"Critic Learning Rates: {args.critic_lrs}")
    print(f"Seeds: {args.n_seeds} (base: {args.seed}) | Updates: {args.num_updates}")
    print(f"Rollout: {args.num_envs} envs x {args.num_steps} steps | Minibatch: {args.minibatch_size} | Epochs: {args.num_epochs}")
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

    # 2. Environment info & baseline
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

    # 3. Main execution loop
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

            ve_final = best_run["final_scores"].get("nn_weighted_VE", {})
            be_final = best_run["final_scores"].get("nn_weighted_BE", {})
            cossim_final = best_run["final_scores"].get("nn_advantage_cossim", {})
            gperf_final = best_run["final_scores"].get("nn_greedy_performance", {})
            vstart_final = best_run["final_scores"].get("v_pred_start", {})

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
            })

    total_time = time.time() - total_start_time
    print(f"\nAll experiments for {env_name} completed in {total_time:.1f}s.")

    # 4. Save results package
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

    # 5. Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(save_env_dir, "summary_auc.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved summary AUC table to: {summary_csv_path}")

    # 6. Save LR tuning summary JSON
    lr_json_path = os.path.join(save_env_dir, "lr_selection_summary.json")
    with open(lr_json_path, "w") as f:
        json.dump(lr_tuning_records, f, indent=2)
    print(f"Saved learning rate tuning summary to: {lr_json_path}")


if __name__ == "__main__":
    main()
