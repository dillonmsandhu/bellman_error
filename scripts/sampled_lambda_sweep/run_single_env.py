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
        "train_builder": make_td_lambda_train,
        "base_hparams": {},
        "color_base": "#ff7f0e",  # Orange base
    },
    "e_lambda": {
        "display_name": "E(λ) [v_aux]",
        "train_builder": make_e_lambda_train,
        "base_hparams": {"E_LAMBDA_VARIANT": "aux", "RETURN_LAMBDA": 1.0},
        "color_base": "#2ca02c",  # Green base
    },
}

LAMBDA_COLORS = {
    "td_0.50": "#fdae6b",
    "td_0.90": "#f16913",
    "td_0.95": "#8c2d04",
    "e_0.50": "#a1d99b",
    "e_0.90": "#41ab5d",
    "e_0.95": "#005a32",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sampled PPO Lambda Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 0.9, 0.95], help="Lambda values to evaluate (default: 0.5 0.9 0.95)")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of rollout updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k", type=int, default=32, help="Critic feature dimension (default: 32)")
    parser.add_argument("--actor_lr", type=float, default=0.0003, help="Actor learning rate (default: 0.0003)")
    parser.add_argument("--critic_lrs", type=float, nargs="+", default=[0.0005, 0.001, 0.003], help="Critic learning rates to sweep (default: 0.0005 0.001 0.003)")
    parser.add_argument("--num_envs", type=int, default=32, help="Parallel environments (default: 32)")
    parser.add_argument("--num_steps", type=int, default=128, help="Steps per rollout (default: 128)")
    parser.add_argument("--minibatch_size", type=int, default=256, help="Minibatch size (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Epochs per update (default: 4)")
    parser.add_argument("--gae_lambda", type=float, default=0.8, help="Actor GAE lambda (default: 0.8)")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon (default: 0.2)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--runway_length", type=int, default=16, help="Runway window length for geometric jumps (default: 16)")
    parser.add_argument("--select_by", type=str, default="std_perf_auc", choices=["std_perf_auc", "std_perf_final", "ve_auc"], help="Metric to select best critic condition")
    parser.add_argument("--out_dir", type=str, default="results/sampled_lambda_sweep", help="Root output directory")
    parser.add_argument("--algos", type=str, nargs="+", default=["td_lambda", "e_lambda"], choices=["td_lambda", "e_lambda"], help="Algorithms to evaluate")
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
        "RUNWAY_LENGTH": args.runway_length,
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
        for lam in args.lambdas:
            lam_str = f"{lam:.2f}"
            cond_key = f"{'td' if algo_key == 'td_lambda' else 'e'}_{lam_str}"
            display_name = f"{'TD' if algo_key == 'td_lambda' else 'E'}({lam_str})"

            print(f"\n========================================================")
            print(f"Evaluating Condition: {cond_key} ({display_name})")
            print(f"========================================================")

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
