"""
scripts/sampled_ppo_sweep/run_single_env.py

Executes full sampled PPO (both actor and critic trained simultaneously via rollouts)
across 5 algorithms on a single benchmark environment:
  1. mc:       Sampled Monte Carlo critic (TD with VALUE_LAMBDA=1.0)
  2. td0:      Sampled TD(0) critic (dynamic targets with stop_gradient)
  3. e0_sep14: Sampled E(0) [Sept 14 / Uncorrected] critic (RETURN_LAMBDA=1.0, E_VARIANT="sep14")
  4. e0_oct5:  Sampled E(0) [Oct 5 / Corrected] critic (RETURN_LAMBDA=1.0, E_VARIANT="oct5")
  5. e0_oct8:  Sampled E(0) [Oct 8 / Start-Anchored] critic (RETURN_LAMBDA=1.0, E_VARIANT="oct8")

Sweeps candidate critic learning rates, selecting the condition by standardized
policy return AUC: V_norm = (V_start - V_rand) / (V_star - V_rand).

Outputs:
  - out.pkl
  - summary.csv (winning run per algorithm)
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
from ppo.sampled_mc import make_train as make_mc_train
from ppo.sampled_td import make_train as make_td0_train
from ppo.sampled_E import make_train as make_e0_train

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
        "color": "#1f77b4",
    },
    "td0": {
        "display_name": "TD(0) [Sampled]",
        "train_builder": make_td0_train,
        "hparams": {},
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
    parser = argparse.ArgumentParser(description="Full Sampled PPO Evaluation Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of rollout updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k", type=int, default=32, help="Critic feature dimension (default: 32)")
    parser.add_argument("--actor_lr", type=float, default=0.0003, help="Actor learning rate (default: 0.0003)")
    parser.add_argument("--critic_lrs", type=float, nargs="+", default=[0.0005, 0.001, 0.003], help="Critic learning rates to sweep (default: 0.0005 0.001 0.003)")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments (default: 16)")
    parser.add_argument("--num_steps", type=int, default=64, help="Steps per rollout (default: 64)")
    parser.add_argument("--minibatch_size", type=int, default=256, help="Minibatch size (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Epochs per update (default: 4)")
    parser.add_argument("--gae_lambda", type=float, default=0.8, help="Actor GAE lambda (default: 0.8)")
    parser.add_argument("--return_lambdas", type=float, nargs="+", default=[0.95, 1.0], help="Return lambda values to sweep for critic targets (default: 0.95 1.0)")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon (default: 0.2)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--select_by", type=str, default="std_perf_auc", choices=["std_perf_auc", "std_perf_final", "ve_auc"], help="Metric to select best critic condition")
    parser.add_argument("--out_dir", type=str, default="results/sampled_ppo_sweep", help="Root output directory")
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
    print("Full Sampled PPO Evaluation Sweep on Single Environment")
    print(f"Environment:       {args.env_name}")
    print(f"Num Updates:       {args.num_updates}")
    print(f"Rollout Config:    {args.num_envs} envs x {args.num_steps} steps (batch={args.num_envs * args.num_steps})")
    print(f"Num Seeds:         {args.n_seeds} (base seed: {args.seed})")
    print(f"Critic Feature k:  {args.k}")
    print(f"Actor LR:          {args.actor_lr}")
    print(f"Critic LRs:        {args.critic_lrs}")
    print(f"Actor GAE Lambda:  {args.gae_lambda}")
    print(f"PPO Clip Epsilon:  {args.clip_eps}")
    print(f"Selection Metric:  {args.select_by}")
    print(f"Algorithms:        {list(ALGORITHMS.keys())}")
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
        "runs": {},               # winning run per algo (E variants: tuned lambda)
        "runs_lam1": {},          # winning run with lambda=1.0 for cross-algo comparison
        "all_runs": {},           # all runs: algo -> cond_key -> res
        "best_critic_lrs": {},
        "best_return_lambdas": {},
        "best_conditions": {},
        "best_conditions_lam1": {},
        "selection_metric": args.select_by,
    }

    summary_rows = []
    all_lrs_rows = []

    for algo_key, algo_info in ALGORITHMS.items():
        print(f"\n========================================================")
        print(f"Algorithm: {algo_key} ({algo_info['display_name']})")
        print(f"========================================================")

        full_results["all_runs"][algo_key] = {}
        lr_candidates = []

        # Return lambdas:
        # - TD(0) bootstraps from v(s') so return_lambda is 0.0
        # - MC strictly uses lambda=1.0 (pure Monte Carlo, no return lambda sweep)
        # - E variants sweep return_lambdas (e.g. [0.95, 1.0])
        if algo_key == "td0":
            cand_lambdas = [0.0]
        elif algo_key == "mc":
            cand_lambdas = [1.0]
        else:
            cand_lambdas = args.return_lambdas

        for c_lr in args.critic_lrs:
            for r_lam in cand_lambdas:
                lam_str = f" | RetLambda = {r_lam}" if algo_key.startswith("e0") else ""
                print(f"  --> Testing Critic LR = {c_lr}{lam_str} ...", end="", flush=True)
                env_cfg = build_env_config(args.env_name, args, critic_lr=c_lr)

                hparams = dict(algo_info["hparams"])
                if algo_key != "td0":
                    hparams["RETURN_LAMBDA"] = r_lam
                    if algo_key == "mc":
                        hparams["VALUE_LAMBDA"] = r_lam

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

                if v_start_seeds is not None:
                    std_perf_seeds = (v_start_seeds - v_rand) / (delta + 1e-12)
                    res["standardized_v_start"] = std_perf_seeds

                    final_perf_per_seed = std_perf_seeds[:, -1]
                    auc_perf_per_seed = np.array([compute_normalized_auc(s) for s in std_perf_seeds])

                    mean_final_perf = float(np.mean(final_perf_per_seed))
                    sem_final_perf = float(np.std(final_perf_per_seed, ddof=1) / np.sqrt(args.n_seeds)) if args.n_seeds > 1 else 0.0

                    mean_auc_perf = float(np.mean(auc_perf_per_seed))
                    sem_auc_perf = float(np.std(auc_perf_per_seed, ddof=1) / np.sqrt(args.n_seeds)) if args.n_seeds > 1 else 0.0

                    raw_final_v_start = float(np.mean(v_start_seeds[:, -1]))
                else:
                    mean_final_perf = sem_final_perf = mean_auc_perf = sem_auc_perf = raw_final_v_start = np.nan

                if ve_seeds is not None:
                    final_ve_per_seed = ve_seeds[:, -1]
                    auc_ve_per_seed = np.array([compute_normalized_auc(s) for s in ve_seeds])
                    mean_final_ve = float(np.mean(final_ve_per_seed))
                    sem_final_ve = float(np.std(final_ve_per_seed, ddof=1) / np.sqrt(args.n_seeds)) if args.n_seeds > 1 else 0.0
                    mean_auc_ve = float(np.mean(auc_ve_per_seed))
                    sem_auc_ve = float(np.std(auc_ve_per_seed, ddof=1) / np.sqrt(args.n_seeds)) if args.n_seeds > 1 else 0.0
                else:
                    mean_final_ve = sem_final_ve = mean_auc_ve = sem_auc_ve = np.nan

                if cossim_seeds is not None:
                    final_cossim_per_seed = cossim_seeds[:, -1]
                    auc_cossim_per_seed = np.array([compute_normalized_auc(s) for s in cossim_seeds])
                    mean_final_cossim = float(np.mean(final_cossim_per_seed))
                    sem_final_cossim = float(np.std(final_cossim_per_seed, ddof=1) / np.sqrt(args.n_seeds)) if args.n_seeds > 1 else 0.0
                    mean_auc_cossim = float(np.mean(auc_cossim_per_seed))
                    sem_auc_cossim = float(np.std(auc_cossim_per_seed, ddof=1) / np.sqrt(args.n_seeds)) if args.n_seeds > 1 else 0.0
                else:
                    mean_final_cossim = sem_final_cossim = mean_auc_cossim = sem_auc_cossim = np.nan

                cond_key = f"lr_{c_lr}_lam_{r_lam}" if algo_key != "td0" else f"lr_{c_lr}"
                full_results["all_runs"][algo_key][cond_key] = res

                row_data = {
                    "env": args.env_name,
                    "algo": algo_key,
                    "display_name": algo_info["display_name"],
                    "k": args.k,
                    "critic_lr": c_lr,
                    "return_lambda": r_lam if algo_key != "td0" else 0.0,
                    "actor_lr": args.actor_lr,
                    "gae_lambda": args.gae_lambda,
                    "clip_eps": args.clip_eps,
                    "runtime_sec": dt,
                    "v_start_final_mean": raw_final_v_start,
                    "std_perf_final_mean": mean_final_perf,
                    "std_perf_final_sem": sem_final_perf,
                    "std_perf_auc_mean": mean_auc_perf,
                    "std_perf_auc_sem": sem_auc_perf,
                    "ve_final_mean": mean_final_ve,
                    "ve_final_sem": sem_final_ve,
                    "ve_auc_mean": mean_auc_ve,
                    "ve_auc_sem": sem_auc_ve,
                    "cossim_final_mean": mean_final_cossim,
                    "cossim_final_sem": sem_final_cossim,
                    "cossim_auc_mean": mean_auc_cossim,
                    "cossim_auc_sem": sem_auc_cossim,
                }
                all_lrs_rows.append(row_data)

                # Determine selection score
                if args.select_by == "std_perf_auc":
                    score = mean_auc_perf  # higher is better
                    higher_better = True
                elif args.select_by == "std_perf_final":
                    score = mean_final_perf  # higher is better
                    higher_better = True
                elif args.select_by == "ve_auc":
                    score = -mean_auc_ve  # lower VE is better, so negate
                    higher_better = True
                else:
                    score = mean_auc_perf
                    higher_better = True

                lr_candidates.append({
                    "critic_lr": c_lr,
                    "return_lambda": r_lam if algo_key != "td0" else 0.0,
                    "score": score,
                    "higher_better": higher_better,
                    "mean_auc_perf": mean_auc_perf,
                    "mean_final_perf": mean_final_perf,
                    "mean_auc_ve": mean_auc_ve,
                    "res": res,
                    "row_data": row_data,
                })

                print(
                    f" Done ({dt:.1f}s) | "
                    f"StdReturn AUC: {mean_auc_perf:.4f}±{sem_auc_perf:.4f} | "
                    f"Final: {mean_final_perf:.4f}±{sem_final_perf:.4f} | "
                    f"AUC VE: {mean_auc_ve:.4f}"
                )

        # Select best condition for algorithm
        if lr_candidates:
            # 1. Best overall condition (for E variants, allows tuning over both 0.95 and 1.0)
            best_cand = max(lr_candidates, key=lambda c: c["score"])
            best_lr = best_cand["critic_lr"]
            best_r_lam = best_cand["return_lambda"]

            full_results["best_critic_lrs"][algo_key] = best_lr
            full_results["best_return_lambdas"][algo_key] = best_r_lam
            full_results["best_conditions"][algo_key] = {"critic_lr": best_lr, "return_lambda": best_r_lam}
            full_results["runs"][algo_key] = best_cand["res"]

            best_row = dict(best_cand["row_data"])
            best_row["best_critic_lr"] = best_lr
            best_row["best_return_lambda"] = best_r_lam
            summary_rows.append(best_row)

            lam_desc = f", ReturnLambda={best_r_lam}" if algo_key.startswith("e0") else ""
            print(f"  ==> Selected Best Condition for {algo_key}: Critic LR={best_lr}{lam_desc} (Score={best_cand['score']:.4f})")

            # 2. For cross-algorithm comparison, record condition strictly with lambda=1.0
            if algo_key.startswith("e0"):
                lam1_candidates = [c for c in lr_candidates if c["return_lambda"] == 1.0]
                if lam1_candidates:
                    best_cand_lam1 = max(lam1_candidates, key=lambda c: c["score"])
                    key_lam1 = f"{algo_key}_lam1"
                    full_results["runs_lam1"][algo_key] = best_cand_lam1["res"]
                    full_results["best_conditions_lam1"][algo_key] = {
                        "critic_lr": best_cand_lam1["critic_lr"],
                        "return_lambda": 1.0,
                    }
                    row_lam1 = dict(best_cand_lam1["row_data"])
                    row_lam1["algo"] = key_lam1
                    row_lam1["display_name"] = f"{algo_info['display_name']} (λ=1)"
                    row_lam1["best_critic_lr"] = best_cand_lam1["critic_lr"]
                    row_lam1["best_return_lambda"] = 1.0
                    summary_rows.append(row_lam1)
            else:
                full_results["runs_lam1"][algo_key] = best_cand["res"]
                full_results["best_conditions_lam1"][algo_key] = {"critic_lr": best_lr, "return_lambda": best_r_lam}

    # Save outputs
    pkl_path = os.path.join(out_env_dir, "out.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(full_results, f)
    print(f"\n[Saved Checkpoint] {pkl_path}")

    csv_path = os.path.join(out_env_dir, "summary.csv")
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)
    print(f"[Saved Summary]    {csv_path}")

    all_csv_path = os.path.join(out_env_dir, "summary_all_lrs.csv")
    df_all = pd.DataFrame(all_lrs_rows)
    df_all.to_csv(all_csv_path, index=False)
    print(f"[Saved All LRs]    {all_csv_path}")

    json_path = os.path.join(out_env_dir, "lr_selection_summary.json")
    with open(json_path, "w") as f:
        json.dump(full_results["best_conditions"], f, indent=2)
    print(f"[Saved LR Config]  {json_path}")

    print("\n" + "=" * 80)
    print(f"Completed Sampled PPO Evaluation for {args.env_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
