"""
scripts/bsmc_sweep/run_single_env.py

Executes sampled PPO comparing:
  1. td_lambda:    Standard TD(lambda) critic baseline
  2. e_lambda:     Geometric jump sampled E(lambda) critic [v_aux] (m ~ Geom(1 - gamma * lambda))
  3. e_lambda_fwd: Geometric jump sampled E(lambda) critic [Forward-Only]
  4. bsmc_aux:     1-Step Bellman-Smoothed Monte Carlo critic [v_aux] (m = 1, matched MC weight)
  5. bsmc_fwd:     1-Step Bellman-Smoothed Monte Carlo critic [Forward-Only] (m = 1, matched MC weight)

Across matching lambda values [0.40, 0.80, 0.95, 0.99] and swept critic learning rates.

Disentangles whether the advantage of high lambda in E(lambda) is due to:
  (A) Multi-step geometric jumps across long state horizons, OR
  (B) Increased weight on the unbiased Monte Carlo return anchor (1 - alpha) = (1 - gamma) / (1 - gamma * lambda).
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
from ppo.sampled_bsmc import make_train as make_bsmc_train

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
    "bsmc_aux": {
        "display_name": "BSMC [v_aux, 1-step]",
        "cond_prefix": "bsmc_aux",
        "train_builder": make_bsmc_train,
        "base_hparams": {"BSMC_VARIANT": "aux", "RETURN_LAMBDA": 1.0},
        "color_base": "#9467bd",  # Purple base
    },
    "bsmc_fwd": {
        "display_name": "BSMC [Forward-Only, 1-step]",
        "cond_prefix": "bsmc_fwd",
        "train_builder": make_bsmc_train,
        "base_hparams": {"BSMC_VARIANT": "forward", "RETURN_LAMBDA": 1.0},
        "color_base": "#e377c2",  # Pink base
    },
}

LAMBDA_COLORS = {
    # TD(lambda) - Orange shades
    "td_0.40": "#fdd0a2", "td_0.80": "#fdae6b", "td_0.95": "#e6550d", "td_0.99": "#8c2d04",
    # E(lambda) [v_aux] - Green shades
    "e_0.40": "#c7e9c0", "e_0.80": "#74c476", "e_0.95": "#238b45", "e_0.99": "#00441b",
    # E(lambda) [Forward-Only] - Blue shades
    "e_fwd_0.40": "#c6dbef", "e_fwd_0.80": "#6baed6", "e_fwd_0.95": "#2171b5", "e_fwd_0.99": "#08306b",
    # BSMC [v_aux] - Purple shades
    "bsmc_aux_0.40": "#dadaeb", "bsmc_aux_0.80": "#bcbddc", "bsmc_aux_0.95": "#9e9ac8", "bsmc_aux_0.99": "#6a51a3",
    # BSMC [Forward-Only] - Pink / Magenta shades
    "bsmc_fwd_0.40": "#fde0dd", "bsmc_fwd_0.80": "#fa9fb5", "bsmc_fwd_0.95": "#e7298a", "bsmc_fwd_0.99": "#67001f",
}


def parse_args():
    parser = argparse.ArgumentParser(description="BSMC vs E(lambda) Disentanglement Sweep on a Single Env")
    parser.add_argument("--env_name", type=str, default="FourRooms-misc", help="Name of the environment")
    parser.add_argument("--env_idx", type=int, default=None, help="Index into ALL_ENVS (0-8)")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.4, 0.8, 0.95, 0.99], help="Lambda values to evaluate")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of rollout updates (default: 100)")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--k", type=int, default=32, help="Critic feature dimension (default: 32)")
    parser.add_argument("--actor_lr", type=float, default=0.0003, help="Actor learning rate (default: 0.0003)")
    parser.add_argument("--critic_lrs", type=float, nargs="+", default=[0.0005, 0.001, 0.003], help="Critic learning rates to sweep")
    parser.add_argument("--num_envs", type=int, default=32, help="Parallel environments (default: 32)")
    parser.add_argument("--num_steps", type=int, default=256, help="Steps per rollout (default: 256)")
    parser.add_argument("--minibatch_size", type=int, default=256, help="Minibatch size (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Epochs per update (default: 4)")
    parser.add_argument("--gae_lambda", type=float, default=0.8, help="Actor GAE lambda (default: 0.8)")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon (default: 0.2)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--runway_length", type=int, default=16, help="Runway window length (default: 16)")
    parser.add_argument("--select_by", type=str, default="std_perf_auc", choices=["std_perf_auc", "std_perf_final", "ve_auc"], help="Metric to select best condition")
    parser.add_argument("--out_dir", type=str, default="results/bsmc_sweep", help="Root output directory")
    parser.add_argument("--algos", type=str, nargs="+", default=["td_lambda", "e_lambda", "e_lambda_fwd", "bsmc_aux", "bsmc_fwd"], help="Algorithms to evaluate")
    parser.add_argument("--calc_true_values", action="store_true", default=False, help="Calculate tabular true values during rollout updates (default: False)")
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
        "TOTAL_TIMESTEPS": args.num_updates * batch_size,
        "NUM_UPDATES": args.num_updates,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "MINIBATCH_SIZE": args.minibatch_size,
        "NUM_EPOCHS": args.num_epochs,
        "LR": critic_lr,
        "CRITIC_LR": critic_lr,
        "ACTOR_LR": args.actor_lr,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "GAMMA": args.gamma,
        "k": args.k,
        "NETWORK_TYPE": "mlp" if is_mc else "cnn",
        "NORMALIZE_OBS": is_mc,
        "RUNWAY_LENGTH": args.runway_length,
        "CALC_TRUE_VALUES": args.calc_true_values,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
        "LOG_VARIANCE_METRICS": False,
    })
    return cfg


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
    print("BSMC vs E(lambda) Disentanglement Sweep on Single Environment")
    print(f"Environment:       {args.env_name}")
    print(f"Num Updates:       {args.num_updates}")
    print(f"Rollout Config:    {args.num_envs} envs x {args.num_steps} steps (batch={args.num_envs * args.num_steps})")
    print(f"Num Seeds:         {args.n_seeds} (base seed: {args.seed})")
    print(f"Critic Feature k:  {args.k}")
    print(f"Actor LR:          {args.actor_lr} (fixed)")
    print(f"Critic LRs:        {args.critic_lrs}")
    print(f"Lambdas:           {args.lambdas}")
    print(f"Selection Metric:  {args.select_by}")
    print(f"Algorithms:        {args.algos}")
    print(f"Baselines:         V* = {v_star:.4f} | V_rand = {v_rand:.4f} | Delta = {delta:.4f}")
    out_env_dir = os.path.join(os.path.abspath(args.out_dir), args.env_name)
    os.makedirs(out_env_dir, exist_ok=True)
    print(f"Output Directory:  {out_env_dir}")
    print("=" * 80)

    # Print calibrated weight schedule for reference
    print("\n--- Analytical MC Weight Equivalence Schedule ---")
    for lam in args.lambdas:
        w_mc, w_dir, alpha = helpers.compute_bsmc_weights(args.gamma, lam)
        mc_pct = (1.0 - alpha) * 100.0
        smooth_pct = alpha * 100.0
        print(f"  Lambda = {lam:0.2f} --> MC Weight (1 - alpha) = {mc_pct:5.2f}% | 1-Step Smoothing alpha = {smooth_pct:5.2f}%")
    print("-------------------------------------------------\n")

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
        cond_prefix = algo_info["cond_prefix"]
        for lam in args.lambdas:
            lam_str = f"{lam:.2f}"
            cond_key = f"{cond_prefix}_{lam_str}"
            display_name = f"{algo_info['display_name']}({lam_str})"

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
                full_results["all_runs"][cond_key][str(c_lr)] = res

                # Calculate standardized metrics across seeds
                # Performance metrics
                v_start_all = res.get("V_start", res.get("v_pred_start", None))
                if v_start_all is None:
                    v_start_all = res.get("returned_episode_returns", np.zeros((args.n_seeds, args.num_updates)))
                
                norm_perf = (v_start_all - v_rand) / delta
                perf_auc_per_seed = trapz_fn(norm_perf, axis=1) / (args.num_updates - 1)
                std_perf_auc = float(np.mean(perf_auc_per_seed))
                std_perf_final = float(np.mean(norm_perf[:, -1]))

                # Episode return metrics
                ep_rets = res.get("returned_episode_returns", v_start_all)
                final_ep_ret_mean = float(np.mean(ep_rets[:, -1]))

                # Value error metrics if available
                ve_vals = res.get("nn_weighted_VE", None)
                if ve_vals is not None and not np.isnan(ve_vals).all():
                    ve_auc = float(np.mean(trapz_fn(ve_vals, axis=1) / (args.num_updates - 1)))
                    ve_final = float(np.mean(ve_vals[:, -1]))
                else:
                    ve_auc = float("nan")
                    ve_final = float("nan")

                rt = res.get("_runtime_sec", 0.0)

                print(f" done ({rt:.1f}s) | Std_Perf_AUC = {std_perf_auc:+.4f} | Final = {std_perf_final:+.4f} | EpRet = {final_ep_ret_mean:.2f}")

                candidate_info = {
                    "critic_lr": c_lr,
                    "std_perf_auc": std_perf_auc,
                    "std_perf_final": std_perf_final,
                    "final_ep_ret_mean": final_ep_ret_mean,
                    "ve_auc": ve_auc,
                    "ve_final": ve_final,
                    "runtime_sec": rt,
                }
                lr_candidates.append(candidate_info)

                all_lrs_rows.append({
                    "Environment": args.env_name,
                    "Algorithm": algo_key,
                    "Algo_Display": algo_info["display_name"],
                    "Lambda": float(lam),
                    "Critic_LR": c_lr,
                    "Condition_Key": cond_key,
                    "Std_Perf_AUC": std_perf_auc,
                    "Std_Perf_Final": std_perf_final,
                    "Final_Ep_Ret": final_ep_ret_mean,
                    "VE_AUC": ve_auc,
                    "VE_Final": ve_final,
                    "Runtime_Sec": rt,
                })

            # Select winning learning rate according to selection metric
            if args.select_by == "std_perf_auc":
                best_cand = max(lr_candidates, key=lambda c: c["std_perf_auc"])
            elif args.select_by == "std_perf_final":
                best_cand = max(lr_candidates, key=lambda c: c["std_perf_final"])
            elif args.select_by == "ve_auc":
                valid_cands = [c for c in lr_candidates if not np.isnan(c["ve_auc"])]
                best_cand = min(valid_cands, key=lambda c: c["ve_auc"]) if valid_cands else lr_candidates[0]
            else:
                best_cand = lr_candidates[0]

            best_lr = best_cand["critic_lr"]
            full_results["best_critic_lrs"][cond_key] = best_lr
            full_results["best_conditions"][cond_key] = best_cand
            full_results["runs"][cond_key] = full_results["all_runs"][cond_key][str(best_lr)]

            print(f"  [WINNING CRITIC LR for {cond_key}]: {best_lr} (Std_Perf_AUC = {best_cand['std_perf_auc']:+.4f})")

            # Extract seed-level metrics for winning condition
            win_res = full_results["runs"][cond_key]
            v_start_win = win_res.get("V_start", win_res.get("v_pred_start", None))
            if v_start_win is None:
                v_start_win = win_res.get("returned_episode_returns", np.zeros((args.n_seeds, args.num_updates)))
            norm_perf_win = (v_start_win - v_rand) / delta
            perf_auc_per_seed = trapz_fn(norm_perf_win, axis=1) / (args.num_updates - 1)
            final_perf_per_seed = norm_perf_win[:, -1]

            w_mc, w_dir, alpha = helpers.compute_bsmc_weights(args.gamma, float(lam))

            summary_rows.append({
                "Environment": args.env_name,
                "Algorithm": algo_key,
                "Algo_Display": algo_info["display_name"],
                "Lambda": float(lam),
                "MC_Weight_Pct": float((1.0 - alpha) * 100.0),
                "Smoothing_Weight_Pct": float(alpha * 100.0),
                "Best_Critic_LR": best_lr,
                "Condition_Key": cond_key,
                "Std_Perf_AUC_Mean": float(np.mean(perf_auc_per_seed)),
                "Std_Perf_AUC_SEM": float(np.std(perf_auc_per_seed, ddof=1) / np.sqrt(args.n_seeds)),
                "Std_Perf_Final_Mean": float(np.mean(final_perf_per_seed)),
                "Std_Perf_Final_SEM": float(np.std(final_perf_per_seed, ddof=1) / np.sqrt(args.n_seeds)),
                "Final_Ep_Ret_Mean": best_cand["final_ep_ret_mean"],
                "VE_AUC": best_cand["ve_auc"],
                "VE_Final": best_cand["ve_final"],
                "Runtime_Sec": best_cand["runtime_sec"],
            })

    # Save output artifacts
    out_pkl = os.path.join(out_env_dir, "out.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(full_results, f)
    print(f"\n[Saved full output dict]: {out_pkl}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_env_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved winning summary CSV]: {summary_csv}")

    all_lrs_df = pd.DataFrame(all_lrs_rows)
    all_lrs_csv = os.path.join(out_env_dir, "summary_all_lrs.csv")
    all_lrs_df.to_csv(all_lrs_csv, index=False)
    print(f"[Saved all LRs summary CSV]: {all_lrs_csv}")

    json_path = os.path.join(out_env_dir, "lr_selection_summary.json")
    with open(json_path, "w") as f:
        json.dump({k: full_results["best_conditions"][k] for k in full_results["best_conditions"]}, f, indent=2)
    print(f"[Saved LR selection JSON]: {json_path}")

    # Display final comparative table in console
    print("\n" + "=" * 100)
    print(f"FINAL WINNING CONDITIONS SUMMARY: {args.env_name}")
    print("=" * 100)
    cols_to_print = ["Algo_Display", "Lambda", "MC_Weight_Pct", "Best_Critic_LR", "Std_Perf_AUC_Mean", "Std_Perf_AUC_SEM", "Std_Perf_Final_Mean", "Final_Ep_Ret_Mean"]
    print(summary_df[cols_to_print].to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
