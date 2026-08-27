"""
sweep_pipeline.py
A unified, modular hyperparameter sweep and cross-algorithm comparison pipeline.

Usage Examples:
    # 1. Run full sweep over all core algorithms for fixed policy on FourRooms-misc:
    python sweep_pipeline.py --policy fixed --env-name FourRooms-misc --n-seeds 3 --total-timesteps 1000

    # 2. Run sweep for specific algorithms:
    python sweep_pipeline.py --policy fixed --algos exact_td exact_mc --n-seeds 5

    # 3. Use custom learning rate grid:
    python sweep_pipeline.py --policy random --lr-grid 0.05 0.01 0.005 0.001 0.0005 0.0001
"""

import os
import sys
import json
import importlib
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import core.config as default_cfg
from core.sweep import tune
from analyze_sweeps import plot_algorithm_comparison, summarize_algorithm_comparison


# Map algorithm shorthand names to module paths
ALGO_REGISTRY = {
    "fixed": {
        "exact_td": "fixed_policy.exact_td",
        "exact_mc": "fixed_policy.exact_mc",
        "exact_E_gd": "fixed_policy.exact_E_gd",
        "exact_E": "fixed_policy.exact_E_gd",
        "exact_td_lambda": "fixed_policy.exact_td_lambda",
        "exact_td_symmetric": "fixed_policy.exact_td_symmetric",
        "td": "fixed_policy.td",
        "mc": "fixed_policy.mc",
        "monte_carlo": "fixed_policy.mc",
    },
    "random": {
        "exact_td": "random_policy.exact_td",
        "exact_mc": "random_policy.exact_mc",
        "exact_E_gd": "random_policy.exact_E_gd",
        "exact_E": "random_policy.exact_E_gd",
        "exact_td_lambda": "random_policy.exact_td_lambda",
        "exact_td_symmetric": "random_policy.exact_td_symmetric",
        "td": "random_policy.td",
        "mc": "random_policy.mc",
        "monte_carlo": "random_policy.mc",
        "sampled_E": "random_policy.sampled_E",
    },
    "ppo": {
        "exact_td": "ppo.exact_td",
        "exact_mc": "ppo.exact_mc",
        "exact_E": "ppo.exact_E",
        "exact_td_lambda": "ppo.exact_td_lambda",
    },
}

DEFAULT_ALGOS = ["exact_td", "exact_mc", "exact_E_gd", "exact_td_lambda"]
DEFAULT_SAMPLED_ALGOS = ["td", "sampled_E", "monte_carlo"]


def get_default_param_grid(algo_name, lr_list=None):
    """Returns sensible default parameter grids for standard and multi-param algorithms."""
    standard_lrs = lr_list if lr_list is not None else [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    
    if "td_lambda" in algo_name:
        # TD(lambda) requires grid over both LR and VALUE_LAMBDA
        reduced_lrs = [1e-2, 3e-3, 1e-3, 3e-4] if lr_list is None else lr_list
        return {
            "LR": reduced_lrs,
            "VALUE_LAMBDA": [0.0, 0.4, 0.8, 0.95, 1.0],
        }
    else:
        return {"LR": standard_lrs}


def resolve_model_load_dir(model_load_dir, env_name, base_results_dir="results"):
    """Finds a valid existing PPO model load directory containing runner_state for fixed policy evaluation."""
    import cloudpickle
    candidates = []

    # If a dict or JSON string mapping env -> model was passed
    if isinstance(model_load_dir, dict):
        model_load_dir = model_load_dir.get(env_name)
    elif isinstance(model_load_dir, str) and model_load_dir.startswith("{"):
        try:
            mapping = json.loads(model_load_dir)
            model_load_dir = mapping.get(env_name)
        except Exception:
            pass

    if model_load_dir and not str(model_load_dir).startswith("PLACEHOLDER"):
        candidates.append(str(model_load_dir))
        candidates.append(f"ground_truth/{model_load_dir}")
    
    gt_dir = os.path.join(base_results_dir, "ppo", "ground_truth")
    if os.path.exists(gt_dir):
        timestamps = sorted([d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d)) and not d.startswith(".")])
        for ts in reversed(timestamps):
            candidates.append(f"ground_truth/{ts}")

    for cand in candidates:
        env_path = os.path.join(base_results_dir, "ppo", cand, env_name)
        pkl_path = os.path.join(env_path, "out.pkl")
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    data = cloudpickle.load(f)
                if isinstance(data, dict) and "runner_state" in data:
                    print(f"[{env_name}] Auto-resolved MODEL_LOAD_DIR to valid checkpoint: '{cand}'")
                    return cand
            except Exception:
                continue

    return model_load_dir or "ground_truth/short_run"


def run_sweep_pipeline(
    policy_type="fixed",
    env_name="FourRooms-misc",
    algos=None,
    n_seeds=3,
    total_timesteps=1000,
    num_envs=None,
    num_steps=None,
    model_load_dir="short_run",
    metric_key="nn_weighted_VE",
    lr_grid=None,
    custom_grids=None,
    config_overrides=None,
    base_save_dir="results",
    log_scale=True,
    use_geom_mean=False,
):
    """
    Runs a parallel hyperparameter sweep for multiple algorithms on a fixed task,
    saves outputs in an interpretable directory hierarchy, and generates comparison plots.
    
    Returns:
        pipeline_results: dict containing results for each algorithm and comparison artifacts.
    """
    if policy_type == "fixed":
        model_load_dir = resolve_model_load_dir(model_load_dir, env_name, base_save_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_batch_name = f"{policy_type}_{env_name}_{timestamp}"
    sweep_root_dir = os.path.join(base_save_dir, policy_type, "sweeps", sweep_batch_name)
    os.makedirs(sweep_root_dir, exist_ok=True)

    if algos is None or algos == ["exact"]:
        algos = [a for a in DEFAULT_ALGOS if a in ALGO_REGISTRY.get(policy_type, {})]
    elif algos == ["sampled"]:
        algos = [a for a in DEFAULT_SAMPLED_ALGOS if a in ALGO_REGISTRY.get(policy_type, {})]
    elif algos == ["all"]:
        algos = list(ALGO_REGISTRY.get(policy_type, {}).keys())

    print("\n" + "=" * 70)
    print(f"STARTING SWEEP PIPELINE: {policy_type.upper()} POLICY")
    print(f"Environment: {env_name} | Timesteps: {total_timesteps} | Seeds: {n_seeds}")
    if num_envs is not None:
        print(f"NUM_ENVS: {num_envs} | NUM_STEPS: {num_steps}")
    print(f"Algorithms to sweep: {algos}")
    print(f"Output Root Directory: {sweep_root_dir}")
    print("=" * 70 + "\n")

    # Base configuration template
    base_config = default_cfg.config.copy()
    base_config["ENV_NAME"] = env_name
    base_config["N_SEEDS"] = n_seeds
    base_config["TOTAL_TIMESTEPS"] = total_timesteps
    base_config["MODEL_LOAD_DIR"] = model_load_dir
    if num_envs is not None:
        base_config["NUM_ENVS"] = num_envs
    if num_steps is not None:
        base_config["NUM_STEPS"] = num_steps
    if config_overrides:
        base_config.update(config_overrides)

    # Save pipeline run specification
    pipeline_meta = {
        "policy_type": policy_type,
        "env_name": env_name,
        "algos": algos,
        "n_seeds": n_seeds,
        "total_timesteps": total_timesteps,
        "model_load_dir": model_load_dir,
        "metric_key": metric_key,
        "timestamp": timestamp,
        "base_config": base_config,
    }
    with open(os.path.join(sweep_root_dir, "pipeline_config.json"), "w") as f:
        json.dump(pipeline_meta, f, indent=4)

    algorithm_results = {}
    completed_runs_for_comparison = {}

    for algo_idx, algo_name in enumerate(algos):
        print(f"\n[{algo_idx+1}/{len(algos)}] Running sweep for algorithm: {algo_name}...")
        
        module_path = ALGO_REGISTRY.get(policy_type, {}).get(algo_name)
        if not module_path:
            print(f"Warning: Algorithm '{algo_name}' not found in registry for policy '{policy_type}'. Skipping.")
            continue

        try:
            module = importlib.import_module(module_path)
            make_train = getattr(module, "make_train")
        except Exception as e:
            print(f"Error importing {module_path}: {e}")
            continue

        # Determine hyperparameter grid
        if custom_grids and algo_name in custom_grids:
            param_grid = custom_grids[algo_name]
        else:
            param_grid = get_default_param_grid(algo_name, lr_list=lr_grid)

        # Output folder for this specific algorithm inside the sweep root
        algo_save_dir = os.path.join(sweep_root_dir, algo_name, "tuning")

        try:
            res = tune(
                make_train=make_train,
                base_config=base_config.copy(),
                param_grid=param_grid,
                metric_key=metric_key,
                save_dir=algo_save_dir,
                log_scale=log_scale,
                save_metrics=True,
            )
            algorithm_results[algo_name] = res
            completed_runs_for_comparison[algo_name] = res
        except Exception as e:
            print(f"!!! Error during sweep of {algo_name} !!!: {e}")
            import traceback
            traceback.print_exc()

    # Generate Cross-Algorithm Comparison
    print("\n" + "=" * 70)
    print("GENERATING CROSS-ALGORITHM COMPARISON")
    print("=" * 70)

    comparison_dir = os.path.join(sweep_root_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    summary_df = summarize_algorithm_comparison(
        completed_runs_for_comparison,
        metric_key=metric_key,
        save_path=os.path.join(comparison_dir, "comparison_summary.csv")
    )
    # Save JSON format
    summary_df.to_json(os.path.join(comparison_dir, "comparison_summary.json"), orient="records", indent=4)

    print("\nSUMMARY OF BEST PERFORMING CONFIGURATIONS:")
    print(summary_df.to_string(index=False))

    plot_path = os.path.join(comparison_dir, "comparison_best_configs.png")
    plot_algorithm_comparison(
        completed_runs_for_comparison,
        metric_key=metric_key,
        env_name=env_name,
        log_scale=log_scale,
        use_geom_mean=use_geom_mean,
        save_path=plot_path,
        title=f"{policy_type.capitalize()} Policy Evaluation ({env_name}) - Algorithm Comparison",
    )

    print("\n" + "=" * 70)
    print(f"SWEEP PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results & Plots saved in: {sweep_root_dir}")
    print(f"Comparison Plot: {plot_path}")
    print("=" * 70 + "\n")

    return {
        "sweep_root_dir": sweep_root_dir,
        "comparison_dir": comparison_dir,
        "summary_df": summary_df,
        "algorithm_results": algorithm_results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Modular Hyperparameter Sweep Pipeline")
    parser.add_argument("--policy", type=str, default="fixed", choices=["fixed", "random", "ppo"],
                        help="Policy type to evaluate (fixed, random, ppo)")
    parser.add_argument("--env-name", type=str, default="FourRooms-misc",
                        help="Environment name (e.g. FourRooms-misc, MountainCar-v0)")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="List of algorithms to sweep (e.g. exact_td exact_mc exact_E_gd exact_td_lambda)")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of random seeds to evaluate per configuration")
    parser.add_argument("--total-timesteps", type=int, default=1000,
                        help="Total training/evaluation updates (timesteps)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (NUM_ENVS)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Number of rollout steps per env (NUM_STEPS)")
    parser.add_argument("--model-dir", type=str, default="short_run",
                        help="Model load directory for fixed/ppo evaluation policy")
    parser.add_argument("--metric", type=str, default="nn_weighted_VE",
                        help="Primary metric to optimize and rank by")
    parser.add_argument("--lr-grid", nargs="+", type=float, default=None,
                        help="Custom learning rate grid (e.g. --lr-grid 0.01 0.001 0.0001)")
    parser.add_argument("--custom-grids-json", type=str, default=None,
                        help="Path to JSON file specifying custom grids per algorithm")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON string or path to JSON file with additional config overrides")
    parser.add_argument("--use-geom-mean", action="store_true",
                        help="Use geometric mean for error bands in comparison plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    custom_grids = None
    if args.custom_grids_json and os.path.exists(args.custom_grids_json):
        with open(args.custom_grids_json, "r") as f:
            custom_grids = json.load(f)

    config_overrides = None
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                config_overrides = json.load(f)
        else:
            config_overrides = json.loads(args.config)

    run_sweep_pipeline(
        policy_type=args.policy,
        env_name=args.env_name,
        algos=args.algos,
        n_seeds=args.n_seeds,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        model_load_dir=args.model_dir,
        metric_key=args.metric,
        lr_grid=args.lr_grid,
        custom_grids=custom_grids,
        config_overrides=config_overrides,
        use_geom_mean=args.use_geom_mean,
    )
