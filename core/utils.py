# This file contains helpers relating to logging, checkpointing, and loading the data.
import os
import yaml
import json
import cloudpickle
import matplotlib.pyplot as plt
from core.networks import *
import pandas as pd

# MAIN function in most algos.
def run_experiment_main(make_train, SAVE_DIR):
    import argparse
    import datetime
    import traceback
    import core.helpers as helpers
    import core.config as config
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-suffix', type=str, default=run_timestamp)
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--save-metrics', action='store_true')
    parser.add_argument('--env-ids', nargs='+', default=[])

    args = parser.parse_args()

    config = config.config
    env_list = [config.get('ENV_NAME')]

    # 3. Environment Priority (CLI takes precedence)
    if args.env_ids:
        env_list = args.env_ids

    for i, env_name in enumerate(env_list):
        if env_name is None: continue
            
        # Create a clean copy for this specific environment run
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        
        # Apply command-line JSON overrides if they exist
        if args.config:
            from core.utils import parse_config_override
            run_config.update(parse_config_override(args.config))
            
        if args.n_seeds > 0:
            run_config['N_SEEDS'] = args.n_seeds

        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name}")
        print(f"Config: {args.config}")
        # print(f"Network: {run_config.get('NETWORK_TYPE')}")
        print(f"{'='*50}")
        
        rng = jax.random.PRNGKey(run_config.get('SEED', 42))
        
        try:
            # Note: make_train and evaluate should be defined in your scope
            evaluate(run_config, make_train, SAVE_DIR, args, rng)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!")
            traceback.print_exc()
            print("Continuing to next environment...")

def parse_config_override(config_str):
    """Parse config override from command line argument."""
    if config_str is None:
        return {}
    
    try:
        # Parse as JSON
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing config override: {e}")
        print("Config override should be valid JSON, e.g.: '{\"LR\": 0.001, \"LAMBDA\": 0.0}'")
        exit(1)

def save_config(config, env_dir):
    config_path = os.path.join(env_dir, f"config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

def save_results(data, config, env_name, env_dir):
    # Create a subdirectory for the environment within the main run directory
    os.makedirs(env_dir, exist_ok=True)

    # Save the pickle file
    pickle_path = os.path.join(env_dir, "out.pkl")
    with open(pickle_path, 'wb') as f:
        cloudpickle.dump(data, f)
    print(f"Results saved to {pickle_path}")
        
    save_config(config, env_dir)
    print(f"Config saved to {os.path.join(env_dir, f'config.json')}")

    return env_dir

def save_plot(env_dir, env_name, steps_per_pi, episodic_return, title):
    y = jnp.asarray(episodic_return)
    if y.ndim == 0:
        y = y[None]
    if y.ndim != 1:
        print(f"Skipping plot {title}: expected 1D series, got shape {tuple(y.shape)}")
        return
    if y.shape[0] == 0:
        print(f"Skipping plot {title}: empty series")
        return

    plt.figure()
    x = [i * steps_per_pi for i in range(int(y.shape[0]))]
    plt.plot(x, y, 'o-', label=title)
    plt.xlabel("Env. Step")
    plt.ylabel(f"{title}")
    plt.title(env_name)
    plt.legend()

    # Save plot as a .png file in the environment directory
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def save_multi_plot(env_dir, env_name, steps_per_pi, metrics_dict, title="Value_Errors", ylabel="Mean Squared Error (MSVE)", log_scale=False):
    """
    Plots multiple 1D metrics on a single graph.
    
    Args:
        env_dir: Directory to save the plot.
        env_name: Name of the environment (for the title).
        steps_per_pi: Number of steps between data points.
        metrics_dict: Dictionary mapping metric names to 1D arrays/lists.
        title: The filename and y-axis label.
    """
    plt.figure(figsize=(10, 6))
    lines_plotted = 0

    for label, data in metrics_dict.items():
        y = jnp.asarray(data)
        
        if y.ndim == 0:
            y = y[None]
        if y.ndim != 1:
            print(f"Skipping line {label}: expected 1D series, got shape {tuple(y.shape)}")
            continue
        if y.shape[0] == 0:
            print(f"Skipping line {label}: empty series")
            continue

        x = [i * steps_per_pi for i in range(y.shape[0])]
        # Using a smaller markersize is usually better when lines overlap
        plt.plot(x, y, 'o-', label=label, markersize=3, alpha=0.8)
        lines_plotted += 1

    if lines_plotted == 0:
        print(f"Skipping plot {title}: no valid data series provided.")
        plt.close()
        return

    plt.xlabel("Env. Step")
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')
    plt.title(f"{env_name} - {title}")
    
    # Adding a grid makes it much easier to compare error asymptotes visually
    plt.grid(True, linestyle='--', alpha=0.6) 
    plt.legend()

    # Save plot
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path, bbox_inches='tight') # bbox_inches prevents cut-off legends
    plt.close()
    print(f"Plot saved to {plot_path}")

def load_run_data(run_folder_name, env_name, results_base_path="../results"):
    """
    Load the configuration and output data for a run given the run folder and environment.
    
    Parameters:
    - run_folder_name (str): The timestamped run folder name, e.g., "dpi_20241110_193658"
    - env_name (str): The environment name, e.g., "Asterix-MinAtar"
    - results_base_path (str): Base path to the results directory, default is a sibling "results" directory.
    
    Returns:
    - config (dict): Loaded JSON configuration.
    - results (object): Loaded output data from pickle.
    """
    # Construct paths
    run_path = os.path.join(results_base_path, run_folder_name, env_name)
    config_path = os.path.join(run_path, "config.json")
    results_path = os.path.join(run_path, "out.pkl")
    
    # Load the config
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    
    # Load the results
    with open(results_path, 'rb') as pkl_file:
        results = cloudpickle.load(pkl_file)
    
    
    return config, results

def evaluate(run_config, make_train, SAVE_DIR, args, rng):
    # Setup specific to this run_config
    steps_per_pi = run_config["NUM_ENVS"] * run_config["NUM_STEPS"]
    
    # JIT the train function for this specific config (important if env changes)
    run_fn = jax.jit(jax.vmap(make_train(run_config)))
    
    rngs = jax.random.split(rng, run_config['N_SEEDS'])
    out = run_fn(rngs)
    metrics = out["metrics"]
    ret = metrics.get('returned_discounted_episode_returns', 0.0)
    print(f"[{run_config['ENV_NAME']}] Mean return: {jnp.mean(ret):.4f}")
    print(f"[{run_config['ENV_NAME']}] Max return:  {jnp.max(ret):.4f}")
    
    # Directory structure: results/SAVE_DIR/timestamp/EnvName-Size/
    base_env_name = run_config['ENV_NAME']
    env_size = run_config.get("ENV_SIZE")
    
    # Create the full name (e.g., DeepSea-bsuite-45)
    full_env_name = f"{base_env_name}-{env_size}" if env_size else base_env_name
    
    run_dir = os.path.join("results", f"{SAVE_DIR}/{args.run_suffix}")
    env_dir = os.path.join(run_dir, full_env_name)
    
    os.makedirs(env_dir, exist_ok=True)
    print(f"Saving {full_env_name} results to {env_dir}")

    # Ensure save_results uses the full name for the filename
    if args.save_checkpoint:
        save_results(out, run_config, full_env_name, env_dir)
    elif args.save_metrics:
        save_results(metrics, run_config, full_env_name, env_dir)
    else: # save config only
        save_results(run_config, run_config, full_env_name, env_dir)
    
    
    # --- Helper for Metrics extraction ---
    def _mean_over_seeds(data):
        arr = jnp.asarray(data)
        if arr.ndim > 0 and arr.shape[0] == run_config['N_SEEDS']:
            arr = arr.mean(0)
        return arr

    def _extract_series(data):
        arr = _mean_over_seeds(data)
        if arr.ndim == 0:
            return arr[None]
        if arr.ndim == 1:
            return arr

    def get_metric(name, slice_idx=0):
        if name not in metrics:
            return None
        series = _extract_series(metrics[name])
        return series[slice_idx:]

    standard_plots = {
        'v_pred': 'v_pred',
        "mean_rew": "mean_rew",
        "returned_episode_returns": "returned_episode_returns",
        "returned_discounted_episode_returns": "returned_discounted_episode_returns",
        "effective_rank": "effective_rank",
        "capacity_angle": "capacity_angle",
        "nn_lstd_diff": "nn_lstd_diff",
        "forward_loss": "forward_loss",
        "done_loss": "done_loss",
        "reward_loss": "reward_loss",
        "vic_loss_cov": "vic_loss_cov",
        "vic_loss_var": "vic_loss_var",
        "v_loss": "v_loss",
        "E": "E",
        "alignment_condition": "Alignment Condition",
        "alignment": "Alignment (cosine similarity)",
        "SA_min_eigenvalue": "Min. Eigenvalue of SA",
        "is_SA_positive_definite": "Is SA Positive Definite",
        "norm_s": "norm_s",
        "norm_k": "norm_k"
    }

    for m_key, save_name in standard_plots.items():
        data = get_metric(m_key, 1)
        if data is not None:
            try:
                save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, data, save_name)
            except:
                print('failed to save plot for', m_key)

# 1. Add the ylabel string to each configuration tuple
    plot_configs = [
        (
            "Value Learning Greedy Accuracy (All)", 
            "Greedy Accuracy",    # <--- New Y-Label
            {
                "LSTD_greedy_correct": "LSTD Greedy Acc.",
                "nn_greedy_correct": "Network Greedy Acc.",
                "VR_greedy_correct": "VR Greedy Acc.",
                "BR_greedy_correct": "BR Greedy Acc.",
                "BR_uniform_greedy_correct": "BR uniform Greedy Acc.",
                "LSTD_uniform_greedy_correct": "LSTD uniform Greedy Acc.",
                "VR_uniform_greedy_correct": "VR uniform Greedy Acc.",
            }, 
            False,
        ),
        (
            "(Uniform Weighted) Value Errors",
            "MSVE (equal state weighting)",      # <--- New Y-Label
            {
                "LSTD_VE": "LSTD (on-policy) VE",
                "VR_VE": "VR (on-policy) VE",
                "nn_VE": "NN (on-policy) VE",
                "LSTD_uniform_VE": "LSTD (uniform) VE",
                "VR_uniform_VE": "VR (uniform) VE",
                "BR_uniform_VE": "BR (uniform) VE",
                # "BR_VE": "BR VE"
            },
            True,
        ),
        (
            "Weighted Value Errors",
            "MSVE (mu-weighted)",      # <--- New Y-Label
            {
                "LSTD_weighted_VE": "LSTD (on-policy) VE",
                "VR_weighted_VE": "VR (on-policy) VE",
                "nn_weighted_VE": "NN (on-policy) VE",
                "BR_VE": "BR (on-policy) VE"
            },
            True
        ),
        (
            "Value Learning Greedy Accuracy (On-Policy estimators)", 
            "Greedy Accuracy",    # <--- New Y-Label
            {
                "LSTD_greedy_correct": "LSTD Greedy Acc.",
                "VR_greedy_correct": "VR Greedy Acc.",
                "nn_greedy_correct": "Network Greedy Acc.",
                "BR_greedy_correct": "BR Greedy Acc.",
            }, 
            False,
        ),
        (
            "Weighted Projected Bellman Error",
            "MSVE (mu-weighted)",      # <--- New Y-Label
            {
                "LSTD_weighted_PBE": "LSTD (on-policy) PBE",
                "VR_weighted_PBE": "VR (on-policy) PBE",
                "nn_weighted_PBE": "NN (on-policy) PBE",
                "BR_weighted_PBE": "BR (on-policy) PBE"
            },
            True
        ),
        (
            "Weighted Bellman Residual",
            "MSBE (mu-weighted)",      # <--- New Y-Label
            {
                "LSTD_weighted_BE": "LSTD (on-policy) BE",
                "VR_weighted_BE": "VR (on-policy) BE",
                "nn_weighted_BE": "NN (on-policy) BE",
                "BR_weighted_BE": "BR (on-policy) BE"
            },
            True
        ),
        
    ]

    # 2. Unpack title, ylabel, and metric_keys
    for title, ylabel, metric_keys, logscale in plot_configs:
        
        plot_data = {legend: get_metric(m_key, 1) for m_key, legend in metric_keys.items()}
        
        save_multi_plot(
            env_dir=env_dir, 
            env_name=run_config['ENV_NAME'], 
            steps_per_pi=steps_per_pi, 
            metrics_dict=plot_data, 
            title=title,
            ylabel=ylabel,
            log_scale=logscale
        )