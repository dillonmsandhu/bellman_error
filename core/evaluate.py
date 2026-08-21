# This file is responsible for running a single training run, called by runner.py 
from core.utils import save_results, save_plot, save_multi_plot, save_feature_spectra, save_heatmap, save_heatmap_stack
import os
import jax
import jax.numpy as jnp

def evaluate(run_config, make_train, run_dir, args, rng):
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
    
    # Directory structure: results/run_dir/timestamp/EnvName-Size/
    base_env_name = run_config['ENV_NAME']
    env_size = run_config.get("ENV_SIZE")
    
    # Create the full name (e.g., DeepSea-bsuite-45)
    full_env_name = f"{base_env_name}-{env_size}" if env_size else base_env_name
    
    
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
        "nn_lstd_diff": "nn_lstd_diff",
        "forward_loss": "forward_loss",
        "done_loss": "done_loss",
        "reward_loss": "reward_loss",
        "vic_loss_cov": "vic_loss_cov",
        "vic_loss_var": "vic_loss_var",
        "v_loss": "v_loss",
        "E": "E",
        "NTK_rank": "NTK_rank",
        "Direlechet_energy": "Direlechet_energy",
        "V_start": "V_start",
        "v_pred_start": "v_pred_start",
    }
    data = get_metric('E', 1)
    E_local = get_metric('E_local', 1)
    # a few log plots:
    save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, data, 'E', True)
    save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, E_local, 'E_local', True)
    try:
        data = get_metric('alignment_condition', 1)
        save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, E_local, 'E_local', True)
        save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, data, 'E', True)
    except:
        data = get_metric('alignment_condition', 1)
        save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, E_local, 'E_local', False)
        save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, data, 'E', False)
    try:
        save_heatmap(env_dir, run_config['ENV_NAME'], metrics['eNTK'][0,-1], 'ntk')
    except Exception as e:
        print("failed to save ntk", e)
    
    try: 
        save_heatmap_stack(env_dir,
             run_config['ENV_NAME'],
             metrics['Jacobian_top_singular_vectors'][0][-1],
             "J Top Singular Vs")
        save_heatmap_stack(env_dir,
             run_config['ENV_NAME'],
             metrics['feature_top_singular_vectors'][0][-1],
             "Feature Top Singular Vs")
    
    except Exception as e:
        print("failed to save top five jacovian left singular vectors", e)
    
    try: 
        save_feature_spectra(env_dir, 
            run_config['ENV_NAME'], 
            metrics['feature_singular_values'][0][0], 
            metrics['feature_singular_values'][0][-1],
            "Feature Singuar Vals"
        )

        save_feature_spectra(env_dir, 
            run_config['ENV_NAME'], 
            metrics['jacobian_singular_values'][0][0], 
            metrics['jacobian_singular_values'][0][-1],
            "J Singular Vals"
        )
    except Exception as e:
        print("failed to plot singular value spectrum", e)

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
            "Value Learning Greedy Accuracy", 
            "Greedy Accuracy",    # <--- New Y-Label
            {
                "LSTD_greedy_correct": "LSTD Greedy Acc.",
                "VR_greedy_correct": "VR Greedy Acc.",
                "nn_greedy_correct": "Network Greedy Acc.",
                "BR_greedy_correct": "BR Greedy Acc.",
            }, 
            False,
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
