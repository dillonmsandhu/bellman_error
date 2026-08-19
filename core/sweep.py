import os
import itertools
import datetime
import jax
import jax.numpy as jnp
import pandas as pd
from core.utils import save_multi_plot, save_results
def tune(
    make_train,
    base_config,
    param_grid,
    metric_key="nn_weighted_VE",
    save_dir="results/tuning",
    rng_seed=42,
    log_scale=True,
    save_checkpoint = False,
    save_metrics = True,
):
    """Parallel hyperparameter tuning using nested jax.vmap."""
    keys = list(param_grid.keys())
    values = [param_grid[k] if isinstance(param_grid[k], (list, tuple)) else [param_grid[k]] for k in keys]
    combinations = list(itertools.product(*values))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = base_config.get("ENV_NAME", "unnamed_env")
    out_dir = os.path.join(save_dir, f"{timestamp}/{env_name}")
    os.makedirs(out_dir, exist_ok=True)

    n_seeds = base_config.get("N_SEEDS", 1)
    rng = jax.random.PRNGKey(base_config.get("SEED", rng_seed))
    rngs = jax.random.split(rng, n_seeds)

    # Create a PyTree of hyperparameters to vmap over
    hparams_tree = {
        k: jnp.array([combo[i] for combo in combinations])
        for i, k in enumerate(keys)
    }

    train_fn = make_train(base_config)
    
    # Inner vmap over seeds (axis 0 of rngs, None for hparams)
    # Outer vmap over configurations (None for rngs, axis 0 of hparams PyTree)
    parallel_train = jax.jit(
        jax.vmap(
            jax.vmap(train_fn, in_axes=(0, None)),
            in_axes=(None, 0)
        )
    )

    print(f"Running parallel sweep for {len(combinations)} configurations x {n_seeds} seeds...")
    out = parallel_train(rngs, hparams_tree)
    metrics = out["metrics"]

    if metric_key not in metrics:
        raise KeyError(f"Metric '{metric_key}' not found. Available keys: {list(metrics.keys())}")

    # metrics[metric_key] shape: (n_combos, n_seeds, time_steps)
    metric_tensor = jnp.asarray(metrics[metric_key])
    mean_trajectories = metric_tensor.mean(axis=1)    # shape: (n_combos, time_steps)

    curves = {}
    table_rows = []

    for idx, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))
        label = ", ".join([f"{k}={v}" for k, v in current_params.items()])

        curve = mean_trajectories[idx]
        curves[label] = curve

        final_val = float(curve[-1])
        mean_val = float(curve.mean())
        min_val = float(curve.min())

        row = {"config_idx": idx, **current_params}
        row[f"final_{metric_key}"] = final_val
        row[f"mean_{metric_key}"] = mean_val
        row[f"min_{metric_key}"] = min_val
        table_rows.append(row)

    summary_df = pd.DataFrame(table_rows)
    if not summary_df.empty and f"final_{metric_key}" in summary_df.columns:
        summary_df = summary_df.sort_values(by=f"final_{metric_key}", ascending=True).reset_index(drop=True)

    csv_path = os.path.join(out_dir, "tuning_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nTuning summary saved to {csv_path}")
    print("\nTop configurations:")
    print(summary_df.head())
    
    if save_checkpoint:
        save_results(out, base_config, env_name, out_dir)
    elif save_metrics:
        save_results(metrics, base_config, env_name, out_dir)
    else: # save config only
        save_results(base_config, base_config, env_name, out_dir)
    
    steps_per_pi = base_config.get("NUM_ENVS", 1) * base_config.get("NUM_STEPS", 1)
    save_multi_plot(
        env_dir=out_dir,
        env_name=env_name,
        steps_per_pi=steps_per_pi,
        metrics_dict=curves,
        title=f"hyperparameter_sweep_{metric_key}",
        ylabel=metric_key,
        log_scale=log_scale,
    )

    return summary_df, curves
