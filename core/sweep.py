def tune(
    make_train,
    base_config,
    param_grid,
    metric_key="nn_weighted_VE",
    save_dir="results/tuning",
    rng_seed=42,
    log_scale=True,
):
    """
    Sweeps over hyperparameters defined in param_grid, computes seed-averaged value error curves,
    plots them, and saves a summary table of performance across configurations.

    Args:
        make_train: Function factory returning the JAX training function.
        base_config: Base dictionary configuration.
        param_grid: Dict mapping hyperparameter names to lists of candidate values.
                    e.g. {'LR': [1e-4, 5e-4, 1e-3], 'GAE_LAMBDA': [0.0, 0.5, 0.95]}
        metric_key: The metric to track and minimize (default: "nn_weighted_VE").
        save_dir: Base directory to save sweep curves and summary tables.
        rng_seed: Base seed for PRNG.
        log_scale: Whether to plot value error curves on a log scale.

    Returns:
        summary_df: pandas.DataFrame containing hyperparameter settings and metrics.
        curves: Dict mapping config label strings to seed-averaged 1D metric trajectories.
    """
    import itertools
    import datetime

    keys = list(param_grid.keys())
    values = [param_grid[k] if isinstance(param_grid[k], (list, tuple)) else [param_grid[k]] for k in keys]
    combinations = list(itertools.product(*values))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = base_config.get("ENV_NAME", "unnamed_env")
    out_dir = os.path.join(save_dir, f"{env_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    curves = {}
    table_rows = []

    print(f"Starting parameter sweep over {len(combinations)} configurations...")

    for idx, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))
        cfg = base_config.copy()
        cfg.update(current_params)

        label = ", ".join([f"{k}={v}" for k, v in current_params.items()])
        print(f"\n[{idx + 1}/{len(combinations)}] Running configuration: {label}")

        n_seeds = cfg.get("N_SEEDS", 1)
        rng = jax.random.PRNGKey(cfg.get("SEED", rng_seed))
        rngs = jax.random.split(rng, n_seeds)

        run_fn = jax.jit(jax.vmap(make_train(cfg)))
        out = run_fn(rngs)
        metrics = out["metrics"]

        if metric_key not in metrics:
            print(f"Warning: Metric '{metric_key}' not found in outputs. Available: {list(metrics.keys())}")
            continue

        raw_metric = jnp.asarray(metrics[metric_key])
        mean_curve = raw_metric.mean(axis=0) if raw_metric.ndim > 1 and raw_metric.shape[0] == n_seeds else raw_metric
        curves[label] = mean_curve

        final_val = float(mean_curve[-1])
        mean_val = float(mean_curve.mean())
        min_val = float(mean_curve.min())

        row = {**current_params}
        row[f"final_{metric_key}"] = final_val
        row[f"mean_{metric_key}"] = mean_val
        row[f"min_{metric_key}"] = min_val
        table_rows.append(row)

        print(f" -> Final {metric_key}: {final_val:.6f} | Mean {metric_key}: {mean_val:.6f}")

    summary_df = pd.DataFrame(table_rows)
    if not summary_df.empty and f"final_{metric_key}" in summary_df.columns:
        summary_df = summary_df.sort_values(by=f"final_{metric_key}", ascending=True).reset_index(drop=True)

    csv_path = os.path.join(out_dir, "tuning_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nTuning summary table saved to {csv_path}")
    print("\nTop 5 Configurations:")
    print(summary_df.head())

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
