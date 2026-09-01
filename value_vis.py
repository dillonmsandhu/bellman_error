# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "jax",
#     "jaxlib",
#     "gymnax",
#     "matplotlib",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    from core.config import config
    from core.helpers import make_env, initialize_evaluator
    from analyze_runs import (
        load_fixed_policy,
        compute_ground_truth,
        resolve_runs,
        get_run_value_grid,
        plot_grid,
        plot_multi_grids,
        plot_error_grids,
        plot_3d_comparison,
        render_3d,
        compute_jaggedness,
        compute_runs_jaggedness,
        plot_feature_singular_vectors,
    )

    return (
        compute_ground_truth,
        compute_runs_jaggedness,
        config,
        get_run_value_grid,
        initialize_evaluator,
        load_fixed_policy,
        make_env,
        mo,
        plot_3d_comparison,
        plot_error_grids,
        plot_feature_singular_vectors,
        plot_multi_grids,
        render_3d,
        resolve_runs,
    )


@app.cell
def _():
    # ---------------------------------------------------------------------------
    # User Configuration: Policy Selection & Runs Specification
    # ---------------------------------------------------------------------------
    # Set policy_type to "random" or "fixed"
    policy_type = "random"
    fixed_policy_dir = "ppo/ground_truth/short_run"  # Pretrained policy directory

    # Dictionary of runs to compare:
    # Each entry defines run_dir (or direct metrics), title, and color.
    runs_spec = {
        "TD": {
            "run_dir": "random/td_exact/tuned_saved_metrics/",
            "title": "Temporal Difference (TD)",
            "color": "blue",
        },
        "MC": {
            "run_dir": "random/mc_exact/tuned_saved_metrics/",
            "title": "Monte Carlo (MC)",
            "color": "red",
        },
        "E": {
            "run_dir": "random/exact_E/tuned_saved_metrics/",
            "title": "Expected Update (E)",
            "color": "green",
        },
    }

    epoch_idx = -1  # Final checkpoint index
    return epoch_idx, fixed_policy_dir, policy_type, runs_spec


@app.cell
def _(
    compute_ground_truth,
    config,
    fixed_policy_dir,
    initialize_evaluator,
    load_fixed_policy,
    make_env,
    policy_type,
):
    # Initialize Environment & Evaluator
    env, env_params = make_env(config)
    evaluator = initialize_evaluator(config, env, env_params)

    # Optional Fixed Policy Loading
    fixed_policy_fn = None
    policy_train_state = None
    if policy_type == "fixed":
        try:
            fixed_policy_fn, policy_train_state, _ = load_fixed_policy(
                fixed_policy_dir, env_name=config.get("ENV_NAME", "FourRooms-misc")
            )
            print(f"Successfully loaded fixed policy from {fixed_policy_dir}")
        except Exception as e:
            print(f"Warning: Could not load fixed policy from '{fixed_policy_dir}': {e}")

    # Compute Ground Truth (V^pi, mu, grids)
    gt = compute_ground_truth(
        evaluator, policy_type=policy_type, fixed_policy_fn=fixed_policy_fn
    )
    return evaluator, gt


@app.cell
def _(epoch_idx, evaluator, get_run_value_grid, gt, resolve_runs, runs_spec):
    # Resolve all specified runs (load data where available or fallback smoothly)
    resolved_runs = resolve_runs(runs_spec)

    run_grids = {}
    for key, rinfo in resolved_runs.items():
        grid = None
        if rinfo["metrics"] is not None:
            try:
                grid = get_run_value_grid(rinfo, evaluator, epoch_idx=epoch_idx)
            except Exception as e:
                print(f"Could not extract value grid for {key}: {e}")
                grid = None

        # If data is not available locally, fallback to ground truth demo for smooth visual preview
        run_grids[key] = {
            "title": rinfo["title"],
            "color": rinfo["color"],
            "grid": grid if grid is not None else gt["v_grid"],
            "is_loaded": grid is not None,
        }
    return resolved_runs, run_grids


@app.cell
def _(evaluator, gt, plot_multi_grids):
    # Ground Truth: True Value Function and Stationary Distribution
    fig_gt = plot_multi_grids(
        {
            "True Value Function ($V^\pi$)": gt["v_grid"],
            "Stationary Distribution ($\mu$)": gt["mu_grid"],
        },
        evaluator,
        cmaps={
            "True Value Function ($V^\pi$)": "viridis",
            "Stationary Distribution ($\mu$)": "magma",
        },
        use_log=False,
        shared_vrange=False,
    )
    fig_gt
    return


@app.cell
def _(evaluator, plot_multi_grids, run_grids):
    # Side-by-side comparison of learned value functions
    display_grids = {info["title"]: info["grid"] for info in run_grids.values()}
    fig_values = plot_multi_grids(
        display_grids,
        evaluator,
        cmap="viridis",
        shared_vrange=True,
    )
    fig_values
    return


@app.cell
def _(evaluator, gt, plot_error_grids, run_grids):
    # Side-by-side comparison of value errors (MSE)
    error_input = {key: info["grid"] for key, info in run_grids.items()}
    fig_errors = plot_error_grids(
        error_input,
        gt["v_grid"],
        evaluator,
        metric_type="squared_error",
        use_log=True,
        shared_vrange=True,
    )
    fig_errors
    return


@app.cell
def _(mo):
    mo.md("""
    # Interactive 3D Analysis
    Use the sliders below to rotate the 3D view.
    - **Left Plot**: Compares the **True Value Surface** with the **Learned Value Wireframes**.
    - **Right Plot**: Shows the **Underestimation Error** ($V_{true} - V_{learned}$).
    """)
    return


@app.cell
def _(mo, runs_spec):
    elev_slider = mo.ui.slider(0, 90, step=5, value=30, label="Elevation")
    azim_slider = mo.ui.slider(-180, 180, step=5, value=-60, label="Azimuth (Rotation)")

    # Dynamic checkboxes for each algorithm in runs_spec
    run_toggles = {
        key: mo.ui.checkbox(
            value=True, label=f"Show {runs_spec[key].get('title', key)}"
        )
        for key in runs_spec
    }
    return azim_slider, elev_slider, run_toggles


@app.cell
def _(
    azim_slider,
    elev_slider,
    evaluator,
    gt,
    mo,
    render_3d,
    run_grids,
    run_toggles,
):
    # Render interactive 3D plots
    toggle_values = {key: toggle.value for key, toggle in run_toggles.items()}

    controls = mo.hstack(
        [elev_slider, azim_slider] + list(run_toggles.values()),
        justify="start",
        wrap=True,
    )
    fig_3d = render_3d(
        gt["v_grid"],
        run_grids,
        evaluator,
        elev=elev_slider.value,
        azim=azim_slider.value,
        show_toggles=toggle_values,
    )
    mo.output.append(controls)
    mo.output.append(fig_3d)
    return


@app.cell
def _(evaluator, gt, plot_3d_comparison, run_grids):
    # 3D Line Wireframe Comparison
    fig_3d_lines = plot_3d_comparison(gt["v_grid"], run_grids, evaluator)
    fig_3d_lines
    return


@app.cell
def _(compute_runs_jaggedness, evaluator, gt, mo, run_grids):
    # Surface Jaggedness / Smoothness Evaluation
    jag_scores = compute_runs_jaggedness(
        run_grids, evaluator.occupied_map, v_true=gt["v_grid"]
    )
    table_rows = [
        {"Algorithm / Run": k, "Jaggedness Score (lower is smoother)": f"{v:.5f}"}
        for k, v in jag_scores.items()
    ]
    jag_table = mo.ui.table(table_rows)
    mo.output.append(mo.md("### Surface Jaggedness Comparison"))
    mo.output.append(jag_table)
    return


@app.cell
def _(evaluator, plot_feature_singular_vectors, resolved_runs):
    # Top Feature Spatial Patterns
    feature_figs = {}
    for key, rinfo in resolved_runs.items():
        if rinfo["metrics"] is not None and "feature_top_singular_vectors" in rinfo["metrics"]:
            fig_feat = plot_feature_singular_vectors(
                rinfo["metrics"],
                evaluator=evaluator,
                title_prefix=rinfo["title"],
            )
            if fig_feat is not None:
                feature_figs[key] = fig_feat
    return (feature_figs,)


@app.cell
def _(feature_figs):
    for fig in feature_figs.values():
        fig
    return


if __name__ == "__main__":
    app.run()
