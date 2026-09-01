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

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    import os

    # Ensure repository root is in sys.path when running from notebooks/ directory
    _current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(".")
    _repo_root = os.path.abspath(os.path.join(_current_dir, "..")) if os.path.basename(_current_dir) == "notebooks" else _current_dir
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

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
        compute_runs_jaggedness,
        plot_feature_singular_vectors,
        plot_jacobian_singular_vectors,
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
        plot_jacobian_singular_vectors,
        plot_multi_grids,
        render_3d,
        resolve_runs,
    )


@app.cell
def _():
    # ---------------------------------------------------------------------------
    # Experiment Presets
    # ---------------------------------------------------------------------------
    PRESETS = {
        "Random Policy (TD vs MC vs E)": {
            "policy_type": "random",
            "fixed_policy_dir": "ppo/ground_truth/partial_four_rooms",
            "runs": {
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
            },
        },
        "Fixed Policy (TD vs MC vs E)": {
            "policy_type": "fixed",
            "fixed_policy_dir": "ppo/ground_truth/partial_four_rooms",
            "runs": {
                "TD": {
                    "run_dir": "fixed/td_exact/tuned_four_rooms_partial/",
                    "title": "Temporal Difference (TD)",
                    "color": "blue",
                },
                "MC": {
                    "run_dir": "fixed/mc_exact/tuned_four_rooms_partial/",
                    "title": "Monte Carlo (MC)",
                    "color": "red",
                },
                "E": {
                    "run_dir": "fixed/E_gd_exact/tuned_four_rooms_partial/",
                    "title": "Expected Update (E)",
                    "color": "green",
                },
            },
        },
    }
    return (PRESETS,)


@app.cell
def _(PRESETS, mo):
    preset_selector = mo.ui.dropdown(
        options=list(PRESETS.keys()),
        value="Random Policy (TD vs MC vs E)",
        label="Select Experiment Preset",
    )
    preset_selector
    return (preset_selector,)


@app.cell
def _(PRESETS, preset_selector):
    _active_preset = PRESETS[preset_selector.value]
    policy_type = _active_preset["policy_type"]
    fixed_policy_dir = _active_preset["fixed_policy_dir"]
    runs_spec = _active_preset["runs"]
    epoch_idx = 500  # Final checkpoint index
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
    _env, _env_params = make_env(config)
    evaluator = initialize_evaluator(config, _env, _env_params)

    # Optional Fixed Policy Loading
    _fixed_policy_fn = None
    if policy_type == "fixed":
        try:
            _fixed_policy_fn, _, _ = load_fixed_policy(
                fixed_policy_dir, env_name=config.get("ENV_NAME", "FourRooms-misc")
            )
            print(f"Successfully loaded fixed policy from {fixed_policy_dir}")
        except Exception as _e:
            print(f"Warning: Could not load fixed policy from '{fixed_policy_dir}': {_e}")

    # Compute Ground Truth (V^pi, mu, grids)
    gt = compute_ground_truth(
        evaluator, policy_type=policy_type, fixed_policy_fn=_fixed_policy_fn
    )
    return evaluator, gt


@app.cell
def _(epoch_idx, evaluator, get_run_value_grid, gt, resolve_runs, runs_spec):
    # Resolve all specified runs (load data where available or fallback smoothly)
    resolved_runs = resolve_runs(runs_spec)

    run_grids = {}
    for _key, _rinfo in resolved_runs.items():
        _grid = None
        if _rinfo["metrics"] is not None:
            try:
                _grid = get_run_value_grid(_rinfo, evaluator, epoch_idx=epoch_idx)
            except Exception as _e:
                print(f"Could not extract value grid for {_key}: {_e}")
                _grid = None

        # Fallback to ground truth if data is not available locally for demo preview
        run_grids[_key] = {
            "title": _rinfo["title"],
            "color": _rinfo["color"],
            "grid": _grid if _grid is not None else gt["v_grid"],
            "is_loaded": _grid is not None,
        }
    return resolved_runs, run_grids


@app.cell
def _(evaluator, gt, plot_multi_grids):
    # Ground Truth: True Value Function and Stationary Distribution
    _fig_gt = plot_multi_grids(
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
    _fig_gt
    return


@app.cell
def _(evaluator, plot_multi_grids, run_grids):
    # Side-by-side comparison of learned value functions
    _display_grids = {info["title"]: info["grid"] for info in run_grids.values()}
    _fig_values = plot_multi_grids(
        _display_grids,
        evaluator,
        cmap="viridis",
        shared_vrange=True,
    )
    _fig_values
    return


@app.cell
def _(evaluator, gt, plot_error_grids, run_grids):
    # Side-by-side comparison of value errors (MSE)
    _error_input = {_k: _info["grid"] for _k, _info in run_grids.items()}
    _fig_errors = plot_error_grids(
        _error_input,
        gt["v_grid"],
        evaluator,
        metric_type="squared_error",
        use_log=True,
        shared_vrange=True,
    )
    _fig_errors
    return


@app.cell
def _(mo):
    mo.md("""
    # Interactive 3D Analysis
    Use the sliders below to rotate the 3D view and checkboxes to toggle algorithms.
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
        _k: mo.ui.checkbox(
            value=True, label=f"Show {runs_spec[_k].get('title', _k)}"
        )
        for _k in runs_spec
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
    _toggle_values = {_k: _t.value for _k, _t in run_toggles.items()}

    _controls = mo.hstack(
        [elev_slider, azim_slider] + list(run_toggles.values()),
        justify="start",
        wrap=True,
    )
    _fig_3d = render_3d(
        gt["v_grid"],
        run_grids,
        evaluator,
        elev=elev_slider.value,
        azim=azim_slider.value,
        show_toggles=_toggle_values,
    )
    mo.output.append(_controls)
    mo.output.append(_fig_3d)
    return


@app.cell
def _(evaluator, gt, plot_3d_comparison, run_grids):
    # 3D Line Wireframe Comparison
    _fig_3d_lines = plot_3d_comparison(gt["v_grid"], run_grids, evaluator)
    _fig_3d_lines
    return


@app.cell
def _(compute_runs_jaggedness, evaluator, gt, mo, run_grids):
    # Surface Jaggedness / Smoothness Evaluation
    _jag_scores = compute_runs_jaggedness(
        run_grids, evaluator.occupied_map, v_true=gt["v_grid"]
    )
    _table_rows = [
        {"Algorithm / Run": _k, "Jaggedness Score (lower is smoother)": f"{_v:.5f}"}
        for _k, _v in _jag_scores.items()
    ]
    _jag_table = mo.ui.table(_table_rows)
    mo.output.append(mo.md("### Surface Jaggedness Comparison"))
    mo.output.append(_jag_table)
    return


@app.cell
def _(evaluator, plot_feature_singular_vectors, resolved_runs):
    # Top Feature Spatial Patterns
    _feature_figs = {}
    for _key, _rinfo in resolved_runs.items():
        if _rinfo["metrics"] is not None and "feature_top_singular_vectors" in _rinfo["metrics"]:
            _fig_feat = plot_feature_singular_vectors(
                _rinfo["metrics"],
                evaluator=evaluator,
                title_prefix=_rinfo["title"],
            )
            if _fig_feat is not None:
                _feature_figs[_key] = _fig_feat

    for _fig in _feature_figs.values():
        _fig
    return


@app.cell
def _(evaluator, plot_jacobian_singular_vectors, resolved_runs):
    # Top Jacobian Singular Vectors
    _jacobian_figs = {}
    for _key, _rinfo in resolved_runs.items():
        if _rinfo["metrics"] is not None and any(
            _k in _rinfo["metrics"]
            for _k in ["Jacobian_top_singular_vectors", "jacobian_top_singular_vectors"]
        ):
            _fig_jac = plot_jacobian_singular_vectors(
                _rinfo["metrics"],
                evaluator=evaluator,
                title_prefix=_rinfo["title"],
            )
            if _fig_jac is not None:
                _jacobian_figs[_key] = _fig_jac

    for _fig in _jacobian_figs.values():
        _fig
    return


if __name__ == "__main__":
    app.run()
