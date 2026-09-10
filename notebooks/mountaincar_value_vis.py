# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "jax",
#     "jaxlib",
#     "gymnax",
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import os

    # Ensure repository root is in sys.path when running from notebooks/ or root
    _current_dir = (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals()
        else os.path.abspath(".")
    )
    _repo_root = (
        os.path.abspath(os.path.join(_current_dir, ".."))
        if os.path.basename(_current_dir) == "notebooks"
        else _current_dir
    )
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import pandas as pd
    import marimo as mo

    from envs.mountaincar_exact import MountainCarExactValue
    from core.config import config
    from notebooks.analyze_runs import (
        load_fixed_policy,
        resolve_runs,
        load_run_data_from_path,
        DEFAULT_COLORS,
    )

    return (
        DEFAULT_COLORS,
        Line2D,
        LogNorm,
        MountainCarExactValue,
        Patch,
        config,
        jax,
        jnp,
        load_fixed_policy,
        load_run_data_from_path,
        mo,
        np,
        pd,
        plt,
        resolve_runs,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Mountain Car: Learned Value Function & 3D Surface Visualization

    This notebook provides an interactive environment to visualize and compare **learned value functions** on the **Mountain Car** benchmark against the exact ground truth ($V^\pi$), reusing the 3D phase-space visualization from `MountainCar_Value.py`.

    ### Features
    - **Interactive 3D Phase-Space Surface**: Continuous Position ($x \in [-1.2, 0.6]$) and Velocity ($v \in [-0.07, 0.07]$) mapping.
    - **Algorithm Comparison**: Compare Temporal Difference (TD), Monte Carlo (MC), and Expected Updates (E) or your custom runs.
    - **Flexible Input**: Insert your own learned result dictionaries directly in `PRESETS` or point to run directories.
    - **Dual View Modes**: Side-by-side 3D surfaces, overlaid translucent true surface with wireframes, and underestimation error surfaces.
    - **Quantitative Metrics**: Uniform MSE, stationary distribution-weighted error ($\mu$-MSE), discounted visitation-weighted error ($d_\gamma$-MSE), and surface jaggedness.
    """)
    return


@app.cell
def _():
    # ---------------------------------------------------------------------------
    # Experiment Presets
    #
    # You can customize these presets or paste your own Mountain Car learned
    # result dictionaries directly under the "metrics" key in each run!
    #
    # Example format for direct metric dictionary:
    # {
    #     "title": "Temporal Difference (TD)",
    #     "color": "#1f77b4",
    #     "metrics": {
    #         "V_nn": <array of shape (num_states,) or (epochs, num_states)>,
    #         # ... other metrics
    #     }
    # }
    # ---------------------------------------------------------------------------
    PRESETS = {
        "Random Policy (TD vs MC vs E)": {
            "policy_type": "random",
            "fixed_policy_dir": None,
            "runs": {
                "TD": {
                    "run_dir": "results/random/td_exact/tuned_saved_metrics/",
                    "title": "Temporal Difference (TD)",
                    "color": "#1f77b4",
                    "metrics": None,
                },
                "MC": {
                    "run_dir": "results/random/mc_exact/tuned_saved_metrics/",
                    "title": "Monte Carlo (MC)",
                    "color": "#d62728",
                    "metrics": None,
                },
                "E": {
                    "run_dir": "results/random/exact_E/tuned_saved_metrics/",
                    "title": "Expected Update (E)",
                    "color": "#2ca02c",
                    "metrics": None,
                },
            },
        },
        "Fixed Policy (TD vs MC vs E)": {
            "policy_type": "fixed",
            "fixed_policy_dir": "ppo/ground_truth/mountaincar_fixed",
            "runs": {
                "TD": {
                    "run_dir": "results/fixed/td_exact/tuned_mountaincar/",
                    "title": "Temporal Difference (TD)",
                    "color": "#1f77b4",
                    "metrics": None,
                },
                "MC": {
                    "run_dir": "results/fixed/mc_exact/tuned_mountaincar/",
                    "title": "Monte Carlo (MC)",
                    "color": "#d62728",
                    "metrics": None,
                },
                "E": {
                    "run_dir": "results/fixed/E_gd_exact/tuned_mountaincar/",
                    "title": "Expected Update (E)",
                    "color": "#2ca02c",
                    "metrics": None,
                },
            },
        },
        "Energy-Pumping Heuristic Policy": {
            "policy_type": "heuristic",
            "fixed_policy_dir": None,
            "runs": {
                "TD": {
                    "run_dir": "results/heuristic/td_exact/",
                    "title": "Temporal Difference (TD)",
                    "color": "#1f77b4",
                    "metrics": None,
                },
                "MC": {
                    "run_dir": "results/heuristic/mc_exact/",
                    "title": "Monte Carlo (MC)",
                    "color": "#d62728",
                    "metrics": None,
                },
                "E": {
                    "run_dir": "results/heuristic/exact_E/",
                    "title": "Expected Update (E)",
                    "color": "#2ca02c",
                    "metrics": None,
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
        label="Experiment Preset",
    )

    resolution_dropdown = mo.ui.dropdown(
        options={
            "32 x 32 (1,024 states — Matches standard sweep runs)": 32,
            "50 x 50 (2,500 states — High definition swirl)": 50,
            "64 x 64 (4,096 states — Fine grid)": 64,
        },
        value="32 x 32 (1,024 states — Matches standard sweep runs)",
        label="Discretization Resolution",
    )

    gamma_slider = mo.ui.slider(
        start=0.80,
        stop=0.999,
        step=0.01,
        value=0.99,
        label="Discount Factor (γ)",
    )

    epoch_idx_input = mo.ui.number(
        start=-1000,
        stop=10000,
        step=1,
        value=-1,
        label="Epoch / Checkpoint Index (-1 = final)",
    )

    mo.hstack(
        [
            mo.vstack([preset_selector, resolution_dropdown]),
            mo.vstack([gamma_slider, epoch_idx_input]),
        ],
        justify="start",
        gap=3,
    )
    return epoch_idx_input, gamma_slider, preset_selector, resolution_dropdown


@app.cell
def _(PRESETS, epoch_idx_input, preset_selector):
    _active = PRESETS[preset_selector.value]
    policy_type = _active["policy_type"]
    fixed_policy_dir = _active.get("fixed_policy_dir", None)
    runs_spec = _active["runs"]
    epoch_idx = int(epoch_idx_input.value)
    return epoch_idx, fixed_policy_dir, policy_type, runs_spec


@app.cell
def _(
    MountainCarExactValue,
    fixed_policy_dir,
    gamma_slider,
    jnp,
    load_fixed_policy,
    np,
    policy_type,
    resolution_dropdown,
):
    # ---------------------------------------------------------------------------
    # Environment & Ground Truth Evaluation
    # ---------------------------------------------------------------------------
    n_res = resolution_dropdown.value
    gamma = gamma_slider.value

    mc_env = MountainCarExactValue(n_pos=n_res, n_vel=n_res, gamma=gamma)

    # 1. Active policy
    if policy_type == "heuristic":
        _pi_np = np.zeros((mc_env.num_states, mc_env.num_actions), dtype=np.float32)
        for _s_idx in range(mc_env.num_states):
            _pos, _vel = mc_env.coords[_s_idx]
            if _vel > 0:
                _pi_np[_s_idx, 2] = 1.0  # Accelerate Right
            elif _vel < 0:
                _pi_np[_s_idx, 0] = 1.0  # Accelerate Left
            else:
                _pi_np[_s_idx, 0 if _pos < -0.5 else 2] = 1.0
        pi_active = jnp.asarray(_pi_np)
    elif policy_type == "fixed" and fixed_policy_dir is not None:
        try:
            _policy_fn, _, _ = load_fixed_policy(
                fixed_policy_dir, env_name="MountainCar-v0"
            )
            _probs = _policy_fn(mc_env.obs_stack)
            pi_active = jnp.asarray(
                _probs.probs if hasattr(_probs, "probs") else _probs
            )
            print(f"Loaded fixed policy from {fixed_policy_dir}")
        except Exception as _e:
            print(f"Warning: Could not load fixed policy ({_e}). Defaulting to uniform random.")
            pi_active = jnp.ones((mc_env.num_states, mc_env.num_actions)) / mc_env.num_actions
    else:
        # Uniform random policy
        pi_active = jnp.ones((mc_env.num_states, mc_env.num_actions)) / mc_env.num_actions

    # Full policy including terminal state
    pi_full = jnp.vstack(
        [pi_active, jnp.ones((1, mc_env.num_actions)) / mc_env.num_actions]
    )

    # Compute Ground Truth
    V_true_vec = mc_env.compute_true_values_raw(pi_full)
    V_true_grid = mc_env.get_value_grid(V_true_vec)

    mu_vec, P_pi = mc_env.compute_stationary_distribution_raw(pi_active)
    mu_grid = mc_env.get_value_grid(mu_vec)

    d_vec = mc_env.compute_discounted_visitation_raw(pi_active)
    d_grid = mc_env.get_value_grid(d_vec)

    # Goal mask
    mc_goal_mask = (mc_env.coords[:, 0] >= mc_env.goal_position) & (
        mc_env.coords[:, 1] >= mc_env.goal_velocity
    )
    p_goal = float(jnp.sum(mu_vec[mc_goal_mask]))
    hitting_time = (1.0 / p_goal) if p_goal > 0 else float("inf")

    gt_info = {
        "V_true_vec": V_true_vec,
        "V_true_grid": V_true_grid,
        "mu_vec": mu_vec,
        "mu_grid": mu_grid,
        "d_vec": d_vec,
        "d_grid": d_grid,
        "p_goal": p_goal,
        "hitting_time": hitting_time,
    }
    return (
        P_pi,
        V_true_grid,
        V_true_vec,
        d_grid,
        d_vec,
        gt_info,
        hitting_time,
        mc_env,
        mc_goal_mask,
        mu_grid,
        mu_vec,
        p_goal,
        pi_active,
        pi_full,
    )


@app.cell
def _(jnp, np):
    # ---------------------------------------------------------------------------
    # Value Extraction Helper for Mountain Car
    # ---------------------------------------------------------------------------
    def extract_mountaincar_value_grid(val_entry, mc_env, epoch_idx=-1, combo_idx=0, seed_idx=0):
        """
        Extracts learned value function from a run entry or metrics dictionary,
        adapting to 1D vectors, 2D grids, or multi-epoch checkpoint arrays.
        """
        if val_entry is None:
            return None

        # 1. If already a 2D numpy array of the right dimensions
        if isinstance(val_entry, (np.ndarray, jnp.ndarray)):
            arr = np.asarray(val_entry)
            if arr.ndim == 2 and arr.shape == (mc_env.n_pos, mc_env.n_vel):
                return arr
            if arr.ndim == 1:
                if len(arr) == mc_env.num_total_states:
                    arr = arr[:mc_env.num_states]
                if len(arr) == mc_env.num_states:
                    return mc_env.get_value_grid(arr)

        # 2. Extract metrics dictionary
        metrics = None
        if isinstance(val_entry, dict):
            if "metrics" in val_entry and val_entry["metrics"] is not None:
                metrics = val_entry["metrics"]
            elif any(k in val_entry for k in ["V_nn", "v_pred", "V", "v", "grid", "value_grid"]):
                metrics = val_entry

        if metrics is None:
            return None

        # Check direct grid keys
        for gk in ["grid", "value_grid"]:
            if gk in metrics and metrics[gk] is not None:
                g = np.asarray(metrics[gk])
                if g.shape == (mc_env.n_pos, mc_env.n_vel):
                    return g

        # Check raw value array keys
        v_data = None
        for k in ["V_nn", "v_pred", "V", "v", "values"]:
            if k in metrics and metrics[k] is not None:
                v_data = np.asarray(metrics[k])
                break

        if v_data is None:
            return None

        # Squeeze down to (num_states,)
        if v_data.ndim == 4:  # (combos, seeds, epochs, states)
            v_vec = v_data[combo_idx, seed_idx, epoch_idx]
        elif v_data.ndim == 3:  # (seeds, epochs, states) or (combos, epochs, states)
            v_vec = v_data[seed_idx, epoch_idx]
        elif v_data.ndim == 2:
            # Could be (n_pos, n_vel) grid already or (epochs, states)
            if v_data.shape == (mc_env.n_pos, mc_env.n_vel):
                return v_data
            v_vec = v_data[epoch_idx]
        elif v_data.ndim == 1:
            v_vec = v_data
        else:
            return None

        # Trim terminal state if present
        if len(v_vec) == mc_env.num_total_states:
            v_vec = v_vec[:mc_env.num_states]

        # Reshape if matching exact state count
        if len(v_vec) == mc_env.num_states:
            return mc_env.get_value_grid(v_vec)

        # If resolution mismatch, resample/interpolate onto current grid
        try:
            n_orig = int(round(np.sqrt(len(v_vec))))
            if n_orig * n_orig == len(v_vec):
                orig_grid = v_vec.reshape((n_orig, n_orig))
                from scipy.ndimage import zoom
                zoom_factors = (mc_env.n_pos / n_orig, mc_env.n_vel / n_orig)
                return zoom(orig_grid, zoom_factors, order=1)
        except Exception:
            pass

        return None

    return (extract_mountaincar_value_grid,)


@app.cell
def _(
    DEFAULT_COLORS,
    V_true_grid,
    epoch_idx,
    extract_mountaincar_value_grid,
    mc_env,
    resolve_runs,
    runs_spec,
):
    # ---------------------------------------------------------------------------
    # Resolve and Load All Specified Runs
    # ---------------------------------------------------------------------------
    resolved_runs = resolve_runs(runs_spec, default_env_name="MountainCar-v0")

    run_grids = {}
    for _i, (_key, _rinfo) in enumerate(resolved_runs.items()):
        _grid = None
        _color = _rinfo.get("color", DEFAULT_COLORS[_i % len(DEFAULT_COLORS)])
        _title = _rinfo.get("title", _key)

        # 1. Try loading from metrics or run directory
        if _rinfo.get("metrics") is not None:
            _grid = extract_mountaincar_value_grid(
                _rinfo["metrics"], mc_env, epoch_idx=epoch_idx
            )
        elif _rinfo.get("run_dir") is not None:
            _grid = extract_mountaincar_value_grid(
                _rinfo, mc_env, epoch_idx=epoch_idx
            )

        _is_loaded = _grid is not None

        # 2. If no data exists yet, provide an informative baseline
        if _grid is None:
            # Fallback for display preview before user supplies result dictionary
            _grid = np.asarray(V_true_grid)

        run_grids[_key] = {
            "title": _title,
            "color": _color,
            "grid": np.asarray(_grid),
            "is_loaded": _is_loaded,
        }
    return resolved_runs, run_grids


@app.cell
def _(hitting_time, mo, p_goal, policy_type, run_grids):
    _loaded_count = sum(1 for _r in run_grids.values() if _r["is_loaded"])
    _total_count = len(run_grids)

    _status_msg = (
        f"**Status**: All {_loaded_count}/{_total_count} runs loaded successfully."
        if _loaded_count == _total_count
        else f"**Status**: {_loaded_count}/{_total_count} runs loaded. Unloaded runs are displaying ground truth previews until dictionaries or run directories are populated."
    )

    _card_goal = mo.stat(
        value=f"{p_goal:.4e}",
        caption=f"{p_goal * 100:.4f}% per step",
        label="Goal Stationary Prob μ(goal)",
    )

    _card_hit = mo.stat(
        value=f"~{hitting_time:,.0f}" if hitting_time < float("inf") else "∞",
        caption="Expected renewal recurrence steps",
        label="Mean Hitting Time E[T_reach]",
    )

    _card_runs = mo.stat(
        value=f"{_loaded_count} / {_total_count}",
        caption="Populate runs in PRESETS",
        label="Learned Runs Loaded",
    )

    mo.vstack(
        [
            mo.md(f"### Environment Setup ({policy_type.capitalize()} Policy)\n{_status_msg}"),
            mo.hstack([_card_goal, _card_hit, _card_runs], justify="space-between"),
        ]
    )
    return


@app.cell
def _(V_true_grid, jnp, mc_env, mu_grid, plt):
    # ---------------------------------------------------------------------------
    # 2D Heatmaps: True Value Function & Stationary Distribution
    # ---------------------------------------------------------------------------
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    _X, _Y = jnp.meshgrid(mc_env.pos_bins, mc_env.vel_bins, indexing="ij")

    # 1. Ground Truth Value Function
    _mesh1 = _ax1.pcolormesh(_X, _Y, V_true_grid, cmap="viridis", shading="auto")
    _cbar1 = _fig.colorbar(_mesh1, ax=_ax1, fraction=0.046, pad=0.04)
    _cbar1.set_label("True Value $V^\\pi(x, v)$", fontsize=11)

    _ax1.plot(-0.5, 0.0, marker="o", markersize=8, color="cyan", markeredgecolor="black", label="Start (x=-0.5, v=0)")
    _ax1.axvline(x=mc_env.goal_position, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Goal Pos (x ≥ 0.5)")
    _ax1.axhline(y=mc_env.goal_velocity, color="orange", linestyle=":", linewidth=1.5, alpha=0.8, label="Goal Vel (v ≥ 0)")
    _ax1.fill_between(
        [mc_env.goal_position, mc_env.max_position],
        [0, 0],
        [mc_env.max_speed, mc_env.max_speed],
        color="red",
        alpha=0.15,
    )
    _ax1.set_title("Ground Truth Value Function $V^\\pi(x, v)$", fontsize=12)
    _ax1.set_xlabel("Position (x)", fontsize=11)
    _ax1.set_ylabel("Velocity (v)", fontsize=11)
    _ax1.legend(loc="lower left", fontsize=8.5)

    # 2. Stationary Distribution
    _mesh2 = _ax2.pcolormesh(_X, _Y, mu_grid, cmap="magma", shading="auto")
    _cbar2 = _fig.colorbar(_mesh2, ax=_ax2, fraction=0.046, pad=0.04)
    _cbar2.set_label("Stationary Density $\\mu(x, v)$", fontsize=11)

    _ax2.plot(-0.5, 0.0, marker="o", markersize=8, color="cyan", markeredgecolor="black")
    _ax2.axvline(x=mc_env.goal_position, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    _ax2.axhline(y=mc_env.goal_velocity, color="orange", linestyle=":", linewidth=1.5, alpha=0.8)
    _ax2.fill_between(
        [mc_env.goal_position, mc_env.max_position],
        [0, 0],
        [mc_env.max_speed, mc_env.max_speed],
        color="red",
        alpha=0.15,
    )
    _ax2.set_title("Stationary Distribution $\\mu(x, v)$", fontsize=12)
    _ax2.set_xlabel("Position (x)", fontsize=11)
    _ax2.set_ylabel("Velocity (v)", fontsize=11)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(V_true_grid, jnp, mc_env, plt, run_grids):
    # ---------------------------------------------------------------------------
    # 2D Side-by-Side Comparison of Learned Values & Errors
    # ---------------------------------------------------------------------------
    _runs = [(_k, _info) for _k, _info in run_grids.items()]
    _n = len(_runs)

    if _n > 0:
        _fig_vals, _axes_vals = plt.subplots(1, _n, figsize=(5.2 * _n, 4.5), squeeze=False)
        _X, _Y = jnp.meshgrid(mc_env.pos_bins, mc_env.vel_bins, indexing="ij")

        # Determine common color limits for learned value grids
        _all_grids = [_info["grid"] for _, _info in _runs]
        _vmin = min(float(np.min(g)) for g in _all_grids)
        _vmax = max(float(np.max(g)) for g in _all_grids)
        if _vmin == _vmax:
            _vmax = _vmin + 1e-4

        for _i, (_key, _info) in enumerate(_runs):
            _ax = _axes_vals[0, _i]
            _mesh = _ax.pcolormesh(
                _X,
                _Y,
                _info["grid"],
                cmap="viridis",
                vmin=_vmin,
                vmax=_vmax,
                shading="auto",
            )
            _fig_vals.colorbar(_mesh, ax=_ax, fraction=0.046, pad=0.04)

            _status_tag = "" if _info["is_loaded"] else " [Preview]"
            _ax.set_title(f"{_info['title']}{_status_tag}", fontsize=11)
            _ax.set_xlabel("Position (x)", fontsize=10)
            if _i == 0:
                _ax.set_ylabel("Velocity (v)", fontsize=10)

            # Start and Goal markers
            _ax.plot(-0.5, 0.0, marker="o", markersize=6, color="cyan", markeredgecolor="black")
            _ax.axvline(x=mc_env.goal_position, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

        plt.tight_layout()
        _fig_vals
    return


@app.cell
def _(V_true_grid, jnp, mc_env, plt, run_grids):
    # ---------------------------------------------------------------------------
    # 2D Absolute Error Maps (|V_true - V_learned|)
    # ---------------------------------------------------------------------------
    _runs = [(_k, _info) for _k, _info in run_grids.items()]
    _n = len(_runs)

    if _n > 0:
        _fig_errs, _axes_errs = plt.subplots(1, _n, figsize=(5.2 * _n, 4.5), squeeze=False)
        _X, _Y = jnp.meshgrid(mc_env.pos_bins, mc_env.vel_bins, indexing="ij")

        _errors = [np.abs(V_true_grid - _info["grid"]) for _, _info in _runs]
        _vmax_err = max(float(np.max(e)) for e in _errors)
        if _vmax_err == 0:
            _vmax_err = 1e-4

        for _i, (_key, _info) in enumerate(_runs):
            _ax = _axes_errs[0, _i]
            _mesh = _ax.pcolormesh(
                _X,
                _Y,
                _errors[_i],
                cmap="inferno",
                vmin=0.0,
                vmax=_vmax_err,
                shading="auto",
            )
            _fig_errs.colorbar(_mesh, ax=_ax, fraction=0.046, pad=0.04)

            _ax.set_title(f"Abs Error: {_info['title']}", fontsize=11)
            _ax.set_xlabel("Position (x)", fontsize=10)
            if _i == 0:
                _ax.set_ylabel("Velocity (v)", fontsize=10)

        plt.tight_layout()
        _fig_errs
    return


@app.cell
def _(mo, runs_spec):
    # ---------------------------------------------------------------------------
    # 3D Visualization Controls
    # ---------------------------------------------------------------------------
    elev_slider = mo.ui.slider(
        0, 90, step=5, value=35, label="3D Elevation (Pitch)"
    )
    azim_slider = mo.ui.slider(
        -180, 180, step=5, value=-55, label="3D Azimuth (Yaw)"
    )

    surface_view_mode = mo.ui.dropdown(
        options=[
            "2-Panel: Overlaid True & Wireframe (Left) + Error Surface (Right)",
            "Side-by-Side: True Surface vs Learned Surfaces",
            "Single 3D Plot: True Surface + Learned Wireframes",
            "Learned Surfaces Only",
            "Error Surfaces (V_true - V_learned)",
        ],
        value="2-Panel: Overlaid True & Wireframe (Left) + Error Surface (Right)",
        label="3D Layout View",
    )

    v_cmap_dropdown = mo.ui.dropdown(
        options=["viridis", "plasma", "inferno", "coolwarm", "cividis", "turbo"],
        value="viridis",
        label="Surface Colormap",
    )

    # Dynamic run toggles
    run_toggles = {
        _k: mo.ui.checkbox(
            value=True,
            label=f"Show {runs_spec[_k].get('title', _k)}",
        )
        for _k in runs_spec
    }

    mo.vstack(
        [
            mo.md("## 3D Interactive Phase-Space Visualization"),
            mo.hstack([elev_slider, azim_slider, surface_view_mode, v_cmap_dropdown], justify="start", wrap=True),
            mo.hstack(list(run_toggles.values()), justify="start", wrap=True),
        ]
    )
    return (
        azim_slider,
        elev_slider,
        run_toggles,
        surface_view_mode,
        v_cmap_dropdown,
    )


@app.cell
def _(
    Line2D,
    Patch,
    V_true_grid,
    azim_slider,
    elev_slider,
    jnp,
    mc_env,
    np,
    plt,
    run_grids,
    run_toggles,
    surface_view_mode,
    v_cmap_dropdown,
):
    # ---------------------------------------------------------------------------
    # 3D Plot Rendering
    # ---------------------------------------------------------------------------
    _elev = elev_slider.value
    _azim = azim_slider.value
    _mode = surface_view_mode.value
    _cmap = v_cmap_dropdown.value

    # Coordinates in Mountain Car continuous phase space
    _X, _Y = jnp.meshgrid(mc_env.pos_bins, mc_env.vel_bins, indexing="ij")
    _X_np = np.asarray(_X)
    _Y_np = np.asarray(_Y)
    _V_true_np = np.asarray(V_true_grid)

    # Filter active runs by user checkboxes
    _active_runs = [
        (_k, _info)
        for _k, _info in run_grids.items()
        if run_toggles[_k].value and _info["grid"] is not None
    ]

    # Stride for 3D wireframe downsampling if grid is high-resolution
    _rstride = max(1, mc_env.n_pos // 25)
    _cstride = max(1, mc_env.n_vel // 25)

    if _mode == "2-Panel: Overlaid True & Wireframe (Left) + Error Surface (Right)":
        _fig = plt.figure(figsize=(16, 7))
        _ax1 = _fig.add_subplot(1, 2, 1, projection="3d")
        _ax2 = _fig.add_subplot(1, 2, 2, projection="3d")

        # Left Panel: True Value Surface (translucent) + Wireframes
        _surf_true = _ax1.plot_surface(
            _X_np,
            _Y_np,
            _V_true_np,
            cmap=_cmap,
            alpha=0.45,
            edgecolor="none",
            antialiased=True,
            rstride=1,
            cstride=1,
        )
        _fig.colorbar(_surf_true, ax=_ax1, shrink=0.5, aspect=14, pad=0.08, label="True Value $V^\\pi$")

        _handles1 = [Patch(facecolor="teal", edgecolor="black", alpha=0.45, label="Ground Truth Surface ($V^\\pi$)")]

        for _k, _info in _active_runs:
            _G = np.asarray(_info["grid"])
            _c = _info["color"]
            _lbl = _info["title"]

            # Plot wireframe lines
            _ax1.plot_wireframe(
                _X_np,
                _Y_np,
                _G,
                color=_c,
                alpha=0.75,
                linewidth=1.2,
                rstride=_rstride,
                cstride=_cstride,
            )
            _handles1.append(Line2D([0], [0], color=_c, lw=2, label=f"{_lbl} (Wireframe)"))

            # Right Panel: Error Surface (V_true - V_learned)
            _err = _V_true_np - _G
            _ax2.plot_surface(
                _X_np,
                _Y_np,
                _err,
                color=_c,
                alpha=0.45,
                edgecolor="none",
                antialiased=True,
                rstride=_rstride,
                cstride=_cstride,
            )

        _ax1.view_init(elev=_elev, azim=_azim)
        _ax1.set_xlabel("Position (x)", labelpad=8)
        _ax1.set_ylabel("Velocity (v)", labelpad=8)
        _ax1.set_zlabel("Value $V(x, v)$", labelpad=8)
        _ax1.set_title("True Surface $V^\\pi(x, v)$ with Learned Wireframes", fontsize=12)
        _ax1.legend(handles=_handles1, loc="upper left", fontsize=8.5)

        _ax2.view_init(elev=_elev, azim=_azim)
        _ax2.set_xlabel("Position (x)", labelpad=8)
        _ax2.set_ylabel("Velocity (v)", labelpad=8)
        _ax2.set_zlabel("Error ($V_{true} - V_{learned}$)", labelpad=8)
        _ax2.set_title("Underestimation Error Surface ($V_{true} - V_{learned}$)", fontsize=12)

    elif _mode == "Side-by-Side: True Surface vs Learned Surfaces":
        _num_plots = 1 + len(_active_runs)
        _fig = plt.figure(figsize=(5.5 * _num_plots, 6))

        # 1. Ground Truth
        _ax_gt = _fig.add_subplot(1, _num_plots, 1, projection="3d")
        _surf_gt = _ax_gt.plot_surface(
            _X_np,
            _Y_np,
            _V_true_np,
            cmap=_cmap,
            edgecolor="none",
            antialiased=True,
            alpha=0.9,
        )
        _ax_gt.view_init(elev=_elev, azim=_azim)
        _ax_gt.set_xlabel("Position (x)", labelpad=8)
        _ax_gt.set_ylabel("Velocity (v)", labelpad=8)
        _ax_gt.set_zlabel("Value", labelpad=8)
        _ax_gt.set_title("Ground Truth $V^\\pi(x, v)$", fontsize=11)
        _fig.colorbar(_surf_gt, ax=_ax_gt, shrink=0.5, aspect=12, pad=0.08)

        # 2. Learned Runs
        for _idx, (_k, _info) in enumerate(_active_runs):
            _ax = _fig.add_subplot(1, _num_plots, 2 + _idx, projection="3d")
            _G = np.asarray(_info["grid"])
            _surf = _ax.plot_surface(
                _X_np,
                _Y_np,
                _G,
                cmap=_cmap,
                edgecolor="none",
                antialiased=True,
                alpha=0.9,
            )
            _ax.view_init(elev=_elev, azim=_azim)
            _ax.set_xlabel("Position (x)", labelpad=8)
            _ax.set_ylabel("Velocity (v)", labelpad=8)
            _ax.set_zlabel("Value", labelpad=8)
            _ax.set_title(_info["title"], fontsize=11)
            _fig.colorbar(_surf, ax=_ax, shrink=0.5, aspect=12, pad=0.08)

    elif _mode == "Single 3D Plot: True Surface + Learned Wireframes":
        _fig = plt.figure(figsize=(10, 7))
        _ax = _fig.add_subplot(1, 1, 1, projection="3d")

        _surf = _ax.plot_surface(
            _X_np,
            _Y_np,
            _V_true_np,
            cmap=_cmap,
            alpha=0.45,
            edgecolor="none",
            antialiased=True,
        )
        _fig.colorbar(_surf, ax=_ax, shrink=0.6, aspect=15, pad=0.08, label="True Value $V^\\pi$")

        _handles = [Patch(facecolor="teal", edgecolor="black", alpha=0.45, label="Ground Truth Surface ($V^\\pi$)")]

        for _k, _info in _active_runs:
            _G = np.asarray(_info["grid"])
            _c = _info["color"]
            _lbl = _info["title"]
            _ax.plot_wireframe(
                _X_np,
                _Y_np,
                _G,
                color=_c,
                alpha=0.75,
                linewidth=1.2,
                rstride=_rstride,
                cstride=_cstride,
            )
            _handles.append(Line2D([0], [0], color=_c, lw=2, label=_lbl))

        _ax.view_init(elev=_elev, azim=_azim)
        _ax.set_xlabel("Position (x)", labelpad=8)
        _ax.set_ylabel("Velocity (v)", labelpad=8)
        _ax.set_zlabel("Value $V(x, v)$", labelpad=8)
        _ax.set_title("Mountain Car Phase-Space Value Surface ($V^\\pi$ & Learned Wireframes)", fontsize=12)
        _ax.legend(handles=_handles, loc="upper left")

    elif _mode == "Learned Surfaces Only":
        _fig = plt.figure(figsize=(10, 7))
        _ax = _fig.add_subplot(1, 1, 1, projection="3d")
        _handles = []

        for _k, _info in _active_runs:
            _G = np.asarray(_info["grid"])
            _c = _info["color"]
            _lbl = _info["title"]
            _ax.plot_surface(
                _X_np,
                _Y_np,
                _G,
                color=_c,
                alpha=0.5,
                edgecolor="none",
                antialiased=True,
                rstride=_rstride,
                cstride=_cstride,
            )
            _handles.append(Patch(facecolor=_c, edgecolor="none", alpha=0.5, label=_lbl))

        _ax.view_init(elev=_elev, azim=_azim)
        _ax.set_xlabel("Position (x)", labelpad=8)
        _ax.set_ylabel("Velocity (v)", labelpad=8)
        _ax.set_zlabel("Value", labelpad=8)
        _ax.set_title("Learned Value Surfaces Comparison", fontsize=12)
        _ax.legend(handles=_handles, loc="upper left")

    else:  # Error Surfaces (V_true - V_learned)
        _fig = plt.figure(figsize=(10, 7))
        _ax = _fig.add_subplot(1, 1, 1, projection="3d")
        _handles = []

        for _k, _info in _active_runs:
            _G = np.asarray(_info["grid"])
            _c = _info["color"]
            _lbl = _info["title"]
            _err = _V_true_np - _G
            _ax.plot_surface(
                _X_np,
                _Y_np,
                _err,
                color=_c,
                alpha=0.5,
                edgecolor="none",
                antialiased=True,
                rstride=_rstride,
                cstride=_cstride,
            )
            _handles.append(Patch(facecolor=_c, edgecolor="none", alpha=0.5, label=f"Error: {_lbl}"))

        _ax.view_init(elev=_elev, azim=_azim)
        _ax.set_xlabel("Position (x)", labelpad=8)
        _ax.set_ylabel("Velocity (v)", labelpad=8)
        _ax.set_zlabel("Error ($V_{true} - V_{learned}$)", labelpad=8)
        _ax.set_title("3D Underestimation Error Surfaces ($V_{true} - V_{learned}$)", fontsize=12)
        _ax.legend(handles=_handles, loc="upper left")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(V_true_grid, d_vec, mc_env, mo, mu_vec, np, pd, run_grids):
    # ---------------------------------------------------------------------------
    # Quantitative Value Error & Smoothness (Jaggedness) Metrics
    # ---------------------------------------------------------------------------
    def _calc_jaggedness(grid):
        _diff_pos = np.abs(grid[1:, :] - grid[:-1, :])
        _diff_vel = np.abs(grid[:, 1:] - grid[:, :-1])
        return float(np.mean(_diff_pos) + np.mean(_diff_vel))

    _metrics_rows = []
    _v_true_flat = np.asarray(V_true_grid).flatten()
    _mu_flat = np.asarray(mu_vec).flatten()
    _d_flat = np.asarray(d_vec).flatten()

    # Ground Truth Baseline
    _metrics_rows.append({
        "Algorithm / Model": "Ground Truth (V^π)",
        "Uniform MSE": "0.00000",
        "μ-Weighted MSE": "0.00000",
        "d_γ-Weighted MSE": "0.00000",
        "Max Abs Error": "0.00000",
        "Surface Jaggedness (Total Variation)": f"{_calc_jaggedness(np.asarray(V_true_grid)):.5f}",
        "Status": "Exact",
    })

    for _k, _info in run_grids.items():
        if _info["grid"] is None:
            continue
        _g = np.asarray(_info["grid"])
        _g_flat = _g.flatten()

        _mse_u = float(np.mean((_v_true_flat - _g_flat) ** 2))
        _mse_mu = float(np.sum(_mu_flat * ((_v_true_flat - _g_flat) ** 2)))
        _mse_d = float(np.sum(_d_flat * ((_v_true_flat - _g_flat) ** 2)))
        _max_err = float(np.max(np.abs(_v_true_flat - _g_flat)))
        _jag = _calc_jaggedness(_g)

        _metrics_rows.append({
            "Algorithm / Model": _info["title"],
            "Uniform MSE": f"{_mse_u:.5f}",
            "μ-Weighted MSE": f"{_mse_mu:.5f}",
            "d_γ-Weighted MSE": f"{_mse_d:.5f}",
            "Max Abs Error": f"{_max_err:.5f}",
            "Surface Jaggedness (Total Variation)": f"{_jag:.5f}",
            "Status": "Loaded" if _info["is_loaded"] else "Preview",
        })

    _df = pd.DataFrame(_metrics_rows)
    _table_ui = mo.ui.table(_df, selection=None)

    mo.vstack(
        [
            mo.md("### Quantitative Error & Surface Smoothness Comparison"),
            _table_ui,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
