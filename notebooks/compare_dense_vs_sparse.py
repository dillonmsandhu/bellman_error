# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "jax",
#     "jaxlib",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "cloudpickle",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys

    # Ensure repository root is on sys.path
    _current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(".")
    _repo_root = os.path.abspath(os.path.join(_current_dir, "..")) if os.path.basename(_current_dir) == "notebooks" else _current_dir
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import colors
    import pandas as pd
    import marimo as mo

    from core.utils import load_run_data_from_path

    return animation, colors, load_run_data_from_path, mo, np, os, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Dense (Shaped) vs Sparse Reward: Algorithm Comparison
    ### Evaluating **TD(λ)**, **E(λ)**, **E**, and **MC** on EightRooms vs EightRooms-Dense

    This interactive notebook compares policy and value learning dynamics between:
    1. **Sparse Reward**: `EightRooms` (Goal reward only upon reaching terminal state).
    2. **Dense Shaped Reward**: `EightRooms-Dense` (Potential-Based Reward Shaping via geodesic distance).

    Key Questions:
    - How does reward density affect Bellman error ($E$) minimization versus temporal difference ($TD$) learning?
    - Does dense potential shaping accelerate value propagation without introducing bias?
    - How do the value function grids, state distributions, and signed value errors evolve over time?
    """)
    return


@app.cell
def _(mo, os):
    # Discover available suffixes under results/ppo/exact_td_lambda or results/ppo/exact_E
    _available_suffixes = []
    _base_check_dirs = [
        "results/ppo/exact_td_lambda",
        "../results/ppo/exact_td_lambda",
        "results/ppo/exact_E",
        "../results/ppo/exact_E",
    ]
    for _bdir in _base_check_dirs:
        if os.path.exists(_bdir):
            for _item in os.listdir(_bdir):
                if os.path.isdir(os.path.join(_bdir, _item)) and _item not in _available_suffixes:
                    _available_suffixes.append(_item)

    _default_suffix = (
        "dense_vs_sparse"
        if "dense_vs_sparse" in _available_suffixes
        else (_available_suffixes[0] if _available_suffixes else "dense_vs_sparse")
    )

    suffix_input = mo.ui.text(
        value=_default_suffix,
        label="Run Suffix",
        placeholder="e.g. dense_vs_sparse or test_dense_smoke",
    )

    max_frames_slider = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=50,
        label="Max GIF Frames (Subsampling)",
    )

    fps_slider = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=5,
        label="GIF FPS",
    )

    seed_idx_input = mo.ui.number(
        start=0,
        stop=10,
        step=1,
        value=0,
        label="Seed Index",
    )

    controls_panel = mo.hstack(
        [suffix_input, max_frames_slider, fps_slider, seed_idx_input],
        justify="start",
        gap=2,
    )
    return (
        controls_panel,
        fps_slider,
        max_frames_slider,
        seed_idx_input,
        suffix_input,
    )


@app.cell
def _(controls_panel, mo):
    mo.vstack([
        mo.md("### ⚙️ Experiment Configuration"),
        controls_panel
    ])
    return


@app.cell
def _(load_run_data_from_path, os, suffix_input):
    current_suffix = suffix_input.value.strip()

    # Algorithm specifications
    ALGOS = {
        "TD_lambda": {"name": "TD(λ)", "save_dir": "ppo/exact_td_lambda", "color": "#1f77b4"},
        "E_lambda": {"name": "E(λ)", "save_dir": "ppo/exact_E_lambda", "color": "#ff7f0e"},
        "E": {"name": "Exact E", "save_dir": "ppo/exact_E", "color": "#2ca02c"},
        "MC": {"name": "MC", "save_dir": "ppo/exact_mc", "color": "#d62728"},
    }

    ENVS = {
        "sparse": {"id": "EightRooms", "label": "EightRooms (Sparse)"},
        "dense": {"id": "EightRooms-Dense", "label": "EightRooms-Dense (Shaped)"},
    }

    runs_data = {}
    diagnostic_rows = []

    for _algo_k, _algo_v in ALGOS.items():
        runs_data[_algo_k] = {}
        for _env_k, _env_v in ENVS.items():
            _env_name = _env_v["id"]
            # Candidates for path
            _candidates = [
                os.path.join("results", _algo_v["save_dir"], current_suffix, _env_name),
                os.path.join("..", "results", _algo_v["save_dir"], current_suffix, _env_name),
                os.path.join(_algo_v["save_dir"], current_suffix, _env_name),
            ]
            _loaded = False
            for _cand in _candidates:
                if os.path.exists(os.path.join(_cand, "config.json")) and os.path.exists(os.path.join(_cand, "out.pkl")):
                    try:
                        _cfg, _raw_out = load_run_data_from_path(_cand)
                        # Extract metrics dict cleanly
                        if isinstance(_raw_out, dict) and "metrics" in _raw_out:
                            _m = _raw_out["metrics"]
                        else:
                            _m = _raw_out
                        runs_data[_algo_k][_env_k] = {
                            "config": _cfg,
                            "metrics": _m,
                            "path": _cand,
                            "algo_name": _algo_v["name"],
                            "env_label": _env_v["label"],
                            "color": _algo_v["color"],
                        }
                        _loaded = True
                        _has_grid = "V_grid" in _m or "value_grid" in _m
                        _timesteps = len(_m.get("V_start", [])) if "V_start" in _m else "N/A"
                        diagnostic_rows.append({
                            "Algorithm": _algo_v["name"],
                            "Environment": _env_v["label"],
                            "Status": "✅ Loaded",
                            "Updates": _timesteps,
                            "Grid Visualizations": "✅ Available" if _has_grid else "❌ Missing (Light Metrics)",
                            "Path": _cand,
                        })
                        break
                    except Exception as _e:
                        diagnostic_rows.append({
                            "Algorithm": _algo_v["name"],
                            "Environment": _env_v["label"],
                            "Status": f"⚠️ Error: {str(_e)[:30]}",
                            "Updates": 0,
                            "Grid Visualizations": "❌",
                            "Path": _cand,
                        })
            if not _loaded:
                runs_data[_algo_k][_env_k] = None
                diagnostic_rows.append({
                    "Algorithm": _algo_v["name"],
                    "Environment": _env_v["label"],
                    "Status": "❌ Not Found",
                    "Updates": 0,
                    "Grid Visualizations": "❌",
                    "Path": _candidates[0],
                })
    return ALGOS, current_suffix, diagnostic_rows, runs_data


@app.cell
def _(diagnostic_rows, mo, pd):
    _df_diag = pd.DataFrame(diagnostic_rows)
    mo.vstack([
        mo.md("### 📊 Loaded Runs Status"),
        mo.ui.table(_df_diag)
    ])
    return


@app.cell
def _(mo, runs_data):
    # Detect all available 1D metrics across loaded runs
    _available_metrics = set()
    for _algo_dict in runs_data.values():
        for _env_run in _algo_dict.values():
            if _env_run and "metrics" in _env_run:
                for _k, _v in _env_run["metrics"].items():
                    if hasattr(_v, "ndim") and _v.ndim in (1, 2) and _k not in ("V_grid", "value_grid", "nn_grid", "state_dist_grid", "stat_dist"):
                        _available_metrics.add(_k)

    _preferred_order = [
        "V_start",
        "v_pred_start",
        "value_loss",
        "actor_loss",
        "entropy",
        "Mean_A",
        "nn_greedy_performance",
        "policy_tv",
        "state_coverage",
        "state_entropy_coverage",
        "SA_min_eigenvalue",
    ]
    _sorted_metrics = [m for m in _preferred_order if m in _available_metrics]
    _sorted_metrics += sorted(list(_available_metrics - set(_sorted_metrics)))

    metric_dropdown = mo.ui.dropdown(
        options=_sorted_metrics if _sorted_metrics else ["V_start"],
        value="V_start" if "V_start" in _sorted_metrics else (_sorted_metrics[0] if _sorted_metrics else "V_start"),
        label="Select Comparison Metric",
    )

    log_scale_checkbox = mo.ui.checkbox(
        value=False,
        label="Log Scale Y-Axis",
    )
    return log_scale_checkbox, metric_dropdown


@app.cell
def _(log_scale_checkbox, metric_dropdown, mo):
    mo.hstack([metric_dropdown, log_scale_checkbox], justify="start", gap=3)
    return


@app.cell
def _(ALGOS, log_scale_checkbox, metric_dropdown, np, plt, runs_data):
    _selected_metric = metric_dropdown.value
    _use_log = log_scale_checkbox.value

    # Helper function to extract 1D series
    def _extract_series(metric_array):
        _arr = np.asarray(metric_array)
        if _arr.ndim == 2:
            return np.mean(_arr, axis=0)  # average over seeds
        elif _arr.ndim == 1:
            return _arr
        return None

    # Plot 1: Sparse vs Dense Comparison (2x2 grid, one for each algo)
    fig_algo_comp, _axes_algo = plt.subplots(2, 2, figsize=(14, 10))
    _axes_algo = _axes_algo.flatten()

    for _i, (_algo_key, _algo_info) in enumerate(ALGOS.items()):
        _ax = _axes_algo[_i]
        _run_sparse = runs_data[_algo_key].get("sparse")
        _run_dense = runs_data[_algo_key].get("dense")

        _plotted = False
        if _run_sparse and _selected_metric in _run_sparse["metrics"]:
            _y_sp = _extract_series(_run_sparse["metrics"][_selected_metric])
            if _y_sp is not None:
                _ax.plot(_y_sp, label="Sparse (EightRooms)", color="#1f77b4", linewidth=2.2, linestyle="-")
                _plotted = True

        if _run_dense and _selected_metric in _run_dense["metrics"]:
            _y_dn = _extract_series(_run_dense["metrics"][_selected_metric])
            if _y_dn is not None:
                _ax.plot(_y_dn, label="Dense Shaped (EightRooms-Dense)", color="#2ca02c", linewidth=2.2, linestyle="--")
                _plotted = True

        _ax.set_title(f"{_algo_info['name']}: Sparse vs Dense", fontsize=13, fontweight="bold")
        _ax.set_xlabel("Update Step", fontsize=11)
        _ax.set_ylabel(_selected_metric, fontsize=11)
        if _use_log:
            _ax.set_yscale("log")
        _ax.grid(True, linestyle=":", alpha=0.6)
        if _plotted:
            _ax.legend(frameon=True, fontsize=10)
        else:
            _ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=_ax.transAxes, alpha=0.5)

    fig_algo_comp.suptitle(f"{_selected_metric} — Sparse vs Shaped Reward for Each Algorithm", fontsize=16, fontweight="bold", y=0.99)
    fig_algo_comp.tight_layout()

    # Plot 2: Environment Benchmark (Sparse on Left, Dense on Right)
    fig_env_comp, (_ax_sp, _ax_dn) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Sparse panel
    _has_sp = False
    for _algo_key, _algo_info in ALGOS.items():
        _run_sparse = runs_data[_algo_key].get("sparse")
        if _run_sparse and _selected_metric in _run_sparse["metrics"]:
            _y = _extract_series(_run_sparse["metrics"][_selected_metric])
            if _y is not None:
                _ax_sp.plot(_y, label=_algo_info["name"], color=_algo_info["color"], linewidth=2.2)
                _has_sp = True
    _ax_sp.set_title("Sparse Reward (EightRooms)", fontsize=13, fontweight="bold")
    _ax_sp.set_xlabel("Update Step", fontsize=11)
    _ax_sp.set_ylabel(_selected_metric, fontsize=11)
    if _use_log:
        _ax_sp.set_yscale("log")
    _ax_sp.grid(True, linestyle=":", alpha=0.6)
    if _has_sp:
        _ax_sp.legend(frameon=True, fontsize=10)

    # Dense panel
    _has_dn = False
    for _algo_key, _algo_info in ALGOS.items():
        _run_dense = runs_data[_algo_key].get("dense")
        if _run_dense and _selected_metric in _run_dense["metrics"]:
            _y = _extract_series(_run_dense["metrics"][_selected_metric])
            if _y is not None:
                _ax_dn.plot(_y, label=_algo_info["name"], color=_algo_info["color"], linewidth=2.2)
                _has_dn = True
    _ax_dn.set_title("Dense Shaped Reward (EightRooms-Dense)", fontsize=13, fontweight="bold")
    _ax_dn.set_xlabel("Update Step", fontsize=11)
    _ax_dn.set_ylabel(_selected_metric, fontsize=11)
    if _use_log:
        _ax_dn.set_yscale("log")
    _ax_dn.grid(True, linestyle=":", alpha=0.6)
    if _has_dn:
        _ax_dn.legend(frameon=True, fontsize=10)

    fig_env_comp.suptitle(f"{_selected_metric} — Algorithm Benchmark by Environment", fontsize=16, fontweight="bold", y=1.02)
    fig_env_comp.tight_layout()

    # Plot 3: Critic Tracking Diagnostic (True V_start vs Predicted v_pred_start)
    fig_critic, _axes_critic = plt.subplots(2, 2, figsize=(14, 9))
    _axes_critic = _axes_critic.flatten()

    for _i, (_algo_key, _algo_info) in enumerate(ALGOS.items()):
        _ax = _axes_critic[_i]
        _plotted_c = False
        for _env_key, _line_style, _env_label in [("sparse", "-", "Sparse"), ("dense", "--", "Dense")]:
            _run = runs_data[_algo_key].get(_env_key)
            if _run and "V_start" in _run["metrics"] and "v_pred_start" in _run["metrics"]:
                _v_true = _extract_series(_run["metrics"]["V_start"])
                _v_pred = _extract_series(_run["metrics"]["v_pred_start"])
                if _v_true is not None and _v_pred is not None:
                    _ax.plot(_v_true, label=f"True V ({_env_label})", linestyle=_line_style, color="#1f77b4", linewidth=2.0)
                    _ax.plot(_v_pred, label=f"Pred v ({_env_label})", linestyle=_line_style, color="#d62728", linewidth=2.0, alpha=0.8)
                    _plotted_c = True

        _ax.set_title(f"{_algo_info['name']}: Value Tracking Bias", fontsize=13, fontweight="bold")
        _ax.set_xlabel("Update Step", fontsize=11)
        _ax.set_ylabel("Value at Start State", fontsize=11)
        _ax.grid(True, linestyle=":", alpha=0.6)
        if _plotted_c:
            _ax.legend(frameon=True, fontsize=9)
        else:
            _ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=_ax.transAxes, alpha=0.5)

    fig_critic.suptitle("Critic Tracking Diagnostic: Predicted vs. True Start State Value", fontsize=16, fontweight="bold", y=0.99)
    fig_critic.tight_layout()
    return fig_algo_comp, fig_critic, fig_env_comp


@app.cell
def _(fig_algo_comp, fig_critic, fig_env_comp, mo):
    mo.vstack([
        mo.md("## 📈 Learning Curve Comparisons"),
        mo.as_html(fig_algo_comp),
        mo.md("---"),
        mo.as_html(fig_env_comp),
        mo.md("---"),
        mo.md("### 🎯 Critic Tracking Bias & Stability\nComparing predicted value $v(s_0)$ against true policy value $V^{\\pi}(s_0)$ to evaluate value function accuracy and drift:"),
        mo.as_html(fig_critic),
    ])
    return


@app.cell
def _(animation, colors, np, os, plt):
    def generate_and_save_grid_gif(run_data, save_path, fps=5, max_frames=50, seed_idx=0):
        """
        Generates and saves a 1x4 animated GIF comparing:
        1. State Distribution Grid
        2. True Value Grid (V)
        3. Predicted Value Grid (v)
        4. Signed Value Error Grid (v - V)
        """
        metrics = run_data["metrics"]
        algo_name = run_data.get("algo_name", "Algorithm")
        env_label = run_data.get("env_label", "Environment")

        state_dist = metrics.get("state_dist_grid", metrics.get("stat_dist"))
        v_true = metrics.get("V_grid", metrics.get("value_grid"))
        v_pred = metrics.get("nn_grid")

        if state_dist is None or v_true is None or v_pred is None:
            raise ValueError(f"Missing grid metrics in {algo_name} ({env_label}). Ensure LIGHT_METRICS=False.")

        # Index seed if 4D tensor (seeds, updates, height, width)
        if state_dist.ndim == 4:
            s_idx = min(seed_idx, state_dist.shape[0] - 1)
            state_dist = state_dist[s_idx]
        if v_true.ndim == 4:
            s_idx = min(seed_idx, v_true.shape[0] - 1)
            v_true = v_true[s_idx]
        if v_pred.ndim == 4:
            s_idx = min(seed_idx, v_pred.shape[0] - 1)
            v_pred = v_pred[s_idx]

        num_updates = len(state_dist)

        # Broadcast v_true if static (2D)
        if v_true.ndim == 2:
            v_true = np.tile(v_true[None, :, :], (num_updates, 1, 1))

        # Subsample frames
        if max_frames and num_updates > max_frames:
            indices = np.linspace(0, num_updates - 1, max_frames, dtype=int)
            state_dist = state_dist[indices]
            v_true = v_true[indices]
            v_pred = v_pred[indices]
            num_updates = len(indices)

        signed_error = v_pred - v_true

        v_min = float(min(np.min(v_true), np.min(v_pred)))
        v_max = float(max(np.max(v_true), np.max(v_pred)))
        if v_min == v_max:
            v_min, v_max = v_min - 1.0, v_max + 1.0

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4.5))
        fig.suptitle(f"{algo_name} — {env_label} (Seed {seed_idx})", fontsize=15, fontweight="bold", y=1.03)

        # 1. State Distribution Grid
        im1 = ax1.imshow(state_dist[0], cmap="viridis", animated=True)
        ax1.set_title("State Distribution Grid", fontsize=12)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 2. True Value Grid
        im2 = ax2.imshow(v_true[0], cmap="RdBu_r", vmin=v_min, vmax=v_max, animated=True)
        ax2.set_title("True Value Grid (V)", fontsize=12)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Predicted Value Grid
        im3 = ax3.imshow(v_pred[0], cmap="RdBu_r", vmin=v_min, vmax=v_max, animated=True)
        ax3.set_title("Predicted Value Grid (v)", fontsize=12)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # 4. Signed Value Error Grid
        abs_err_max = max(abs(v_min), abs(v_max), float(np.max(np.abs(signed_error))))
        if abs_err_max == 0:
            abs_err_max = 1.0
        im4 = ax4.imshow(
            signed_error[0],
            cmap="RdBu_r",
            animated=True,
            norm=colors.SymLogNorm(linthresh=0.01, vmin=-abs_err_max, vmax=abs_err_max, base=10),
        )
        ax4.set_title("Signed Value Error (v - V)", fontsize=12)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        frame_text = fig.text(0.5, 0.01, f"Frame 1 / {num_updates}", ha="center", fontsize=11, fontweight="bold")
        plt.tight_layout()

        def update(frame):
            im1.set_array(state_dist[frame])
            im1.set_clim(np.min(state_dist[frame]), np.max(state_dist[frame]))
            im2.set_array(v_true[frame])
            im3.set_array(v_pred[frame])
            im4.set_array(signed_error[frame])
            frame_text.set_text(f"Frame {frame + 1} / {num_updates}")
            return im1, im2, im3, im4, frame_text

        ani = animation.FuncAnimation(fig, update, frames=num_updates, interval=1000 // fps, blit=False)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)
        return save_path

    return (generate_and_save_grid_gif,)


@app.cell
def _(ALGOS, mo):
    algo_gif_selector = mo.ui.dropdown(
        options=list(ALGOS.keys()),
        value="TD_lambda",
        label="Algorithm for Side-by-Side Comparison",
    )

    generate_btn = mo.ui.button(
        label="🎬 Generate / Refresh Comparison GIFs",
    )
    return algo_gif_selector, generate_btn


@app.cell
def _(algo_gif_selector, generate_btn, mo):
    mo.vstack([
        mo.md("## 🎬 Animated Grid Visualizations (Sparse vs Dense)"),
        mo.md("Select an algorithm to render and compare its animated evolution side-by-side:"),
        mo.hstack([algo_gif_selector, generate_btn], justify="start", gap=2),
    ])
    return


@app.cell
def _(
    ALGOS,
    algo_gif_selector,
    current_suffix,
    fps_slider,
    generate_and_save_grid_gif,
    generate_btn,
    max_frames_slider,
    mo,
    os,
    runs_data,
    seed_idx_input,
):
    _selected_algo = algo_gif_selector.value
    _algo_label = ALGOS[_selected_algo]["name"]

    _sparse_run = runs_data[_selected_algo].get("sparse")
    _dense_run = runs_data[_selected_algo].get("dense")

    _figures_dir = os.path.join("figures", "dense_vs_sparse", current_suffix)
    os.makedirs(_figures_dir, exist_ok=True)

    _gif_sparse_path = os.path.join(_figures_dir, f"{_selected_algo}_EightRooms.gif")
    _gif_dense_path = os.path.join(_figures_dir, f"{_selected_algo}_EightRooms-Dense.gif")

    _output_elements = []

    # Trigger generation when button is clicked or if GIFs already exist
    _need_gen = generate_btn.value

    # Process sparse
    if _sparse_run:
        if _need_gen or not os.path.exists(_gif_sparse_path):
            try:
                generate_and_save_grid_gif(
                    _sparse_run,
                    _gif_sparse_path,
                    fps=fps_slider.value,
                    max_frames=max_frames_slider.value,
                    seed_idx=seed_idx_input.value,
                )
            except Exception as _e:
                _gif_sparse_path = None
                _output_elements.append(mo.md(f"⚠️ Sparse GIF error: {_e}"))
    else:
        _gif_sparse_path = None

    # Process dense
    if _dense_run:
        if _need_gen or not os.path.exists(_gif_dense_path):
            try:
                generate_and_save_grid_gif(
                    _dense_run,
                    _gif_dense_path,
                    fps=fps_slider.value,
                    max_frames=max_frames_slider.value,
                    seed_idx=seed_idx_input.value,
                )
            except Exception as _e:
                _gif_dense_path = None
                _output_elements.append(mo.md(f"⚠️ Dense GIF error: {_e}"))
    else:
        _gif_dense_path = None

    # Render side by side
    _col_sparse = mo.vstack([
        mo.md(f"### 🚪 Sparse Reward: EightRooms — {_algo_label}"),
        mo.image(src=_gif_sparse_path) if _gif_sparse_path and os.path.exists(_gif_sparse_path) else mo.md("*(GIF not generated or run not found)*")
    ])

    _col_dense = mo.vstack([
        mo.md(f"### ⚡ Shaped Dense: EightRooms-Dense — {_algo_label}"),
        mo.image(src=_gif_dense_path) if _gif_dense_path and os.path.exists(_gif_dense_path) else mo.md("*(GIF not generated or run not found)*")
    ])

    mo.vstack([
        mo.hstack([_col_sparse, _col_dense], justify="start", gap=3),
        *_output_elements
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
