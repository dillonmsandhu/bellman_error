import os
import json
import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import jax.numpy as jnp
import jax
from core.utils import load_run_data, load_run_data_from_path
from core.helpers import make_env, initialize_evaluator

DEFAULT_COLORS = [
    "#1f77b4",  # Blue
    "#d62728",  # Red
    "#2ca02c",  # Green
    "#9467bd",  # Purple
    "#ff7f0e",  # Orange
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# ---------------------------------------------------------------------------
# Policy Loading & Ground Truth Computation
# ---------------------------------------------------------------------------

def load_fixed_policy(model_load_dir_or_path, env_name="FourRooms-misc", base_results_path="results"):
    """
    Loads a pretrained policy (e.g. from PPO) to evaluate as a fixed policy.
    
    Args:
        model_load_dir_or_path: Path or subpath to the model run directory, 
                               e.g. 'ppo/ground_truth/short_run' or 'ground_truth/short_run'
                               or an absolute / direct path containing config.json and out.pkl.
        env_name: Environment name (e.g. 'FourRooms-misc').
        base_results_path: Base results directory.
        
    Returns:
        policy_fn: Callable lambda obs: action_probs (or categorical distribution)
        policy_train_state: Loaded train state object
        policy_config: Loaded config dictionary
    """
    # 1. Try direct / absolute path
    if os.path.exists(os.path.join(model_load_dir_or_path, "config.json")):
        policy_config, out = load_run_data_from_path(model_load_dir_or_path)
    elif os.path.exists(os.path.join(model_load_dir_or_path, env_name, "config.json")):
        policy_config, out = load_run_data_from_path(os.path.join(model_load_dir_or_path, env_name))
    else:
        # 2. Try relative path under base_results_path
        target_sub = model_load_dir_or_path
        if not target_sub.startswith("ppo/") and not os.path.exists(os.path.join(base_results_path, target_sub, env_name)):
            candidate = os.path.join(base_results_path, "ppo", target_sub, env_name)
            if os.path.exists(candidate):
                target_sub = os.path.join("ppo", target_sub)
        policy_config, out = load_run_data(target_sub, env_name, base_results_path)
    
    policy_train_state = out['runner_state'][0]
    policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_train_state.params)
    
    def policy_fn(obs):
        res = policy_train_state.apply_fn(policy_params, obs)
        pi_dist = res[0] if isinstance(res, (tuple, list)) else res
        if hasattr(pi_dist, 'probs'):
            return pi_dist.probs
        return pi_dist
        
    return policy_fn, policy_train_state, policy_config


def compute_ground_truth(evaluator, policy_type="random", fixed_policy_fn=None):
    """
    Computes true value function V^pi and stationary distribution mu for either
    a uniform random policy or an imported fixed policy.
    
    Args:
        evaluator: An initialized evaluator object (e.g. FourRoomsExactValue).
        policy_type: "random" or "fixed"
        fixed_policy_fn: Callable policy taking obs and returning action probabilities.
                         Required if policy_type == "fixed".
                         
    Returns:
        dict containing:
            - 'V_pi': 1D array of true values including terminal state (shape S+1)
            - 'v_grid': 2D grid of true values (shape H, W)
            - 'mu_vec': 1D stationary distribution vector (shape S)
            - 'mu_grid': 2D grid of stationary distribution (shape H, W)
            - 'target_policy_probs': active states policy matrix (shape S, A)
            - 'pi': policy matrix including terminal state (shape S+1, A)
    """
    n_states = len(evaluator.obs_stack)
    n_actions = evaluator.num_actions
    
    if policy_type == "fixed":
        if fixed_policy_fn is None:
            raise ValueError("fixed_policy_fn must be provided when policy_type is 'fixed'")
        target_policy_probs = fixed_policy_fn(evaluator.obs_stack)
        if hasattr(target_policy_probs, 'probs'):
            target_policy_probs = target_policy_probs.probs
        target_policy_probs = jnp.asarray(target_policy_probs)
    else:
        # Uniform random policy
        target_policy_probs = jnp.ones((n_states, n_actions)) / n_actions
        
    terminal_policy = jnp.ones((1, n_actions), dtype=target_policy_probs.dtype) / n_actions
    pi = jnp.vstack([target_policy_probs, terminal_policy])
    
    V_pi = evaluator.compute_true_values_raw(pi)
    mu_vec, _ = evaluator.compute_stationary_distribution_raw(target_policy_probs)
    
    v_grid = evaluator.get_value_grid(V_pi[:-1])
    mu_grid = evaluator.get_value_grid(mu_vec)
    
    return {
        "V_pi": V_pi,
        "v_grid": v_grid,
        "mu_vec": mu_vec,
        "mu_grid": mu_grid,
        "target_policy_probs": target_policy_probs,
        "pi": pi,
    }


# ---------------------------------------------------------------------------
# Run Resolution & Value Extraction
# ---------------------------------------------------------------------------

def resolve_runs(runs_dict, default_env_name="FourRooms-misc", base_results_path="results"):
    """
    Standardizes and loads a dictionary of runs.
    
    Args:
        runs_dict: Dict mapping run_key (str) to a dict with configuration:
            {
                "run_dir": "path/to/run", # optional if metrics is provided
                "title": "Display Title", # optional, defaults to run_key
                "color": "blue",          # optional, assigned automatically if missing
                "metrics": {...},         # optional
                "config": {...},          # optional
            }
            or mapping run_key to a string path "path/to/run".
            
    Returns:
        resolved_runs: Dict mapping run_key to normalized dict with:
            {"title", "color", "run_dir", "config", "metrics"}
    """
    resolved = {}
    for i, (key, spec) in enumerate(runs_dict.items()):
        if isinstance(spec, str):
            spec = {"run_dir": spec}
        else:
            spec = dict(spec)
            
        title = spec.get("title", key)
        color = spec.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        run_dir = spec.get("run_dir", None)
        metrics = spec.get("metrics", None)
        config = spec.get("config", None)
        
        # If metrics is not provided, try to load from run_dir
        if metrics is None and run_dir is not None:
            try:
                # 1. Try direct path
                if os.path.exists(os.path.join(run_dir, "config.json")):
                    config, metrics = load_run_data_from_path(run_dir)
                elif os.path.exists(os.path.join(run_dir, default_env_name, "config.json")):
                    config, metrics = load_run_data_from_path(os.path.join(run_dir, default_env_name))
                else:
                    # 2. Try relative to base_results_path
                    config, metrics = load_run_data(run_dir, default_env_name, base_results_path)
            except Exception as e:
                print(f"Note: Could not load run '{key}' from '{run_dir}': {e}")
                metrics = None
                config = None
                
        resolved[key] = {
            "title": title,
            "color": color,
            "run_dir": run_dir,
            "config": config,
            "metrics": metrics,
        }
    return resolved


def get_run_value_grid(run_entry_or_metrics, evaluator, epoch_idx=-1, combo_idx=0, seed_idx=0):
    """
    Extracts the learned value function from a run's metrics at a given epoch/step
    and formats it as a 2D spatial grid.
    
    Args:
        run_entry_or_metrics: Either a dict containing 'metrics', or the metrics dict itself.
        evaluator: An initialized evaluator object with .get_value_grid().
        epoch_idx: Index of epoch/step to extract (default: -1, final step).
        combo_idx: Hyperparameter combo index if batched (default: 0).
        seed_idx: Seed index if batched (default: 0).
        
    Returns:
        v_grid: 2D numpy / jax array of estimated values.
    """
    if isinstance(run_entry_or_metrics, dict) and "metrics" in run_entry_or_metrics:
        metrics = run_entry_or_metrics["metrics"]
    else:
        metrics = run_entry_or_metrics
        
    if metrics is None:
        return None
        
    # Check for V_nn, v_pred, or V keys
    v_data = None
    for k in ["V_nn", "v_pred", "V", "v"]:
        if k in metrics:
            v_data = np.asarray(metrics[k])
            break
            
    if v_data is None:
        raise KeyError("Could not find value array (checked 'V_nn', 'v_pred', 'V') in metrics.")
        
    # Squeeze or index down to (num_states,)
    if v_data.ndim == 4:  # (combos, seeds, epochs, states)
        v_vec = v_data[combo_idx, seed_idx, epoch_idx]
    elif v_data.ndim == 3:  # (seeds, epochs, states) or (combos, epochs, states)
        v_vec = v_data[seed_idx, epoch_idx]
    elif v_data.ndim == 2:  # (epochs, states)
        v_vec = v_data[epoch_idx]
    elif v_data.ndim == 1:  # (states,)
        v_vec = v_data
    else:
        raise ValueError(f"Unexpected shape for value data: {v_data.shape}")
        
    # If v_vec has terminal state included (e.g. S+1 states), take first S states
    n_states = len(evaluator.obs_stack)
    if len(v_vec) > n_states:
        v_vec = v_vec[:n_states]
        
    return evaluator.get_value_grid(v_vec)


# ---------------------------------------------------------------------------
# 2D Grid Visualizations
# ---------------------------------------------------------------------------

def plot_grid(grid, evaluator, title="Value Grid", cmap="viridis", logscale=False, ax=None, show_start_goal=True, vmin=None, vmax=None):
    """
    Plots a single 2D grid with environment walls masked and optional Start/Goal annotations.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure

    mask = evaluator.occupied_map
    masked_grid = jnp.where(mask == 1, jnp.nan, grid)

    # Background walls
    ax.imshow(mask, cmap="Greys", alpha=0.3)

    if logscale:
        valid_pos = masked_grid[jnp.isfinite(masked_grid) & (masked_grid > 0)]
        calc_min = float(jnp.min(valid_pos)) if len(valid_pos) > 0 else 1e-6
        calc_max = float(jnp.max(valid_pos)) if len(valid_pos) > 0 else 1.0

        _vmin = float(vmin) if vmin is not None and vmin > 0 else calc_min
        _vmax = float(vmax) if vmax is not None and vmax > 0 else calc_max

        if _vmin <= 0:
            _vmin = 1e-6
        if _vmax <= _vmin:
            _vmax = _vmin * 10.0 if _vmin > 0 else 1.0

        norm = LogNorm(vmin=_vmin, vmax=_vmax)
        im = ax.imshow(masked_grid, cmap=cmap, norm=norm)
    else:
        im = ax.imshow(masked_grid, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_start_goal and hasattr(evaluator, "start") and hasattr(evaluator, "goal"):
        start_y, start_x = evaluator.start
        goal_y, goal_x = evaluator.goal
        ax.text(start_x, start_y, 'S', ha='center', va='center', color='blue', fontweight='bold', fontsize=14)
        ax.text(goal_x, goal_y, 'G', ha='center', va='center', color='red', fontweight='bold', fontsize=14)

    ax.set_title(title)
    ax.axis("off")
    
    if created_fig:
        plt.tight_layout()
        return fig
    return ax


def plot_multi_grids(grids_dict, evaluator, cmap="viridis", cmaps=None, use_log=False, shared_vrange=True, ncols=None, figsize=None, show_start_goal=True):
    """
    Plots an arbitrary number of 2D grids side-by-side.
    
    Args:
        grids_dict: Dict mapping label -> 2D grid array, or list of (label, 2D grid array).
        evaluator: Evaluator object with .occupied_map, .start, .goal.
        cmap: Default colormap.
        cmaps: Optional dict mapping label -> colormap name.
        use_log: Whether to use LogNorm.
        shared_vrange: If True, uses the min/max across all grids for a shared colorbar range.
        ncols: Number of subplot columns (defaults to len(grids_dict) or up to 4 per row).
        figsize: Tuple (width, height) or auto-computed.
        show_start_goal: Whether to mark Start ('S') and Goal ('G').
        
    Returns:
        fig: Matplotlib Figure object.
    """
    items = list(grids_dict.items()) if isinstance(grids_dict, dict) else grids_dict
    items = [(lbl, g) for (lbl, g) in items if g is not None]
    n = len(items)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No valid grids to display", ha="center", va="center")
        ax.axis("off")
        return fig

    if ncols is None:
        ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    
    if figsize is None:
        figsize = (5.5 * ncols, 5 * nrows)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    mask = evaluator.occupied_map
    
    # Calculate shared vmin/vmax if requested
    if shared_vrange:
        all_masked = [jnp.where(mask == 1, jnp.nan, g) for (_, g) in items]
        if use_log:
            pos_vals = [m[jnp.isfinite(m) & (m > 0)] for m in all_masked]
            pos_vals = [p for p in pos_vals if len(p) > 0]
            if len(pos_vals) > 0:
                vmin = min(float(jnp.min(p)) for p in pos_vals)
                vmax = max(float(jnp.max(p)) for p in pos_vals)
            else:
                vmin, vmax = 1e-6, 1.0
            if vmin <= 0:
                vmin = 1e-6
            if vmax <= vmin:
                vmax = vmin * 10.0
        else:
            vmin = min(float(jnp.nanmin(m)) for m in all_masked)
            vmax = max(float(jnp.nanmax(m)) for m in all_masked)
            if vmin == vmax:
                vmax = vmin + 1e-6
    else:
        vmin = vmax = None

    for i, (label, grid) in enumerate(items):
        ax = axes_flat[i]
        c = cmaps.get(label, cmap) if cmaps else cmap
        plot_grid(grid, evaluator, title=label, cmap=c, logscale=use_log, ax=ax, show_start_goal=show_start_goal, vmin=vmin, vmax=vmax)

    # Hide any unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    return fig


def plot_error_grids(runs_or_grids, v_true_grid, evaluator, metric_type="squared_error", use_log=True, shared_vrange=True, cmap="magma", cmaps=None, ncols=None, figsize=None):
    """
    Computes and plots error heatmaps relative to v_true_grid for multiple runs.
    
    Args:
        runs_or_grids: Dict mapping label -> value_grid (or run dict with 'metrics').
        v_true_grid: 2D array of true values.
        evaluator: Evaluator object.
        metric_type: "squared_error" ((v - v_true)^2), "underestimation" (v_true - v), "abs_error" (|v - v_true|).
        use_log: Whether to use log color scale.
        shared_vrange: Whether all error plots share the same colorbar range.
    """
    error_dict = {}
    items = runs_or_grids.items() if isinstance(runs_or_grids, dict) else runs_or_grids
    
    for key, val in items:
        # Resolve grid
        if isinstance(val, dict) and "metrics" in val:
            g = get_run_value_grid(val, evaluator)
            lbl = val.get("title", key)
        elif isinstance(val, dict) and "grid" in val:
            g = val["grid"]
            lbl = val.get("title", key)
        else:
            g = val
            lbl = key
            
        if g is None:
            continue
            
        if metric_type == "squared_error":
            err = (g - v_true_grid) ** 2
            title = f"{lbl} ($MSE$)"
        elif metric_type == "underestimation":
            err = v_true_grid - g
            title = f"{lbl} ($V_{{true}} - V_{{learned}}$)"
        elif metric_type == "abs_error":
            err = np.abs(g - v_true_grid)
            title = f"{lbl} ($|V_{{true}} - V_{{learned}}|$)"
        else:
            err = (g - v_true_grid) ** 2
            title = f"{lbl} Error"
            
        error_dict[title] = err
        
    return plot_multi_grids(error_dict, evaluator, cmap=cmap, cmaps=cmaps, use_log=use_log, shared_vrange=shared_vrange, ncols=ncols, figsize=figsize)


# ---------------------------------------------------------------------------
# 3D Visualizations & Interactive Comparison
# ---------------------------------------------------------------------------

def plot_3d_comparison(v_true, runs_value_dict, evaluator, title="Value Function Comparison (3D Lines)", figsize=(12, 10)):
    """
    Plots a 3D comparison showing lines along rows and columns for True value and each learned run.
    
    Args:
        v_true: 2D grid of ground truth values.
        runs_value_dict: Dict mapping run_name/label -> 2D value grid, 
                         or run_name/label -> {"grid": 2D grid, "color": str, "title": str}.
        evaluator: Evaluator object with .occupied_map.
        title: Plot title.
        figsize: Figure dimensions.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    mask = evaluator.occupied_map
    ny, nx = v_true.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    vt = jnp.where(mask == 1, jnp.nan, v_true)

    # Plot True value lines
    for i in range(ny):
        ax.plot(X[i, :], Y[i, :], vt[i, :], color='black', alpha=0.5, linewidth=1.5, label='True' if i == 0 else "")
    for j in range(nx):
        ax.plot(X[:, j], Y[:, j], vt[:, j], color='black', alpha=0.2, linewidth=1)

    legend_handles = [Line2D([0], [0], color='black', lw=3, alpha=0.8, label='True')]

    for i_run, (key, val) in enumerate(runs_value_dict.items()):
        if isinstance(val, dict):
            grid = val.get("grid", val.get("value_grid", None))
            color = val.get("color", DEFAULT_COLORS[i_run % len(DEFAULT_COLORS)])
            label = val.get("title", key)
        else:
            grid = val
            color = DEFAULT_COLORS[i_run % len(DEFAULT_COLORS)]
            label = key

        if grid is None:
            continue

        vg = jnp.where(mask == 1, jnp.nan, grid)

        # Plot rows & cols
        for i in range(ny):
            ax.plot(X[i, :], Y[i, :], vg[i, :], color=color, alpha=0.6, linewidth=1.5)
        for j in range(nx):
            ax.plot(X[:, j], Y[:, j], vg[:, j], color=color, alpha=0.6, linewidth=1.5)

        legend_handles.append(Line2D([0], [0], color=color, lw=2, alpha=0.8, label=label))

    ax.set_title(title)
    ax.set_xlabel('X (Col)')
    ax.set_ylabel('Y (Row)')
    ax.set_zlabel('Value')
    ax.legend(handles=legend_handles)

    plt.tight_layout()
    return fig


def render_3d(v_true, runs_value_dict, evaluator, elev=30, azim=-60, show_toggles=None, title_prefix="Value Comparison", figsize=(14, 7)):
    """
    Renders two interactive 3D plots:
    Plot 1 (Left): Ground truth value surface with overlaid wireframes for each active run.
    Plot 2 (Right): Underestimation error surfaces (V_true - V_learned) for each active run.
    
    Args:
        v_true: 2D ground truth value grid.
        runs_value_dict: Dict mapping run_name -> 2D value grid or dict with "grid", "color", "title".
        evaluator: Evaluator object with .occupied_map.
        elev: Elevation viewing angle in degrees.
        azim: Azimuth viewing angle in degrees.
        show_toggles: Optional dict mapping run_name -> bool (to toggle visibility).
        title_prefix: Plot title prefix.
    """
    fig = plt.figure(figsize=figsize)
    mask = evaluator.occupied_map
    ny, nx = v_true.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    vt = jnp.where(mask == 1, jnp.nan, v_true)

    # Plot 1: Comparison
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, vt, cmap='viridis', alpha=0.3, antialiased=True)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    legend_elements = [Line2D([0], [0], color='green', lw=4, alpha=0.3, label='True (Surface)')]

    # Plot 2: Error
    ax2 = fig.add_subplot(122, projection='3d')

    for i_run, (key, val) in enumerate(runs_value_dict.items()):
        if isinstance(val, dict):
            grid = val.get("grid", val.get("value_grid", None))
            color = val.get("color", DEFAULT_COLORS[i_run % len(DEFAULT_COLORS)])
            label = val.get("title", key)
        else:
            grid = val
            color = DEFAULT_COLORS[i_run % len(DEFAULT_COLORS)]
            label = key

        if grid is None:
            continue

        # Check visibility toggle
        if show_toggles is not None:
            is_visible = show_toggles.get(key, show_toggles.get(label, True))
            if not is_visible:
                continue

        vg = jnp.where(mask == 1, jnp.nan, grid)

        # Plot 1 wireframe lines
        for i in range(ny):
            ax1.plot(X[i, :], Y[i, :], vg[i, :], color=color, alpha=0.8, linewidth=1)
        for j in range(nx):
            ax1.plot(X[:, j], Y[:, j], vg[:, j], color=color, alpha=0.8, linewidth=1)

        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))

        # Plot 2 error surface
        err_surface = vt - vg
        ax2.plot_surface(X, Y, err_surface, color=color, alpha=0.4)

    ax1.set_title(f"{title_prefix} (Surface = True)")
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Value')
    ax1.legend(handles=legend_elements)

    ax2.set_title("Underestimation Error ($V_{true} - V_{learned}$)")
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Error')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Jaggedness & Feature Analysis
# ---------------------------------------------------------------------------

def compute_jaggedness(grid, mask):
    """
    Computes total variation / smoothness metric over non-wall cells.
    """
    grid_masked = jnp.where(mask == 1, jnp.nan, grid)
    diff_h = jnp.abs(grid_masked[:, 1:] - grid_masked[:, :-1])
    diff_v = jnp.abs(grid_masked[1:, :] - grid_masked[:-1, :])
    return float(jnp.nanmean(diff_h) + jnp.nanmean(diff_v))


def compute_runs_jaggedness(runs_value_dict, mask, v_true=None):
    """
    Computes jaggedness for true values and all runs.
    Returns a dict mapping label -> jaggedness score.
    """
    res = {}
    if v_true is not None:
        res["Ground Truth"] = compute_jaggedness(v_true, mask)
        
    for i, (key, val) in enumerate(runs_value_dict.items()):
        if isinstance(val, dict):
            grid = val.get("grid", val.get("value_grid", None))
            label = val.get("title", key)
        else:
            grid = val
            label = key
            
        if grid is not None:
            res[label] = compute_jaggedness(grid, mask)
            
    return res


def plot_feature_singular_vectors(metrics, evaluator=None, title_prefix="Feature Spatial Patterns", seed=0, epoch_idx=-1, n_components=5, figsize=None):
    """
    Plots the top feature singular vectors from metrics.
    """
    if metrics is None or "feature_top_singular_vectors" not in metrics:
        return None
        
    top_vectors = np.array(metrics["feature_top_singular_vectors"])
    while top_vectors.ndim > 4:
        top_vectors = top_vectors[0]
        
    stack = top_vectors[epoch_idx] if top_vectors.ndim == 4 else top_vectors
    if stack.ndim == 4:
        stack = stack[seed]
        
    n_components = min(n_components, len(stack))
    if figsize is None:
        figsize = (4 * n_components, 4)
        
    fig, axes = plt.subplots(1, n_components, figsize=figsize)
    if n_components == 1:
        axes = [axes]
        
    mask = evaluator.occupied_map if evaluator is not None else None
    
    for i in range(n_components):
        ax = axes[i]
        grid = stack[i]
        
        if mask is not None:
            ax.imshow(mask, cmap="Greys", alpha=0.3)
            masked_grid = np.where(mask == 1, np.nan, grid)
        else:
            masked_grid = grid
            
        im = ax.imshow(masked_grid, cmap="coolwarm")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Component {i+1}")
        ax.axis("off")
        
    fig.suptitle(f"{title_prefix} Top Feature Spatial Patterns", fontsize=16)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Existing Legacy & General Utilities
# ---------------------------------------------------------------------------

def get_latest_run_path(base_tuning_dir):
    """Finds the latest timestamped directory and environment name under a tuning or results directory."""
    if not os.path.exists(base_tuning_dir):
        raise FileNotFoundError(f"Directory not found: {base_tuning_dir}")
    timestamps = sorted([d for d in os.listdir(base_tuning_dir) if os.path.isdir(os.path.join(base_tuning_dir, d))])
    if not timestamps:
        raise FileNotFoundError(f"No run timestamps found in {base_tuning_dir}")
    latest_ts = timestamps[-1]
    ts_path = os.path.join(base_tuning_dir, latest_ts)
    envs = [e for e in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, e))]
    if not envs:
        raise FileNotFoundError(f"No environment folders found in {ts_path}")
    env_name = envs[0]
    return latest_ts, env_name, os.path.join(base_tuning_dir, latest_ts, env_name)

def load_sweep_run(tuning_base_dir):
    """Loads config, metrics, and tuning summary for a sweep/tuning run."""
    ts, env_name, run_dir = get_latest_run_path(tuning_base_dir)
    config, metrics = load_run_data(ts, env_name, results_base_path=tuning_base_dir)
    
    summary_path = os.path.join(run_dir, "tuning_summary.csv")
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    
    print(f"Loaded sweep run from {run_dir}")
    return config, metrics, summary_df

def load_sweep_data_from_path(run_dir):
    """Loads config, metrics, and tuning summary for a sweep/tuning run."""
    config, metrics = load_run_data_from_path(run_dir)
    
    summary_path = os.path.join(run_dir, "tuning_summary.csv")
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    
    print(f"Loaded sweep run from {run_dir}")
    return config, metrics, summary_df

def plot_mean_std_curve(data_tensor, steps_per_pi=1, metric_name="Metric", ylabel="Value", log_scale=True, label_prefix="Run", tuning_summary=None):
    """
    Plots a single run's mean trajectory with a standard deviation band (± 1 std) across seeds.
    """
    return plot_multi_mean_std_curves(
        {label_prefix: data_tensor},
        steps_per_pi=steps_per_pi,
        metric_name=metric_name,
        ylabel=ylabel,
        log_scale=log_scale,
        tuning_summaries={label_prefix: tuning_summary} if tuning_summary is not None else None
    )

def plot_multi_mean_std_curves(runs_dict, steps_per_pi=1, metric_name="Metric", ylabel="Value", log_scale=True, tuning_summaries=None):
    """
    Plots multiple runs' mean trajectories with standard deviation bands (± 1 std) on the same plot for comparison.
    """
    fig = plt.figure(figsize=(10, 6))

    for label, data_tensor in runs_dict.items():
        arr = np.array(data_tensor)
        if arr.ndim == 3:
            if tuning_summaries is not None and label in tuning_summaries and tuning_summaries[label] is not None:
                summary_df = tuning_summaries[label]
                best_idx = int(summary_df.iloc[0]['config_idx'])
                lr = summary_df.loc[summary_df['config_idx'] == best_idx]['LR'].item()
                print(f"[{label}] Using best combo index {best_idx} from tuning summary.")
            else:
                print(f"label {label} not in tuning summaries")
                final_means = arr[:, :, -1].mean(axis=1)
                best_idx = int(np.argmin(final_means))
                print(f"[{label}] Using best combo index {best_idx} from 3D tensor.")
            arr = arr[best_idx]
        
        if arr.ndim != 2:
            raise ValueError(f"[{label}] Expected 2D array (n_seeds, time_steps), got shape {arr.shape}")

        n_seeds, time_steps = arr.shape
        print('number of seeds:', n_seeds)
        x = [i * steps_per_pi for i in range(time_steps)]
        
        log_arr = np.log(arr)
        log_mean = np.mean(log_arr, axis=0)
        log_std = np.std(log_arr, axis=0)

        geom_mean = np.exp(log_mean)
        lower_bound = np.exp(log_mean - log_std)
        upper_bound = np.exp(log_mean + log_std)
        try:
            line, = plt.plot(x, geom_mean, label=f"{label} (LR = {lr})", linewidth=2)
        except:
            line, = plt.plot(x, geom_mean, label=f"{label}", linewidth=2)
        plt.fill_between(x, lower_bound, upper_bound, color=line.get_color(), alpha=0.2)

    if log_scale:
        plt.yscale('log')
    plt.xlabel("Gradient Update Steps")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} Comparison over Training Course")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    return fig

def plot_generalization_heatmaps(metrics, env_name="Environment", show_start=True):
    """
    Plots heatmaps for NTK, Centered NTK, Jacobian singular vectors, and Feature singular vectors.
    """
    if "eNTK" in metrics:
        eNTK_data = np.array(metrics["eNTK"])
        while eNTK_data.ndim > 2:
            eNTK_data = eNTK_data[0]
        
        plt.figure(figsize=(8, 6))
        plt.matshow(eNTK_data, cmap="viridis")
        plt.colorbar(label="NTK Value")
        plt.title(f"{env_name} - Empirical NTK Matrix (eNTK)")
        plt.xlabel("State Index")
        plt.ylabel("State Index")
        plt.tight_layout()
        plt.show()

    if "gradient_covariance_matrix" in metrics:
        cov_data = np.array(metrics["gradient_covariance_matrix"])
        while cov_data.ndim > 2:
            cov_data = cov_data[0]
        
        plt.figure(figsize=(8, 6))
        plt.matshow(cov_data, cmap="viridis")
        plt.colorbar(label="Covariance Value")
        plt.title(f"{env_name} - Centered NTK (Gradient Covariance Matrix)")
        plt.xlabel("State Index")
        plt.ylabel("State Index")
        plt.tight_layout()
        plt.show()

    if "Jacobian_top_singular_vectors" in metrics:
        j_svs = np.array(metrics["Jacobian_top_singular_vectors"])
        while j_svs.ndim > 4:
            j_svs = j_svs[0]
        stack = j_svs[-1] if j_svs.ndim == 4 else j_svs
        n_components = min(5, len(stack))
        
        fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
        fig.suptitle(f"{env_name} - Top {n_components} Jacobian Singular Vectors", fontsize=16)
        
        if n_components == 1:
            axes = [axes]
            
        for i in range(n_components):
            grid = stack[i]
            max_abs = np.max(np.abs(grid))
            if max_abs == 0:
                max_abs = 1.0
            
            ax = axes[i]
            im = ax.matshow(grid, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
            ax.set_title(f"Component {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    if "feature_top_singular_vectors" in metrics:
        f_svs = np.array(metrics["feature_top_singular_vectors"])
        while f_svs.ndim > 4:
            f_svs = f_svs[0]
        stack = f_svs[-1] if f_svs.ndim == 4 else f_svs
        n_components = min(5, len(stack))
        
        fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
        fig.suptitle(f"{env_name} - Top {n_components} Feature Singular Vectors", fontsize=16)
        
        if n_components == 1:
            axes = [axes]
            
        for i in range(n_components):
            grid = stack[i]
            max_abs = np.max(np.abs(grid))
            if max_abs == 0:
                max_abs = 1.0
            
            ax = axes[i]
            im = ax.matshow(grid, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
            ax.set_title(f"Component {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

plot_geneneralizatio_heatmaps = plot_generalization_heatmaps

def plot_multi_spectra(metrics_dict, env_name="Environment", show_start=True):
    """
    Plots spectra for Jacobian and Features comparing multiple runs (e.g. TD vs MC).
    """
    colors = plt.cm.tab10.colors
    figs = []
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    has_jacobian = False
    
    for i, (label, metrics) in enumerate(metrics_dict.items()):
        if "jacobian_singular_values" in metrics:
            has_jacobian = True
            color = colors[i % len(colors)]
            j_sv = np.array(metrics["jacobian_singular_values"])
            while j_sv.ndim > 2:
                j_sv = j_sv[0]
            start_sv = j_sv[0] if j_sv.ndim == 2 else j_sv
            end_sv = j_sv[-1] if j_sv.ndim == 2 else j_sv
            
            if show_start and j_sv.ndim == 2:
                light_color = tuple(min(1.0, c + 0.5 * (1.0 - c)) for c in color[:3])
                ax1.plot(np.arange(1, len(start_sv)+1), start_sv, linestyle='--', color=light_color, alpha=0.7, label=f'{label} Start')
            ax1.plot(np.arange(1, len(end_sv)+1), end_sv, color=color, label=f'{label} End')
            
    if has_jacobian:
        ax1.set_yscale('log')
        ax1.set_xlabel("Singular Value Index")
        ax1.set_ylabel("Singular Value Magnitude (Log Scale)")
        ax1.set_title(f"{env_name} - Jacobian Singular Value Spectrum Comparison")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.legend(loc='upper right')
        fig1.tight_layout()
        figs.append(fig1)
    else:
        plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    has_features = False
    
    for i, (label, metrics) in enumerate(metrics_dict.items()):
        if "feature_singular_values" in metrics:
            has_features = True
            color = colors[i % len(colors)]
            f_sv = np.array(metrics["feature_singular_values"])
            while f_sv.ndim > 2:
                f_sv = f_sv[0]
            start_sv = f_sv[0] if f_sv.ndim == 2 else f_sv
            end_sv = f_sv[-1] if f_sv.ndim == 2 else f_sv
            
            if show_start and f_sv.ndim == 2:
                light_color = tuple(min(1.0, c + 0.5 * (1.0 - c)) for c in color[:3])
                ax2.plot(np.arange(1, len(start_sv)+1), start_sv, linestyle='--', color=light_color, alpha=0.7, label=f'{label} Start')
            ax2.plot(np.arange(1, len(end_sv)+1), end_sv, color=color, label=f'{label} End')

    if has_features:
        ax2.set_yscale('log')
        ax2.set_xlabel("Singular Value Index")
        ax2.set_ylabel("Singular Value Magnitude (Log Scale)")
        ax2.set_title(f"{env_name} - Feature Singular Value Spectrum Comparison")
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)
        ax2.legend(loc='upper right')
        fig2.tight_layout()
        figs.append(fig2)
    else:
        plt.close(fig2)

    return figs if len(figs) > 1 else (figs[0] if figs else None)

def save_3d_value_surface(env_dir, env_name, value_grid, title, algorithm_name):
    """
    Plots the value function as a 3D surface to reveal local jaggedness/smoothness.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    y = np.arange(value_grid.shape[0])
    x = np.arange(value_grid.shape[1])
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, value_grid, cmap='viridis', 
                           linewidth=0.1, edgecolor='k', antialiased=False, alpha=0.9)
    
    ax.set_title(f"{env_name} - {algorithm_name} Value Surface\n({title})")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Estimated Value")
    ax.view_init(elev=30, azim=45)
    
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    os.makedirs(env_dir, exist_ok=True)
    save_path = os.path.join(env_dir, f"{env_name}_{algorithm_name}_3d_surface.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_jacobian_singular_vectors(metrics, env_name="Environment"):
    """
    Plots the top Jacobian singular vectors from the final training epoch.
    """
    j_svs = np.array(metrics["Jacobian_top_singular_vectors"])
    while j_svs.ndim > 4:
        j_svs = j_svs[0]
        
    stack = j_svs[-1] if j_svs.ndim == 4 else j_svs
    n_components = min(5, len(stack))
    
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
    fig.suptitle(f"{env_name} - Top {n_components} Jacobian Singular Vectors", fontsize=16)
    
    if n_components == 1:
        axes = [axes]
        
    for i in range(n_components):
        grid = stack[i]
        max_abs = np.max(np.abs(grid))
        if max_abs == 0:
            max_abs = 1.0
        
        ax = axes[i]
        im = ax.matshow(grid, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
        ax.set_title(f"Component {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    return fig