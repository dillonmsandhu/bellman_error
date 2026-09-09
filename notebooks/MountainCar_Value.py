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

    # Ensure repository root is on sys.path when running from notebooks/ or root
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
    import pandas as pd
    import marimo as mo

    from envs.mountaincar_exact import MountainCarExactValue
    from envs.fourrooms import FourRoomsExactValue
    from core.config import config

    return (
        FourRoomsExactValue,
        LogNorm,
        MountainCarExactValue,
        jnp,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # Mountain Car: Value Function & Stationary Distribution Analysis

    This notebook analyzes the **discretized Mountain Car** environment:
    1. **Value Function ($V^\pi$) & State Distribution ($\mu$)** under a uniform **Random Policy**.
    2. **Goal Reachability Metric**: The undiscounted stationary distribution probability in the goal region $\mu(	ext{goal})$ and its associated mean recurrence / hitting time $\mathbb{E}[T_{	ext{reach}}] = 1 / \mu(	ext{goal})$.
    3. **Comparison to Four Rooms**: Contrasting reaching a goal via 2D diffusive Brownian motion vs. escaping a gravitational potential well in phase space.
    """)
    return


@app.cell
def _(mo):
    resolution_dropdown = mo.ui.dropdown(
        options={
            "32 x 32 (1,024 states — Fastest)": 32,
            "50 x 50 (2,500 states — Balanced, Recommended)": 50,
            "64 x 64 (4,096 states — High Res)": 64,
            "80 x 80 (6,400 states — Fine Grid)": 80,
        },
        value="50 x 50 (2,500 states — Balanced, Recommended)",
        label="Discretization Grid Resolution",
    )

    policy_selector = mo.ui.dropdown(
        options={
            "Uniform Random Policy": "random",
            "Energy-Pumping Heuristic Policy": "heuristic",
        },
        value="Uniform Random Policy",
        label="Policy Type",
    )

    gamma_slider = mo.ui.slider(
        start=0.80,
        stop=0.999,
        step=0.01,
        value=0.99,
        label="Discount Factor (γ)",
    )

    fourrooms_fail_prob = mo.ui.slider(
        start=0.0,
        stop=0.5,
        step=0.05,
        value=0.0,
        label="Four Rooms Slippage / Fail Prob",
    )

    v_cmap_dropdown = mo.ui.dropdown(
        options=["viridis", "plasma", "inferno", "coolwarm", "cividis"],
        value="viridis",
        label="Value Colormap",
    )

    mu_cmap_dropdown = mo.ui.dropdown(
        options=["magma", "plasma", "inferno", "hot", "viridis"],
        value="magma",
        label="Stationary Dist Colormap",
    )

    mu_log_toggle = mo.ui.checkbox(
        value=False,
        label="Use Log Scale for Stationary Distribution (μ)",
    )

    mo.hstack(
        [
            mo.vstack([resolution_dropdown, policy_selector]),
            mo.vstack([gamma_slider, fourrooms_fail_prob]),
            mo.vstack([v_cmap_dropdown, mu_cmap_dropdown, mu_log_toggle]),
        ],
        justify="start",
        gap=2,
    )
    return (
        fourrooms_fail_prob,
        gamma_slider,
        mu_cmap_dropdown,
        mu_log_toggle,
        policy_selector,
        resolution_dropdown,
        v_cmap_dropdown,
    )


@app.cell
def _(
    FourRoomsExactValue,
    MountainCarExactValue,
    fourrooms_fail_prob,
    gamma_slider,
    jnp,
    np,
    policy_selector,
    resolution_dropdown,
):
    # ---------------------------------------------------------------------------
    # 1. Mountain Car Environment & Evaluation
    # ---------------------------------------------------------------------------
    n_res = resolution_dropdown.value
    gamma = gamma_slider.value
    pol_type = policy_selector.value

    mc_env = MountainCarExactValue(n_pos=n_res, n_vel=n_res, gamma=gamma)

    if pol_type == "heuristic":
        # Heuristic energy-pumping policy
        pi_mc_active = np.zeros((mc_env.num_states, mc_env.num_actions), dtype=np.float32)
        for s_idx in range(mc_env.num_states):
            pos, vel = mc_env.coords[s_idx]
            if vel > 0:
                pi_mc_active[s_idx, 2] = 1.0  # Accelerate Right
            elif vel < 0:
                pi_mc_active[s_idx, 0] = 1.0  # Accelerate Left
            else:
                if pos < -0.5:
                    pi_mc_active[s_idx, 0] = 1.0
                else:
                    pi_mc_active[s_idx, 2] = 1.0
        pi_mc_active = jnp.asarray(pi_mc_active)
    else:
        # Uniform random policy
        pi_mc_active = jnp.ones((mc_env.num_states, mc_env.num_actions)) / mc_env.num_actions

    # Full policy including terminal state
    pi_mc_full = jnp.vstack(
        [pi_mc_active, jnp.ones((1, mc_env.num_actions)) / mc_env.num_actions]
    )

    # Compute Mountain Car Value Function and Distributions
    V_mc_grid = mc_env.compute_true_values(pi_mc_full)
    mu_mc_vec, P_mc_pi = mc_env.compute_stationary_distribution_raw(pi_mc_active)
    mu_mc_grid = mc_env.get_value_grid(mu_mc_vec)

    # Goal definition in Mountain Car: pos >= 0.5 and vel >= 0.0
    mc_goal_mask = (mc_env.coords[:, 0] >= mc_env.goal_position) & (
        mc_env.coords[:, 1] >= mc_env.goal_velocity
    )
    p_goal_mc = float(jnp.sum(mu_mc_vec[mc_goal_mask]))
    hitting_time_mc = (1.0 / p_goal_mc) if p_goal_mc > 0 else float("inf")

    # Discounted state visitation for Mountain Car
    d_mc_vec = mc_env.compute_discounted_visitation_raw(pi_mc_active)
    d_mc_grid = mc_env.get_value_grid(d_mc_vec)
    d_goal_mc = float(jnp.sum(d_mc_vec[mc_goal_mask]))

    # ---------------------------------------------------------------------------
    # 2. Four Rooms Environment & Evaluation (Random Policy Reference)
    # ---------------------------------------------------------------------------
    fr_fail_p = fourrooms_fail_prob.value
    fr_env = FourRoomsExactValue(
        start_pos=(3, 1),
        goal_pos=(11, 11),
        fail_prob=fr_fail_p,
        gamma=gamma,
    )
    pi_fr_active = jnp.ones((fr_env.num_states, fr_env.num_actions)) / fr_env.num_actions
    pi_fr_full = jnp.vstack(
        [pi_fr_active, jnp.ones((1, fr_env.num_actions)) / fr_env.num_actions]
    )

    V_fr_grid = fr_env.compute_true_values(pi_fr_full)
    mu_fr_vec, P_fr_pi = fr_env.compute_stationary_distribution_raw(pi_fr_active)
    mu_fr_grid = fr_env.get_value_grid(mu_fr_vec)

    p_goal_fr = float(mu_fr_vec[fr_env.goal_idx])
    hitting_time_fr = (1.0 / p_goal_fr) if p_goal_fr > 0 else float("inf")

    d_fr_vec = fr_env.compute_discounted_visitation_raw(pi_fr_active)
    d_goal_fr = float(d_fr_vec[fr_env.goal_idx])
    return (
        P_fr_pi,
        P_mc_pi,
        V_fr_grid,
        V_mc_grid,
        d_goal_fr,
        d_goal_mc,
        fr_env,
        gamma,
        hitting_time_fr,
        hitting_time_mc,
        mc_env,
        mc_goal_mask,
        mu_fr_grid,
        mu_fr_vec,
        mu_mc_grid,
        mu_mc_vec,
        p_goal_fr,
        p_goal_mc,
        pol_type,
    )


@app.cell
def _(hitting_time_fr, hitting_time_mc, mo, p_goal_fr, p_goal_mc, pol_type):
    # Ratio calculations
    _ratio_p = p_goal_fr / p_goal_mc if p_goal_mc > 0 else float("inf")
    _ratio_time = hitting_time_mc / hitting_time_fr if hitting_time_fr > 0 else float("inf")

    _pol_title = (
        "Random Policy" if pol_type == "random" else "Energy-Pumping Heuristic"
    )

    _card_mc = mo.stat(
        value=f"{p_goal_mc:.4e}",
        caption=f"{p_goal_mc * 100:.4f}% | Mean Hitting Time: ~{hitting_time_mc:,.1f} steps",
        label=f"Mountain Car ({_pol_title}) Goal Prob μ(goal)",
    )

    _card_fr = mo.stat(
        value=f"{p_goal_fr:.4e}",
        caption=f"{p_goal_fr * 100:.4f}% | Mean Hitting Time: ~{hitting_time_fr:,.1f} steps",
        label="Four Rooms (Random Policy) Goal Prob μ(goal)",
    )

    _card_ratio = mo.stat(
        value=f"{_ratio_p:.2f}x",
        caption=f"Four Rooms reaches goal {_ratio_p:.2f}x more frequently per step",
        label="Goal Reachability Ratio (Four Rooms / Mountain Car)",
    )

    mo.vstack(
        [
            mo.md("## 1. Goal Reachability: Undiscounted Stationary Distribution in Goal"),
            mo.hstack([_card_mc, _card_fr, _card_ratio], justify="space-between"),
        ]
    )
    return


@app.cell
def _(
    d_goal_fr,
    d_goal_mc,
    fr_env,
    hitting_time_fr,
    hitting_time_mc,
    mc_env,
    mc_goal_mask,
    mo,
    np,
    p_goal_fr,
    p_goal_mc,
    pd,
):
    _table_data = [
        {
            "Metric": "Stationary Prob in Goal μ(goal)",
            "Mountain Car": f"{p_goal_mc:.6e} ({p_goal_mc * 100:.4f}%)",
            "Four Rooms": f"{p_goal_fr:.6e} ({p_goal_fr * 100:.4f}%)",
            "Comparison (FR vs MC)": f"Four Rooms is {p_goal_fr / p_goal_mc:.2f}x higher" if p_goal_mc > 0 else "N/A",
        },
        {
            "Metric": "Mean Hitting / Recurrence Time E[T_reach]",
            "Mountain Car": f"~{hitting_time_mc:,.1f} steps",
            "Four Rooms": f"~{hitting_time_fr:,.1f} steps",
            "Comparison (FR vs MC)": f"Mountain Car takes {hitting_time_mc / hitting_time_fr:.2f}x longer" if hitting_time_fr > 0 else "N/A",
        },
        {
            "Metric": "Discounted Goal Visitation d_γ(goal)",
            "Mountain Car": f"{d_goal_mc:.6e}",
            "Four Rooms": f"{d_goal_fr:.6e}",
            "Comparison (FR vs MC)": f"Four Rooms is {d_goal_fr / d_goal_mc:.2f}x higher" if d_goal_mc > 0 else "N/A",
        },
        {
            "Metric": "Goal Region Specification",
            "Mountain Car": f"pos ≥ {mc_env.goal_position}, vel ≥ {mc_env.goal_velocity} ({np.sum(mc_goal_mask)} bins)",
            "Four Rooms": f"Single cell (y=11, x=11)",
            "Comparison (FR vs MC)": "Region vs Single Point",
        },
        {
            "Metric": "State Space Structure",
            "Mountain Car": f"Continuous 2D Phase Space ({mc_env.n_pos}x{mc_env.n_vel} = {mc_env.num_states} bins)",
            "Four Rooms": f"Discrete 11x11 Grid ({fr_env.num_states} walkable states)",
            "Comparison (FR vs MC)": "Phase Space vs Spatial Maze",
        },
        {
            "Metric": "Dynamics / Motion Regime",
            "Mountain Car": "Underactuated inertia in a gravity well (restoring force -g·cos(3x))",
            "Four Rooms": "Purely diffusive random walk / bounded Brownian motion",
            "Comparison (FR vs MC)": "Potential barrier vs Geometric search",
        },
    ]

    _df = pd.DataFrame(_table_data)
    _table_ui = mo.ui.table(_df, selection=None)

    _explanation_md = mo.md(
        """
        > [!NOTE]
        > **Why does the stationary distribution probability $\\mu(\\text{goal})$ measure goal reachability?**
        > 
        > In the continuing renewal formulation used by both `FourRoomsExactValue` and `MountainCarExactValue`, 
        > entering the goal transitions back to the start state $s_0$ with reward $+1$.
        > 
        > By **Kac's Lemma / The Renewal-Reward Theorem** for ergodic Markov chains:
        > $$\\mu(\\text{goal}) = \\frac{\\mathbb{E}[\\text{time in goal state per cycle}]}{\\mathbb{E}[\\text{cycle duration}]} = \\frac{1}{\\mathbb{E}[T_{\\text{reach}}]}$$
        > 
        > where $\\mathbb{E}[T_{\\text{reach}}]$ is the expected number of steps for the policy to travel from the start state to the goal.
        > 
        > **Intuition behind the difference:**
        > - In **Four Rooms**, a uniform random policy executes unbiased Brownian motion across 104 discrete walkable cells. Bounded diffusion in 2D explores the space effectively and locates the goal corner in ~1,141 steps.
        > - In **Mountain Car**, a random policy takes actions uniformly in $\\{-1, 0, +1\\}$. The engine thrust is too weak to climb the hill directly; reaching the goal requires coordinated momentum buildup. Gravity acts as a restoring force pulling the car back toward $x = -0.5$. Because momentum fluctuates with zero mean, escaping the gravitational potential well requires a rare deviation, requiring thousands of steps!
        """
    )

    mo.vstack([_table_ui, _explanation_md])
    return


@app.cell
def _(hitting_time_fr, hitting_time_mc, p_goal_fr, p_goal_mc, plt):
    # Bar plot comparison of goal probability and expected hitting time
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    _envs = ["Mountain Car\n(Random Policy)", "Four Rooms\n(Random Policy)"]
    _probs = [p_goal_mc, p_goal_fr]
    _times = [hitting_time_mc, hitting_time_fr]
    _colors = ["#1f77b4", "#d62728"]

    # 1. Goal Probability
    _bars1 = _ax1.bar(_envs, _probs, color=_colors, alpha=0.85, edgecolor="black", width=0.5)
    _ax1.set_ylabel("Stationary Probability $\\mu(\\mathrm{goal})$ (log scale)")
    _ax1.set_yscale("log")
    _ax1.set_title("Goal State Probability $\\mu(\\mathrm{goal})$")
    _ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for _bar, _val in zip(_bars1, _probs):
        _y = _bar.get_height()
        _ax1.text(
            _bar.get_x() + _bar.get_width() / 2.0,
            _y * 1.15,
            f"{_val:.2e}\n({_val * 100:.4f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # 2. Expected Hitting Time
    _bars2 = _ax2.bar(_envs, _times, color=_colors, alpha=0.85, edgecolor="black", width=0.5)
    _ax2.set_ylabel("Expected Steps $\\mathbb{E}[T_{\\mathrm{reach}}] = 1 / \\mu(\\mathrm{goal})$ (log scale)")
    _ax2.set_yscale("log")
    _ax2.set_title("Expected Hitting Time to Goal (Steps)")
    _ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for _bar, _val in zip(_bars2, _times):
        _y = _bar.get_height()
        _ax2.text(
            _bar.get_x() + _bar.get_width() / 2.0,
            _y * 1.15,
            f"~{_val:,.1f} steps",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    LogNorm,
    V_mc_grid,
    jnp,
    mc_env,
    mu_cmap_dropdown,
    mu_log_toggle,
    mu_mc_grid,
    plt,
    pol_type,
    v_cmap_dropdown,
):
    # ---------------------------------------------------------------------------
    # 2D Heatmaps: Mountain Car Value Function & Stationary Distribution
    # ---------------------------------------------------------------------------
    _fig, (_ax_v, _ax_mu) = plt.subplots(1, 2, figsize=(14, 5.5))

    _X, _Y = jnp.meshgrid(mc_env.pos_bins, mc_env.vel_bins, indexing="ij")

    # 1. Value Function Plot
    _mesh_v = _ax_v.pcolormesh(
        _X,
        _Y,
        V_mc_grid,
        cmap=v_cmap_dropdown.value,
        shading="auto",
    )
    _cbar_v = _fig.colorbar(_mesh_v, ax=_ax_v, fraction=0.046, pad=0.04)
    _cbar_v.set_label("True Value $V^\\pi(s)$", fontsize=11)

    # Mark Start Position (pos=-0.5, vel=0.0)
    _ax_v.plot(-0.5, 0.0, marker="o", markersize=9, color="cyan", markeredgecolor="black", label="Start (x=-0.5, v=0)")
    _ax_v.text(-0.5, 0.008, "S", color="white", fontweight="bold", ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    # Mark Goal Region Boundary (pos >= 0.5, vel >= 0.0)
    _ax_v.axvline(x=mc_env.goal_position, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Goal Pos (x ≥ 0.5)")
    _ax_v.axhline(y=mc_env.goal_velocity, color="orange", linestyle=":", linewidth=1.5, alpha=0.8, label="Goal Vel (v ≥ 0.0)")
    _ax_v.fill_between([mc_env.goal_position, mc_env.max_position], [0, 0], [mc_env.max_speed, mc_env.max_speed], color="red", alpha=0.15)
    _ax_v.text(0.55, 0.035, "GOAL\nREGION", color="red", fontweight="bold", ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    _pol_label = "Random Policy" if pol_type == "random" else "Energy-Pumping Heuristic"
    _ax_v.set_title(f"Mountain Car True Value Function $V^\\pi$\n({_pol_label})", fontsize=12)
    _ax_v.set_xlabel("Position", fontsize=11)
    _ax_v.set_ylabel("Velocity", fontsize=11)
    _ax_v.legend(loc="lower left", fontsize=8.5)

    # 2. Stationary Distribution Plot
    if mu_log_toggle.value:
        _min_val = float(jnp.min(mu_mc_grid[mu_mc_grid > 0])) if jnp.any(mu_mc_grid > 0) else 1e-7
        _max_val = float(jnp.max(mu_mc_grid))
        _norm = LogNorm(vmin=max(_min_val, 1e-8), vmax=_max_val)
        _mesh_mu = _ax_mu.pcolormesh(
            _X,
            _Y,
            mu_mc_grid,
            cmap=mu_cmap_dropdown.value,
            shading="auto",
            norm=_norm,
        )
    else:
        _mesh_mu = _ax_mu.pcolormesh(
            _X,
            _Y,
            mu_mc_grid,
            cmap=mu_cmap_dropdown.value,
            shading="auto",
        )

    _cbar_mu = _fig.colorbar(_mesh_mu, ax=_ax_mu, fraction=0.046, pad=0.04)
    _cbar_mu.set_label("Stationary Density $\\mu(s)$" + (" (Log Scale)" if mu_log_toggle.value else ""), fontsize=11)

    _ax_mu.plot(-0.5, 0.0, marker="o", markersize=9, color="cyan", markeredgecolor="black", label="Start (x=-0.5, v=0)")
    _ax_mu.axvline(x=mc_env.goal_position, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Goal Pos (x ≥ 0.5)")
    _ax_mu.axhline(y=mc_env.goal_velocity, color="orange", linestyle=":", linewidth=1.5, alpha=0.8)
    _ax_mu.fill_between([mc_env.goal_position, mc_env.max_position], [0, 0], [mc_env.max_speed, mc_env.max_speed], color="red", alpha=0.15)

    _ax_mu.set_title(f"Stationary Distribution $\\mu(s)$\n(Concentrated in Potential Well Bottom)", fontsize=12)
    _ax_mu.set_xlabel("Position", fontsize=11)
    _ax_mu.set_ylabel("Velocity", fontsize=11)
    _ax_mu.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Interactive 3D Surface Visualizations
    Use the sliders below to rotate the 3D surface view of the Value Function and Stationary Distribution.
    Notice the archetypal **phase-space swirl / vortex** of $V^\pi(x, v)$ leading toward the goal region.
    """)
    return


@app.cell
def _(mo):
    elev_slider = mo.ui.slider(0, 90, step=5, value=35, label="3D Elevation (Pitch)")
    azim_slider = mo.ui.slider(-180, 180, step=5, value=-55, label="3D Azimuth (Yaw)")
    surface_view_mode = mo.ui.radio(
        options=["Value Function (V^π)", "Stationary Distribution (μ)", "Both Side-by-Side"],
        value="Both Side-by-Side",
        label="Surface to Display",
    )

    mo.hstack([elev_slider, azim_slider, surface_view_mode], justify="start", gap=2)
    return azim_slider, elev_slider, surface_view_mode


@app.cell
def _(
    V_mc_grid,
    azim_slider,
    elev_slider,
    jnp,
    mc_env,
    mu_cmap_dropdown,
    mu_mc_grid,
    np,
    plt,
    surface_view_mode,
    v_cmap_dropdown,
):
    _mode = surface_view_mode.value
    _elev = elev_slider.value
    _azim = azim_slider.value

    _X, _Y = jnp.meshgrid(mc_env.pos_bins, mc_env.vel_bins, indexing="ij")
    _X_np = np.asarray(_X)
    _Y_np = np.asarray(_Y)
    _V_np = np.asarray(V_mc_grid)
    _mu_np = np.asarray(mu_mc_grid)

    if _mode == "Both Side-by-Side":
        _fig = plt.figure(figsize=(15, 6))
        _ax1 = _fig.add_subplot(1, 2, 1, projection="3d")
        _ax2 = _fig.add_subplot(1, 2, 2, projection="3d")

        # 3D Value Function
        _surf1 = _ax1.plot_surface(
            _X_np,
            _Y_np,
            _V_np,
            cmap=v_cmap_dropdown.value,
            edgecolor="none",
            antialiased=True,
            rstride=1,
            cstride=1,
            alpha=0.9,
        )
        _ax1.view_init(elev=_elev, azim=_azim)
        _ax1.set_xlabel("Position", labelpad=8)
        _ax1.set_ylabel("Velocity", labelpad=8)
        _ax1.set_zlabel("Value $V^\\pi$", labelpad=8)
        _ax1.set_title("3D Surface: Value Function $V^\\pi(x, v)$", fontsize=11)
        _fig.colorbar(_surf1, ax=_ax1, shrink=0.5, aspect=12, pad=0.1)

        # 3D Stationary Distribution
        _surf2 = _ax2.plot_surface(
            _X_np,
            _Y_np,
            _mu_np,
            cmap=mu_cmap_dropdown.value,
            edgecolor="none",
            antialiased=True,
            rstride=1,
            cstride=1,
            alpha=0.9,
        )
        _ax2.view_init(elev=_elev, azim=_azim)
        _ax2.set_xlabel("Position", labelpad=8)
        _ax2.set_ylabel("Velocity", labelpad=8)
        _ax2.set_zlabel("Density $\\mu$", labelpad=8)
        _ax2.set_title("3D Surface: Stationary Distribution $\\mu(x, v)$", fontsize=11)
        _fig.colorbar(_surf2, ax=_ax2, shrink=0.5, aspect=12, pad=0.1)

    elif _mode == "Value Function (V^π)":
        _fig = plt.figure(figsize=(9, 6))
        _ax = _fig.add_subplot(1, 1, 1, projection="3d")
        _surf = _ax.plot_surface(
            _X_np,
            _Y_np,
            _V_np,
            cmap=v_cmap_dropdown.value,
            edgecolor="none",
            antialiased=True,
            alpha=0.9,
        )
        _ax.view_init(elev=_elev, azim=_azim)
        _ax.set_xlabel("Position", labelpad=8)
        _ax.set_ylabel("Velocity", labelpad=8)
        _ax.set_zlabel("Value $V^\\pi$", labelpad=8)
        _ax.set_title("3D Surface: Mountain Car Value Function $V^\\pi(x, v)$", fontsize=12)
        _fig.colorbar(_surf, ax=_ax, shrink=0.6, aspect=15)

    else:
        _fig = plt.figure(figsize=(9, 6))
        _ax = _fig.add_subplot(1, 1, 1, projection="3d")
        _surf = _ax.plot_surface(
            _X_np,
            _Y_np,
            _mu_np,
            cmap=mu_cmap_dropdown.value,
            edgecolor="none",
            antialiased=True,
            alpha=0.9,
        )
        _ax.view_init(elev=_elev, azim=_azim)
        _ax.set_xlabel("Position", labelpad=8)
        _ax.set_ylabel("Velocity", labelpad=8)
        _ax.set_zlabel("Density $\\mu$", labelpad=8)
        _ax.set_title("3D Surface: Stationary Distribution $\\mu(x, v)$", fontsize=12)
        _fig.colorbar(_surf, ax=_ax, shrink=0.6, aspect=15)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Four Rooms Ground Truth Reference
    For direct visual comparison, below are the **True Value Function ($V^\pi$)** and **Stationary Distribution ($\mu$)** for Four Rooms under a uniform random policy.
    """)
    return


@app.cell
def _(V_fr_grid, fr_env, jnp, mu_fr_grid, plt):
    # Four Rooms Visual Reference
    _fig_fr, (_ax_fr_v, _ax_fr_mu) = plt.subplots(1, 2, figsize=(11, 4.8))

    _mask = fr_env.occupied_map
    _masked_v = jnp.where(_mask == 1, jnp.nan, V_fr_grid)
    _masked_mu = jnp.where(_mask == 1, jnp.nan, mu_fr_grid)

    # 1. Four Rooms Value Function
    _ax_fr_v.imshow(_mask, cmap="Greys", alpha=0.25)
    _im_v = _ax_fr_v.imshow(_masked_v, cmap="viridis")
    _fig_fr.colorbar(_im_v, ax=_ax_fr_v, fraction=0.046, pad=0.04, label="Value $V^\\pi$")
    _ax_fr_v.text(fr_env.start[1], fr_env.start[0], "S", color="blue", fontweight="bold", fontsize=14, ha="center", va="center")
    _ax_fr_v.text(fr_env.goal[1], fr_env.goal[0], "G", color="red", fontweight="bold", fontsize=14, ha="center", va="center")
    _ax_fr_v.set_title("Four Rooms: True Value $V^\\pi$\n(Random Policy)", fontsize=11)
    _ax_fr_v.axis("off")

    # 2. Four Rooms Stationary Distribution
    _ax_fr_mu.imshow(_mask, cmap="Greys", alpha=0.25)
    _im_mu = _ax_fr_mu.imshow(_masked_mu, cmap="magma")
    _fig_fr.colorbar(_im_mu, ax=_ax_fr_mu, fraction=0.046, pad=0.04, label="Stationary Dist $\\mu$")
    _ax_fr_mu.text(fr_env.start[1], fr_env.start[0], "S", color="blue", fontweight="bold", fontsize=14, ha="center", va="center")
    _ax_fr_mu.text(fr_env.goal[1], fr_env.goal[0], "G", color="red", fontweight="bold", fontsize=14, ha="center", va="center")
    _ax_fr_mu.set_title("Four Rooms: Stationary Dist $\\mu$\n(Diffusive Flow from Goal to Start)", fontsize=11)
    _ax_fr_mu.axis("off")

    plt.tight_layout()
    _fig_fr
    return


@app.cell
def _(P_fr_pi, P_mc_pi, gamma, jnp, mo, mu_fr_vec, mu_mc_vec, pd):
    # ---------------------------------------------------------------------------
    # Spectral Asymmetry Matrices (A, S, K)
    # ---------------------------------------------------------------------------
    def compute_ASK(mu, P_pi, g):
        D = jnp.diag(mu)
        I = jnp.eye(D.shape[0])
        A = D @ (I - g * P_pi)
        S = 0.5 * (A + A.T)
        K = 0.5 * (A - A.T)
        return A, S, K

    A_mc, S_mc, K_mc = compute_ASK(mu_mc_vec, P_mc_pi, gamma)
    A_fr, S_fr, K_fr = compute_ASK(mu_fr_vec, P_fr_pi, gamma)

    # 1. Relative Spectral Norm (||K||_2 / ||S||_2)
    norm_S_mc = jnp.max(jnp.abs(jnp.linalg.eigvalsh(S_mc)))
    norm_K_mc = jnp.max(jnp.abs(jnp.linalg.eigvalsh(1j * K_mc)))

    norm_S_fr = jnp.max(jnp.abs(jnp.linalg.eigvalsh(S_fr)))
    norm_K_fr = jnp.max(jnp.abs(jnp.linalg.eigvalsh(1j * K_fr)))

    spectral_ratio_mc = float(norm_K_mc / (norm_S_mc + 1e-12))
    spectral_ratio_fr = float(norm_K_fr / (norm_S_fr + 1e-12))

    # 2. Spectral Radius of K
    rad_K_mc = float(norm_K_mc)
    rad_K_fr = float(norm_K_fr)

    # 3. Worst-case Alignment Eigenvalue
    def get_worst_alignment(S, K):
        SA_sym = S @ S + 0.5 * (S @ K - K @ S)
        return jnp.min(jnp.linalg.eigvalsh(SA_sym))

    worst_align_mc = float(get_worst_alignment(S_mc, K_mc))
    worst_align_fr = float(get_worst_alignment(S_fr, K_fr))

    spectral_table = pd.DataFrame([
        {
            "Metric": "Relative Spectral Norm (||K||_2 / ||S||_2)",
            "Mountain Car": f"{spectral_ratio_mc:.4f}",
            "Four Rooms": f"{spectral_ratio_fr:.4f}",
            "Comparison": f"Mountain Car is {spectral_ratio_mc / max(spectral_ratio_fr, 1e-12):.1f}x more non-reversible",
        },
        {
            "Metric": "Spectral Radius of K (||K||_2)",
            "Mountain Car": f"{rad_K_mc:.6f}",
            "Four Rooms": f"{rad_K_fr:.6f}",
            "Comparison": "Magnitude of skew-symmetric flows",
        },
        {
            "Metric": "Worst-case Alignment λ_min(S² + SK)",
            "Mountain Car": f"{worst_align_mc:.6e}",
            "Four Rooms": f"{worst_align_fr:.6e}",
            "Comparison": "Negative means TD might increase error",
        },
    ])

    spectral_ui = mo.ui.table(spectral_table, selection=None)

    mo_spectral = mo.md(
        """
        ## 4. Spectral Analysis of Asymmetry
        We can decompose the key matrix $A = D(I - \\gamma P_\\pi)$ into its symmetric and skew-symmetric components:
        - $S = \\frac{1}{2}(A + A^T)$: The **reversible** probability flow.
        - $K = \\frac{1}{2}(A - A^T)$: The **non-reversible** (skew-symmetric) probability flow.

        The relative magnitude of $K$ compared to $S$ characterizes the asymmetry of the Markov chain.

        > [!NOTE]
        > **Relative Spectral Norm ($||K||_2 / ||S||_2$)**: Measures the maximum non-reversible flow relative to the maximum reversible flow. A purely reversible chain (like uniform random walk on an undirected graph) has $K=0$, so this is $0$.
        > 
        > **Worst-case Alignment Eigenvalue**: Tang and Munos (2023) show that value error is strictly decreased if $S^2 + \\frac{1}{2}(SK - KS)$ is positive definite. The minimum eigenvalue $\\lambda_{\\text{min}}$ of this matrix measures the margin of safety for the TD update. If it's negative, there exist error vectors where TD increases the value error!
        """
    )
    return mo_spectral, spectral_ui


@app.cell
def _(mo, mo_spectral, spectral_ui):
    mo.vstack([mo_spectral, spectral_ui])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
