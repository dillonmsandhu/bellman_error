# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "jax",
#     "jaxlib",
#     "gymnax",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from envs.fourrooms import FourRoomsExactValue
    from core.config import config 
    import gymnax 
    import jax.numpy as jnp
    import jax
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from core.helpers import initialize_evaluator, make_env
    from core.utils import load_run_data
    import marimo as mo
    from core.mail import email_pdf

    td_config, td_metrics = load_run_data('random/td_exact/tuned_saved_metrics/', 'FourRooms-misc', 'results')
    mc_config, mc_metrics = load_run_data('random/mc_exact/tuned_saved_metrics/', 'FourRooms-misc', 'results')

    env, env_params = make_env(td_config)
    evaluator = initialize_evaluator(td_config, env, env_params)

    def random_policy(_):
        m = evaluator.num_actions
        return jnp.ones(m) / m


    return (
        email_pdf,
        evaluator,
        jax,
        jnp,
        mc_metrics,
        mo,
        np,
        plt,
        random_policy,
        td_metrics,
    )


@app.cell
def _(jnp, plt):
    def plot_grid(grid, evaluator, title, cmap="viridis", logscale=False):
        plt.figure(figsize=(6, 6))

        # Create a mask for walls (where occupied_map is 1)
        # evaluator.occupied_map is 1 for walls, 0 for free space
        mask = evaluator.occupied_map

        # Plot background walls in gray
        plt.imshow(mask, cmap="Greys", alpha=0.3)

        # Mask the grid values where there are walls to avoid plotting 0s in wall cells
        masked_grid = jnp.where(mask == 1, jnp.nan, grid)

        # Display the values
        if logscale:
            im = plt.imshow(masked_grid, cmap=cmap, norm='log')
        else:    
            im = plt.imshow(masked_grid, cmap=cmap)

        plt.colorbar(im, fraction=0.046, pad=0.04)

        # Annotate Start and Goal
        # evaluator.start and evaluator.goal are [y, x] which corresponds to [row, col]
        start_y, start_x = evaluator.start
        goal_y, goal_x = evaluator.goal

        # Text labels for Start and Goal
        plt.text(start_x, start_y, 'S', ha='center', va='center', color='blue', fontweight='bold', fontsize=15)
        plt.text(goal_x, goal_y, 'G', ha='center', va='center', color='red', fontweight='bold', fontsize=15)

        plt.title(title)
        plt.axis("off")
        return plt.gca()

    return


@app.cell
def _(evaluator, jax, jnp, random_policy):
    target_policy_probs = jax.vmap(random_policy)(evaluator.obs_stack)
    pi = jnp.vstack([target_policy_probs, target_policy_probs[0]])
    V_pi = evaluator.compute_true_values_raw(pi)
    mu_vector, _ = evaluator.compute_stationary_distribution_raw(target_policy_probs)
    mu_grid = evaluator.get_value_grid(mu_vector)
    return V_pi, target_policy_probs


@app.cell
def _(V_pi, evaluator):
    v_grid = evaluator.get_value_grid(V_pi[:-1])
    # plot_grid(v_grid, evaluator, "True Value Function ($V^\pi$)", cmap="viridis")
    return


@app.cell
def _(V_pi, evaluator, jnp, plt, target_policy_probs):
    def plot_combined(
        grid1,
        grid2,
        evaluator,
        title1="Title 1",
        title2="Title 2",
        cmap1="magma",
        cmap2="viridis",
        use_log=False,
        shared_vrange=True,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        mask = evaluator.occupied_map
        start_y, start_x = evaluator.start
        goal_y, goal_x = evaluator.goal

        g1_masked = jnp.where(mask == 1, jnp.nan, grid1)
        g2_masked = jnp.where(mask == 1, jnp.nan, grid2)

        # Calculate vmin/vmax for shared scale
        if shared_vrange:
            vmin = min(jnp.nanmin(g1_masked), jnp.nanmin(g2_masked))
            vmax = max(jnp.nanmax(g1_masked), jnp.nanmax(g2_masked))
        else:
            vmin = vmax = None

        from matplotlib.colors import LogNorm

        norm = LogNorm(vmin=max(1e-10, vmin) if vmin is not None else None, vmax=vmax) if use_log else None
        if use_log and not shared_vrange:
             # If not shared, we'll let imshow handle norm per axis if we don't pass vmin/vmax
             norm1 = LogNorm(vmin=max(1e-10, jnp.nanmin(g1_masked)), vmax=jnp.nanmax(g1_masked))
             norm2 = LogNorm(vmin=max(1e-10, jnp.nanmin(g2_masked)), vmax=jnp.nanmax(g2_masked))
        else:
             norm1 = norm2 = norm

        # Plot first grid
        axes[0].imshow(mask, cmap="Greys", alpha=0.3)
        im0 = axes[0].imshow(g1_masked, cmap=cmap1, norm=norm1, vmin=None if use_log else vmin, vmax=None if use_log else vmax)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].text(start_x, start_y, "S", ha="center", va="center", color="blue", fontweight="bold")
        axes[0].text(goal_x, goal_y, "G", ha="center", va="center", color="red", fontweight="bold")
        axes[0].set_title(title1)
        axes[0].axis("off")

        # Plot second grid
        axes[1].imshow(mask, cmap="Greys", alpha=0.3)
        im1 = axes[1].imshow(g2_masked, cmap=cmap2, norm=norm2, vmin=None if use_log else vmin, vmax=None if use_log else vmax)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].text(start_x, start_y, "S", ha="center", va="center", color="blue", fontweight="bold")
        axes[1].text(goal_x, goal_y, "G", ha="center", va="center", color="red", fontweight="bold")
        axes[1].set_title(title2)
        axes[1].axis("off")

        plt.tight_layout()
        return fig

    mu_vec, _ = evaluator.compute_stationary_distribution_raw(target_policy_probs)
    mu_g = evaluator.get_value_grid(mu_vec)
    v_g = evaluator.get_value_grid(V_pi[:-1])
    return (v_g,)


@app.cell
def _(email_pdf, evaluator, jnp, mc_metrics, plt, td_metrics, v_g):
    def plot_combined_errs(
        grid1,
        grid2,
        evaluator,
        title1="Title 1",
        title2="Title 2",
        cmap1="magma",
        cmap2="magma",
        use_log=False,
        shared_vrange=True,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        mask = evaluator.occupied_map
        start_y, start_x = evaluator.start
        goal_y, goal_x = evaluator.goal

        g1_masked = jnp.where(mask == 1, jnp.nan, grid1)
        g2_masked = jnp.where(mask == 1, jnp.nan, grid2)

        # Calculate vmin/vmax for shared scale
        if shared_vrange:
            vmin = min(jnp.nanmin(g1_masked), jnp.nanmin(g2_masked))
            vmax = max(jnp.nanmax(g1_masked), jnp.nanmax(g2_masked))
        else:
            vmin = vmax = None

        from matplotlib.colors import LogNorm

        norm = LogNorm(vmin=max(1e-10, vmin) if vmin is not None else None, vmax=vmax) if use_log else None
        if use_log and not shared_vrange:
             # If not shared, we'll let imshow handle norm per axis if we don't pass vmin/vmax
             norm1 = LogNorm(vmin=max(1e-10, jnp.nanmin(g1_masked)), vmax=jnp.nanmax(g1_masked))
             norm2 = LogNorm(vmin=max(1e-10, jnp.nanmin(g2_masked)), vmax=jnp.nanmax(g2_masked))
        else:
             norm1 = norm2 = norm

        # Plot first grid
        axes[0].imshow(mask, cmap="Greys", alpha=0.3)
        im0 = axes[0].imshow(g1_masked, cmap=cmap1, norm=norm1, vmin=None if use_log else vmin, vmax=None if use_log else vmax)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].text(start_x, start_y, "S", ha="center", va="center", color="blue", fontweight="bold")
        axes[0].text(goal_x, goal_y, "G", ha="center", va="center", color="red", fontweight="bold")
        axes[0].set_title(title1)
        axes[0].axis("off")

        # Plot second grid
        axes[1].imshow(mask, cmap="Greys", alpha=0.3)
        im1 = axes[1].imshow(g2_masked, cmap=cmap2, norm=norm2, vmin=None if use_log else vmin, vmax=None if use_log else vmax)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].text(start_x, start_y, "S", ha="center", va="center", color="blue", fontweight="bold")
        axes[1].text(goal_x, goal_y, "G", ha="center", va="center", color="red", fontweight="bold")
        axes[1].set_title(title2)
        axes[1].axis("off")

        plt.tight_layout()
        return fig

    mc_err = (evaluator.get_value_grid(mc_metrics["V_nn"][0, 1000]) - v_g) ** 2
    td_err = (evaluator.get_value_grid(td_metrics["V_nn"][0, 1000]) - v_g) ** 2

    fig = plot_combined_errs(td_err, mc_err, evaluator, title1 = 'TD', title2= 'Supervised', use_log=True)
    fig.savefig('figures/td_vs_mc_value_errors.pdf', bbox_inches='tight')
    email_pdf('figures/td_vs_mc_value_errors.pdf')
    return


@app.cell
def _(evaluator, mc_metrics, td_metrics):
    td_v_last = evaluator.get_value_grid(td_metrics['V_nn'][0,500])
    mc_v_last = evaluator.get_value_grid(mc_metrics['V_nn'][0,500])

    return mc_v_last, td_v_last


@app.cell
def _(mo):
    mo.md("""
    # Interactive 3D Analysis
    Use the sliders below to rotate the 3D view. The plot on the left compares the **True Value Surface** with the **Learned Value Wireframes**. The plot on the right shows the **Underestimation Error** ($V_{true} - V_{learned}$).
    """)
    return


@app.cell
def _(mo):
    elev_slider = mo.ui.slider(0, 90, step=5, value=30, label="Elevation")
    azim_slider = mo.ui.slider(-180, 180, step=5, value=-60, label="Azimuth (Rotation)")
    show_td = mo.ui.checkbox(value=True, label="Show TD")
    show_mc = mo.ui.checkbox(value=True, label="Show MC")
    return azim_slider, elev_slider, show_mc, show_td


@app.cell
def _(jnp, np, plt):
    def plot_3d_comparison(v_true, v_td, v_mc, evaluator, title="Value Function Comparison (3D Lines)"):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        mask = evaluator.occupied_map
        ny, nx = v_true.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        # Mask out walls by setting them to NaN
        vt = jnp.where(mask == 1, jnp.nan, v_true)
        vtd = jnp.where(mask == 1, jnp.nan, v_td)
        vmc = jnp.where(mask == 1, jnp.nan, v_mc)

        # Plot lines along rows
        for i in range(ny):
            ax.plot(X[i, :], Y[i, :], vt[i, :], color='black', alpha=0.5, linewidth=1.5, label='True' if i == 0 else "")
            ax.plot(X[i, :], Y[i, :], vtd[i, :], color='blue', alpha=0.6, linewidth=1.5, label='TD' if i == 0 else "")
            ax.plot(X[i, :], Y[i, :], vmc[i, :], color='red', alpha=0.6, linewidth=1.5, label='MC' if i == 0 else "")

        # Plot lines along columns
        for j in range(nx):
            ax.plot(X[:, j], Y[:, j], vt[:, j], color='black', alpha=0.2, linewidth=1)
            ax.plot(X[:, j], Y[:, j], vtd[:, j], color='blue', alpha=0.6, linewidth=1.5)
            ax.plot(X[:, j], Y[:, j], vmc[:, j], color='red', alpha=0.6, linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel('X (Col)')
        ax.set_ylabel('Y (Row)')
        ax.set_zlabel('Value')

        # Custom legend to avoid duplicates
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='black', lw=4, alpha=0.8),
                        Line2D([0], [0], color='blue', lw=1.5, alpha=0.8),
                        Line2D([0], [0], color='red', lw=1.5, alpha=0.8)]
        ax.legend(custom_lines, ['True', 'TD', 'MC'])

        return fig

    return


@app.cell
def _(
    azim_slider,
    elev_slider,
    evaluator,
    jnp,
    mc_v_last,
    mo,
    np,
    plt,
    show_mc,
    show_td,
    td_v_last,
    v_g,
):
    def render_3d(v_true, v_td, v_mc, evaluator, elev, azim, show_td, show_mc):
        fig = plt.figure(figsize=(14, 7))
        mask = evaluator.occupied_map
        ny, nx = v_true.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
        vt = jnp.where(mask == 1, jnp.nan, v_true)
        vtd = jnp.where(mask == 1, jnp.nan, v_td)
        vmc = jnp.where(mask == 1, jnp.nan, v_mc)

        # Plot 1: Comparison
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, vt, cmap='viridis', alpha=0.3, antialiased=True)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        if show_td:
            for i in range(ny): ax1.plot(X[i, :], Y[i, :], vtd[i, :], color='blue', alpha=0.8, linewidth=1)
            for j in range(nx): ax1.plot(X[:, j], Y[:, j], vtd[:, j], color='blue', alpha=0.8, linewidth=1)
        if show_mc:
            for i in range(ny): ax1.plot(X[i, :], Y[i, :], vmc[i, :], color='red', alpha=0.8, linewidth=1)
            for j in range(nx): ax1.plot(X[:, j], Y[:, j], vmc[:, j], color='red', alpha=0.8, linewidth=1)

        ax1.set_title("Value Comparison (Surface=True)")
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Value')

        # Plot 2: Error
        ax2 = fig.add_subplot(122, projection='3d')
        if show_td: ax2.plot_surface(X, Y, vt - vtd, color='blue', alpha=0.5)
        if show_mc: ax2.plot_surface(X, Y, vt - vmc, color='red', alpha=0.5)
        ax2.set_title("Underestimation Error ($V_{true} - V_{learned}$)")
        ax2.view_init(elev=elev, azim=azim)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Error')

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='green', lw=4, alpha=0.3, label='True (Surface)')]
        if show_td: legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='TD'))
        if show_mc: legend_elements.append(Line2D([0], [0], color='red', lw=2, label='MC'))
        ax1.legend(handles=legend_elements)
        plt.tight_layout()
        return fig


    controls = mo.hstack([elev_slider, azim_slider, show_td, show_mc], justify="start")
    mo.output.append(controls)
    mo.output.append(render_3d(v_g, td_v_last, mc_v_last, evaluator, elev_slider.value, azim_slider.value, show_td.value, show_mc.value))
    return


@app.cell
def _(evaluator, jnp, mc_v_last, td_v_last, v_g):
    def compute_jaggedness(grid, mask):
        # mask is 1 for walls
        grid_masked = jnp.where(mask == 1, jnp.nan, grid)
        # Difference between adjacent cells (horizontal and vertical)
        diff_h = jnp.abs(grid_masked[:, 1:] - grid_masked[:, :-1])
        diff_v = jnp.abs(grid_masked[1:, :] - grid_masked[:-1, :])
        return jnp.nanmean(diff_h) + jnp.nanmean(diff_v)

    mask = evaluator.occupied_map
    jag_true = compute_jaggedness(v_g, mask)
    jag_td = compute_jaggedness(td_v_last, mask)
    jag_mc = compute_jaggedness(mc_v_last, mask)
    return


@app.cell
def _(evaluator, jnp, plt, td_metrics):
    def plot_features(metrics, title_prefix):
        # Extract the final epoch's top 5 singular vectors
        # Shape: (5, H, W)
        top_vectors = metrics["feature_top_singular_vectors"][-1]

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        mask = evaluator.occupied_map

        for i in range(5):
            ax = axes[i]
            grid = top_vectors[i]

            # Mask walls
            ax.imshow(mask, cmap="Greys", alpha=0.3)
            masked_grid = jnp.where(mask == 1, jnp.nan, grid)

            # Plot feature activation
            im = ax.imshow(masked_grid, cmap="coolwarm")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(f"Component {i+1}")
            ax.axis("off")

        fig.suptitle(f"{title_prefix} Top Feature Spatial Patterns", fontsize=16)
        plt.tight_layout()
        return fig

    # Plot both
    fig_td_feats = plot_features(td_metrics, "Temporal Difference (TD)")
    # fig_mc_feats = plot_features(mc_metrics, "Monte Carlo (MC)")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
