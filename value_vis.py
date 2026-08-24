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

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    from envs.fourrooms import FourRoomsExactValue
    from core.config import config 
    import gymnax 
    import jax.numpy as jnp
    import jax
    import matplotlib.pyplot as plt

    env, env_params = gymnax.make(config["ENV_NAME"], use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
    env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config['FAIL_PROB']
    )
    evaluator = FourRoomsExactValue(start_pos = env.pos_fixed, goal_pos = env.goal_fixed, fail_prob= env_params.fail_prob) # for computing the true 
    def random_policy(_):
        m = evaluator.num_actions
        return jnp.ones(m) / m

    jax.vmap(random_policy)(evaluator.obs_stack).shape
    return evaluator, jax, jnp, plt, random_policy


@app.cell
def _(jnp, plt):
    def plot_grid(grid, evaluator, title, cmap="viridis"):
        plt.figure(figsize=(6, 6))

        # Create a mask for walls (where occupied_map is 1)
        # evaluator.occupied_map is 1 for walls, 0 for free space
        mask = evaluator.occupied_map

        # Plot background walls in gray
        plt.imshow(mask, cmap="Greys", alpha=0.3)

        # Mask the grid values where there are walls to avoid plotting 0s in wall cells
        masked_grid = jnp.where(mask == 1, jnp.nan, grid)

        # Display the values
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

    return (plot_grid,)


@app.cell
def _(evaluator, jax, jnp, plot_grid, random_policy):
    target_policy_probs = jax.vmap(random_policy)(evaluator.obs_stack)
    pi = jnp.vstack([target_policy_probs, target_policy_probs[0]])
    V_pi = evaluator.compute_true_values_raw(pi)
    mu_vector, _ = evaluator.compute_stationary_distribution_raw(target_policy_probs)
    mu_grid = evaluator.get_value_grid(mu_vector)

    plot_grid(mu_grid, evaluator, "Stationary Distribution ($\mu$)", cmap="magma")
    return V_pi, mu_vector, target_policy_probs


@app.cell
def _(V_pi, evaluator, plot_grid):
    v_grid = evaluator.get_value_grid(V_pi[:-1])
    plot_grid(v_grid, evaluator, "True Value Function ($V^\pi$)", cmap="viridis")
    return


@app.cell
def _(V_pi, evaluator, jnp, plt, target_policy_probs):
    def plot_combined(mu_grid, v_grid, evaluator):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        mask = evaluator.occupied_map
        start_y, start_x = evaluator.start
        goal_y, goal_x = evaluator.goal

        # Plot Stationary Distribution
        axes[0].imshow(mask, cmap="Greys", alpha=0.3)
        mu_masked = jnp.where(mask == 1, jnp.nan, mu_grid)
        im0 = axes[0].imshow(mu_masked, cmap="magma")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].text(start_x, start_y, 'S', ha='center', va='center', color='blue', fontweight='bold')
        axes[0].text(goal_x, goal_y, 'G', ha='center', va='center', color='red', fontweight='bold')
        axes[0].set_title("Stationary Distribution ($\mu$)")
        axes[0].axis("off")

        # Plot Value Function
        axes[1].imshow(mask, cmap="Greys", alpha=0.3)
        v_masked = jnp.where(mask == 1, jnp.nan, v_grid)
        im1 = axes[1].imshow(v_masked, cmap="viridis")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].text(start_x, start_y, 'S', ha='center', va='center', color='blue', fontweight='bold')
        axes[1].text(goal_x, goal_y, 'G', ha='center', va='center', color='red', fontweight='bold')
        axes[1].set_title("True Value Function ($V^\pi$)")
        axes[1].axis("off")

        plt.tight_layout()
        return fig

    mu_vec, _ = evaluator.compute_stationary_distribution_raw(target_policy_probs)
    mu_g = evaluator.get_value_grid(mu_vec)
    v_g = evaluator.get_value_grid(V_pi[:-1])

    plot_combined(mu_g, v_g, evaluator)
    return (v_g,)


@app.cell
def _(mu_vector):
    mu_vector.shape
    return


@app.cell
def _(evaluator, plot_grid, target_policy_probs):
    mu_discounted = evaluator.compute_discounted_visitation_raw(target_policy_probs)

    mu_discounted_grid = evaluator.get_value_grid(mu_discounted)

    plot_grid(mu_discounted_grid, evaluator, "Stationary Distribution ($\mu$)", cmap="magma")
    return


@app.cell
def _():
    from core.utils import load_run_data
    td_config, td_metrics = load_run_data('random/td_exact/20260824_143231/', 'FourRooms-misc', 'results')

    print("Config ID: ", id(td_config))
    print("Metrics ID:", id(td_metrics))
    print("Are they the exact same object in memory?", td_config is td_metrics)
    print("Config keys: ", list(td_config.keys()) if isinstance(td_config, dict) else type(td_config))
    print("Metrics keys:", list(td_metrics.keys()) if isinstance(td_metrics, dict) else type(td_metrics))
    return load_run_data, td_config, td_metrics


@app.cell
def _(evaluator, plot_grid, td_metrics, v_g):
    plot_grid((evaluator.get_value_grid(td_metrics['V_nn'][0,-1]) - v_g)**2, evaluator, "TD Value Estimate")

    return


@app.cell
def _(load_run_data):
    mc_config, mc_metrics = load_run_data('random/mc_exact/20260824_143558/', 'FourRooms-misc', 'results')
    return (mc_metrics,)


@app.cell
def _(evaluator, mc_metrics, plot_grid, v_g):
    plot_grid((evaluator.get_value_grid(mc_metrics['V_nn'][0,-1]) - v_g)**2, evaluator, "MC Value Estimate")

    return


@app.cell
def _(evaluator, mc_metrics, plot_grid, td_metrics):
    plot_grid((evaluator.get_value_grid(mc_metrics['V_nn'][0,-1]) - evaluator.get_value_grid(td_metrics['V_nn'][0,-1]))**2, evaluator, "MC Value Estimate")

    return


@app.cell
def _(td_config, td_metrics):
    import marimo as mo

    mo.vstack([
        mo.md("### Config"),
        td_config,
        mo.md("### Metrics"),
        td_metrics
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
