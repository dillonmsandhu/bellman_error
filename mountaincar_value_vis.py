# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    print("Python Executable:", sys.executable)
    print("Site-Packages:", [p for p in sys.path if "site-packages" in p])
    return


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from envs.mountaincar_exact import MountainCarExactValue

    return MountainCarExactValue, jnp, np, plt


@app.cell
def _(MountainCarExactValue, jnp, np, plt):
    # Initialize the discretized Mountain Car environment
    # Using a 100x100 grid for a smooth, high-resolution swirl
    env = MountainCarExactValue(n_pos=200, n_vel=120, gamma=0.99)
    # Create an energy-pumping policy
    pi_heuristic = np.zeros((env.num_total_states, env.num_actions))

    for s_idx in range(env.num_states):
        pos, vel = env.coords[s_idx]
        if vel > 0:
            pi_heuristic[s_idx, 2] = 1.0  # Accelerate Right
        elif vel < 0:
            pi_heuristic[s_idx, 0] = 1.0  # Accelerate Left
        else:
            # Break ties at zero velocity based on position
            if pos < -0.5:
                pi_heuristic[s_idx, 0] = 1.0 
            else:
                pi_heuristic[s_idx, 2] = 1.0

    # Ensure terminal/goal states have a valid distribution (though it doesn't affect V)
    pi_heuristic[env.goal_idx, 1] = 1.0
    pi_heuristic[env.terminal_idx, 1] = 1.0

    pi_heuristic = jnp.asarray(pi_heuristic)

    # Compute true values on the grid
    V_pi_grid = env.compute_true_values(pi_heuristic)

    # Plot the value function to reveal the archetypal swirl
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use pcolormesh to map exact continuous bins to the plot axes
    X, Y = jnp.meshgrid(env.pos_bins, env.vel_bins, indexing='ij')
    c = ax.pcolormesh(X, Y, V_pi_grid, cmap='viridis', shading='auto')

    fig.colorbar(c, ax=ax, label='Value $V^\pi(s)$')
    ax.set_title("Mountain Car Exact Value Function\n(Random Policy, 100x100 Discretization)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")

    plt.tight_layout()
    fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
