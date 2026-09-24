import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _(alpha, theta):
    import numpy as np
    import scipy.linalg
    import matplotlib.pyplot as plt

    P = [ 
        [0.5, 0, 0.5 ],
        [0.5, 0.5, 0],
        [0, 0.5, 0.5 ],
        ]
    R = [0,0,0]
    P = np.array(P)
    R = np.array(R)
    gamma = 0.99
    def T(v):
        R + gamma * P @ v

    Q = [ 
        [1, 0.5, 1.5 ],
        [1.5, 1.0, 0.5],
        [0.5, 1.5, 1],
        ]
    Q = np.array(Q)

    Q.T @ P

    I = np.eye(3)
    D =  I/3
    A = D @(I - gamma *  P)

    def td_update(v, dv):
        # e = -v
        # theta = theta + alpha * dv.T @ A @ e
        # theta = theta + alpha * dv.T @ A @ (-v)
        theta = theta - alpha * dv.T @ A @ v
        return theta

    def td_lambda_update(theta, v, dv, alpha, λ=0):
        L = np.linalg.inv(I - gamma * λ * P)
        theta = theta - alpha * dv.T @ A @ L @ v
        return theta

    def get_v(theta, eps, v0):
        """
        v(theta) using the matrix exponential.
        v(theta) = e^((Q + eps*I)r) J(0)
        """
        # Uses your global Q and I variables
        matrix_exponent = (Q + eps * I) * theta
        return scipy.linalg.expm(matrix_exponent) @ v0

    # Initial condition v0 = <(1,1,1), J0 > =0, so v0 must sum to 0
    return A, D, I, P, Q, gamma, get_v, np, plt


@app.cell
def _(get_v, np, plt):
    def plot_representable_space(eps=0.01, theta_max=6.0, steps=300):
        # 1. Define v0 orthogonal to [1,1,1] 
        v0 = np.array([1.0, -1.0, 0.0])

        # 2. Generate the trajectory of v over a range of theta values
        thetas = np.linspace(-20, theta_max, steps)
        v_traj = np.array([get_v(theta, eps, v0) for theta in thetas])

        # 3. Create an orthonormal basis for the plane {v in R^3 | e'v = 0}
        u1 = np.array([1, -1, 0]) / np.sqrt(2)
        u2 = np.array([1, 1, -2]) / np.sqrt(6)

        # 4. Project the 3D trajectory onto our 2D basis
        x_proj = v_traj @ u1
        y_proj = v_traj @ u2

        # Find the index closest to theta = 0 to accurately place the start marker
        idx_zero = np.argmin(np.abs(thetas))

        # 5. Create a 1x2 grid for the subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # --- Plot 1: 2D Spiral Projection ---
        ax1.plot(x_proj, y_proj, color='blue', label=r'Representable functions $v(\theta)$')
        ax1.scatter(x_proj[idx_zero], y_proj[idx_zero], color='red', zorder=5, label=r'$v_0$ (Start, $\theta=0$)')

        ax1.set_title('Divergent Spiral of the Nonlinear Function Approximator')
        ax1.set_xlabel('Orthogonal Basis Direction 1')
        ax1.set_ylabel('Orthogonal Basis Direction 2')
        ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax1.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal') 

        # --- Plot 2: Value Components vs Theta ---
        ax2.plot(thetas, v_traj[:, 0], label=r'$v_1(\theta)$', linewidth=2)
        ax2.plot(thetas, v_traj[:, 1], label=r'$v_2(\theta)$', linewidth=2)
        ax2.plot(thetas, v_traj[:, 2], label=r'$v_3(\theta)$', linewidth=2)

        # Mark the starting theta=0 location for all three state values
        ax2.scatter([0, 0, 0], [v_traj[idx_zero, 0], v_traj[idx_zero, 1], v_traj[idx_zero, 2]], color='red', zorder=5)

        ax2.set_title(r'Individual State Value Estimates vs $\theta$')
        ax2.set_xlabel(r'Parameter $\theta$')
        ax2.set_ylabel('Value Estimate')
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax2.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    # Run the visualization
    plot_representable_space(0.2, 10)
    return


@app.cell
def _(A, D, I, P, Q, gamma, get_v, np):
    def trace_E_grad_descent(theta_init, alpha, lam, eps, v0, steps=1000):
        theta = theta_init
        trajectory_theta = []
        trajectory_v = []
        trajectory_loss = []

        L = np.linalg.inv(I - gamma * lam * P)
        LA = L @ A
        A_sym = LA + LA.T

        for _ in range(steps):
            trajectory_theta.append(theta)

            v = get_v(theta, eps, v0)
            trajectory_v.append(v)

            # Record the objective E
            trajectory_loss.append(v.T @ LA @ v)

            dv = (Q + eps * I) @ v
            grad_E = dv.T @ A_sym @ v
            theta = theta - alpha * grad_E

        def loss_fn(t):
            vt = get_v(t, eps, v0)
            return vt.T @ LA @ vt

        return trajectory_theta, np.array(trajectory_v), trajectory_loss, loss_fn, "Objective E"


    def trace_expected_td_lambda(theta_init, alpha, lam, eps, v0, steps=1000):
        theta = theta_init
        trajectory_theta = []
        trajectory_v = []
        trajectory_loss = []

        L = np.linalg.inv(I - gamma * lam * P)
        LA = L @ A

        for _ in range(steps):
            trajectory_theta.append(theta)

            v = get_v(theta, eps, v0)
            trajectory_v.append(v)

            dv = (Q + eps * I) @ v

            # Expected TD update
            expected_update = -dv.T @ LA @ v

            # Record the MSPBE
            mspbe = (expected_update ** 2) / (dv.T @ dv)
            trajectory_loss.append(mspbe)

            theta = theta + alpha * expected_update

        def loss_fn(t):
            vt = get_v(t, eps, v0)
            dvt = (Q + eps * I) @ vt
            update = -dvt.T @ LA @ vt
            return (update ** 2) / (dvt.T @ dvt)

        return trajectory_theta, np.array(trajectory_v), trajectory_loss, loss_fn, "MSPBE"


    def trace_partially_fitted_vi(theta_init, alpha, lam, eps, v0, steps=1000, inner_steps=20):
        theta = theta_init
        trajectory_theta = []
        trajectory_v = []
        trajectory_loss = []

        L = np.linalg.inv(I - gamma * lam * P)
        y = None 

        # Pre-calculate the initial target to render the static background plot
        v_start = get_v(theta_init, eps, v0)
        y0 = v_start - L @ (I - gamma * P) @ v_start

        for step in range(steps):
            if step % inner_steps == 0:
                v_target = get_v(theta, eps, v0)
                y = v_target - L @ (I - gamma * P) @ v_target
    
            trajectory_theta.append(theta)
            v = get_v(theta, eps, v0)
            trajectory_v.append(v)

            # Record the current FVI Euclidean distance
            trajectory_loss.append(0.5 * np.sum((v - y)**2))

            dv = (Q + eps * I) @ v
            grad = dv.T @ (v - y)
            theta = theta - alpha * grad

        def loss_fn(t):
            vt = get_v(t, eps, v0)
            return 0.5 * np.sum((vt - y0)**2)

        return trajectory_theta, np.array(trajectory_v), trajectory_loss, loss_fn, "FVI Euclidean Loss (Initial Target)"


    def trace_MSPBE_grad_descent(theta_init, alpha, lam, eps, v0, steps=1000):
        theta = theta_init
        trajectory_theta = []
        trajectory_v = []
        trajectory_loss = []

        # Resolvent matrix for TD(lambda)
        L = np.linalg.inv(I - gamma * lam * P)
        AL = A @ L

        for _ in range(steps):
            trajectory_theta.append(theta)

            # 1. Forward pass
            v = get_v(theta, eps, v0)
            trajectory_v.append(v)

            # 2. First and second derivatives of v w.r.t theta
            dv = (Q + eps * I) @ v
            ddv = (Q + eps * I) @ dv

            # 3. Compute the scalar components of the MSPBE objective
            N = -dv.T @ AL @ v      
            M = dv.T @ D @ dv       

            # Record the true MSPBE
            mspbe = (N**2) / M
            trajectory_loss.append(mspbe)

            # 4. Compute gradients of the scalar components
            dN = -(ddv.T @ AL @ v + dv.T @ AL @ dv)
            dM = 2.0 * (dv.T @ D @ ddv)

            # 5. Exact true gradient via the quotient rule
            grad_MSPBE = (2.0 * N * dN * M - (N**2) * dM) / (M**2)

            # 6. Apply gradient descent step
            theta = theta - alpha * grad_MSPBE

        def loss_fn(t):
            vt = get_v(t, eps, v0)
            dvt = (Q + eps * I) @ vt
            Nt = -dvt.T @ AL @ vt
            Mt = dvt.T @ D @ dvt
            return (Nt**2) / Mt

        return trajectory_theta, np.array(trajectory_v), trajectory_loss, loss_fn, "True MSPBE Landscape"

    return (
        trace_E_grad_descent,
        trace_MSPBE_grad_descent,
        trace_expected_td_lambda,
        trace_partially_fitted_vi,
    )


@app.cell
def _(I, P, Q, gamma, get_v, np, plt):

    def visualize_dynamics(update_fn, theta_init=0, alpha=0.1, lam=0, eps=0.1, v0=np.array([1.0, -0.5, -0.5]), steps=10000, show_bellman_error=False, show_projection=False, **kwargs):

        # Run the numerical trace (unpacking the 5 returned values)
        thetas, vs, losses, loss_fn, loss_name = update_fn(theta_init, alpha, lam, eps, v0, steps, **kwargs)

        # Projection basis for the 2D spiral plane
        u1 = np.array([1, -1, 0]) / np.sqrt(2)
        u2 = np.array([1, 1, -2]) / np.sqrt(6)

        x_proj = vs @ u1
        y_proj = vs @ u2

        # Plotting
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # --- Plot 1: Theta vs Update Steps ---
        ax1.plot(range(len(thetas)), thetas, color='blue', linewidth=2)
        ax1.set_title(r'Parameter $\theta$ over Time')
        ax1.set_xlabel('Expected Update Steps')
        ax1.set_ylabel(r'$\theta$')
        ax1.grid(True, alpha=0.3)
        min_theta = min(-20, min(thetas) - 2)
        # --- Plot 2: Value Function Trajectory on the Spiral ---
        bg_thetas = np.linspace(min_theta, max(thetas) + 2, 10000)
        bg_vs = np.array([get_v(t, eps, v0) for t in bg_thetas])

        ax2.plot(bg_vs @ u1, bg_vs @ u2, color='gray', linestyle='--', alpha=0.5, label='Representable Space')
        ax2.plot(x_proj, y_proj, color='red', linewidth=2, label='Algorithm Path')
        ax2.scatter(x_proj[0], y_proj[0], color='black', zorder=5, label='Start')

        if show_bellman_error or show_projection:
            v_start = vs[0]
            # P, Q, I, gamma should be accessible in your broader scope
            L = np.linalg.inv(I - gamma * lam * P)
            delta = -L @ (I - gamma * P) @ v_start
            dv = (Q + eps * I) @ v_start
            scalar_proj = (dv.T @ delta) / (dv.T @ dv)
            pi_delta = scalar_proj * dv
    
            start_2d = np.array([x_proj[0], y_proj[0]])
            delta_2d = np.array([delta @ u1, delta @ u2])
            pi_delta_2d = np.array([pi_delta @ u1, pi_delta @ u2])
            dv_2d = np.array([dv @ u1, dv @ u2])
    
            if show_bellman_error:
                ax2.quiver(*start_2d, *delta_2d, angles='xy', scale_units='xy', scale=1, 
                           color='purple', label=r'Bellman Error ($\delta$)', zorder=6, width=0.005)
            if show_projection:
                ax2.quiver(*start_2d, *dv_2d, angles='xy', scale_units='xy', scale=1, 
                           color='orange', alpha=0.3, label=r'Tangent ($dv$)', zorder=4, width=0.003)
                ax2.quiver(*start_2d, *pi_delta_2d, angles='xy', scale_units='xy', scale=1, 
                           color='green', label=r'Projected Error ($\Pi \delta$)', zorder=7, width=0.006)

        ax2.set_title('Value Function on the Plane')
        ax2.set_xlabel('Basis Direction 1')
        ax2.set_ylabel('Basis Direction 2')
        ax2.axis('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Plot 3: Objective Loss Landscape ---
        # Define bounds to render a clean background landscape curve
        min_t_plot = min(np.min(thetas) - 2, -15)
        max_t_plot = max(np.max(thetas) + 2, 10)

        # Cap upper bound for visualization if standard TD/FVI shoots to infinity
        if max_t_plot > 20: 
            max_t_plot = 20 
    
        grid_thetas = np.linspace(min_t_plot, max_t_plot, 1000)
        grid_losses = [loss_fn(t) for t in grid_thetas]

        # Plot the smooth theoretical loss surface
        ax3.plot(grid_thetas, grid_losses, color='blue', alpha=0.4, linewidth=2, label=f'{loss_name} Surface')

        # Overlay the actual step-by-step path the algorithm took
        ax3.plot(thetas, losses, color='red', linewidth=2, marker='.', markersize=4, label='Algorithm Path')
        ax3.scatter(thetas[0], losses[0], color='black', zorder=5, s=50, label='Start')

        ax3.set_title(f'Loss Landscape: {loss_name}')
        ax3.set_xlabel(r'$\theta$')
        ax3.set_ylabel('Objective Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return (visualize_dynamics,)


@app.cell
def _(trace_expected_td_lambda, visualize_dynamics):
    #TD 0:
    # Run with updated defaults
    visualize_dynamics(trace_expected_td_lambda, lam=0, alpha=0.01, steps = 1000, show_bellman_error=True, show_projection=True)
    return


@app.cell
def _(trace_expected_td_lambda, visualize_dynamics):
    # TD(lambda)
    # Run with updated defaults
    visualize_dynamics(trace_expected_td_lambda, lam=0.9)
    return


@app.cell
def _(trace_expected_td_lambda, visualize_dynamics):
    #MC:
    # Run with updated defaults
    visualize_dynamics(trace_expected_td_lambda, lam=1)
    return


@app.cell
def _(trace_E_grad_descent, visualize_dynamics):
    visualize_dynamics(trace_E_grad_descent, lam=0.0)
    return


@app.cell
def _(trace_E_grad_descent, visualize_dynamics):
    visualize_dynamics(trace_E_grad_descent, lam=0.9)
    return


@app.cell
def _():
    # E = e^T δ (since D =I)
    # E = e^T (gamma P v - v) 
    # E = -v^T (gamma P v - v) (since V=0)
    # E = v^Tv - gamma v^T P v (since V=0)

    # MC = e^T D e = e^T e = v^Tv

    # so MC minimizes v^Tv while E minimizes v^Tv - gamma v^T P v

    # But what about TD? can we see it as minimizing an inner product? MSPBE? no...
    return


@app.cell
def _(trace_MSPBE_grad_descent, visualize_dynamics):
    visualize_dynamics(trace_MSPBE_grad_descent, lam=0.0, show_bellman_error = True, show_projection=True)
    return


@app.cell
def _(trace_MSPBE_grad_descent, visualize_dynamics):
    visualize_dynamics(trace_MSPBE_grad_descent, lam=1.0)
    return


@app.cell
def _(trace_partially_fitted_vi, visualize_dynamics):
    # FVI: One must get further from Tv before nearing it again. 
    # i.e. ||Tv-v|| non-monotonic in theta
    visualize_dynamics(trace_partially_fitted_vi, lam=0.0, alpha=0.0001, show_bellman_error=True)
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
