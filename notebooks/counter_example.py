import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    # Non-uniform, highly asymmetric chain
    P = np.array([
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.8, 0.2, 0.0]
    ])

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(P.T)
    pi = np.real(eigvecs[:, np.isclose(eigvals, 1.0)])
    pi = (pi / pi.sum()).flatten()
    D = np.diag(pi)

    gamma = 0.9
    A = D @ (np.eye(3) - gamma * P)
    S = 0.5 * (A + A.T)
    K = 0.5 * (A - A.T)

    # An ill-conditioned feature kernel Theta = J J^T
    # where features couple states with opposing signs
    J = np.array([
        [ 1.0,  0.5],
        [-0.8,  1.2],
        [ 0.2, -0.7]
    ])
    Theta = J @ J.T

    M = S @ Theta @ A
    M_sym = 0.5 * (M + M.T)

    w_eig, v_eig = np.linalg.eigh(M_sym)
    print("Eigenvalues of M_sym:", np.round(w_eig, 6))

    # Pick the eigenvector corresponding to the most negative eigenvalue
    idx_min = np.argmin(w_eig)
    if w_eig[idx_min] < 0:
        e_bad = v_eig[:, idx_min]
        drift = e_bad.T @ S @ Theta @ A @ e_bad
        print(f"Found negative drift! Drift = {drift:.6f}")
        print(f"dE/dt = {-0.1 * drift:.6f} > 0")
    else:
        print("No negative eigenvalue found.")
    return A, D, S, np


@app.cell
def _(A, D, S, np):
    SA = S @ A

    # 4. Extract the Symmetric Part of SA
    SA_sym = 0.5 * (SA + SA.T)

    # 5. Find the Eigenvalues and Eigenvectors of SA_sym
    # eigh returns eigenvalues in ascending order, so index 0 is the minimum
    eigenvalues, eigenvectors = np.linalg.eigh(SA_sym)
    min_eig = eigenvalues[0]

    print(f"Stationary Distribution (mu): {D.diagonal()}")
    print(f"Minimum Eigenvalue of SA_sym: {min_eig:.6f}\n")

    if min_eig < 0:
        print("FATAL FLAW DETECTED: SA has a negative eigenvalue.")

        # 6. Extract the fatal error vector
        fatal_e = eigenvectors[:, 0]
        print(f"The Fatal Error Vector (e): {fatal_e}")

        # 7. Prove the Alignment Condition is Negative
        # Trace = e^T * SA * e
        alignment_trace = fatal_e.T @ SA @ fatal_e

        print(f"Alignment Condition (e^T SA e): {alignment_trace:.6f}")
    else:
        print("Matrix is safe. SA is Positive Definite.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
