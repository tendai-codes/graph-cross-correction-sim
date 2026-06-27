import numpy as np


def build_geometry_graph(positions, threshold):
    """Build adjacency, degree, and Laplacian matrices from one-dimensional positions."""
    n = len(positions)
    adjacency = np.zeros((n, n))  # shape: (n, n)
    for i in range(n):
        for j in range(n):
            if i != j and abs(positions[i] - positions[j]) < threshold:
                adjacency[i, j] = 1.0
    degree = np.diag(adjacency.sum(axis=1))  # shape: (n, n)
    laplacian = degree - adjacency  # shape: (n, n)
    return adjacency, degree, laplacian


def compute_laplacian_eigendecomposition(laplacian):
    """Return eigenvalues and eigenvectors of a symmetric graph Laplacian."""
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    return eigenvalues, eigenvectors


def stability_bound(alpha, lambda_max):
    """Return the explicit Euler timestep stability limit for graph diffusion."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if lambda_max <= 0:
        raise ValueError("lambda_max must be positive")
    return 2.0 / (alpha * lambda_max)


def run_diffusion_simulation(laplacian, q, alpha, beta, dt, n_steps):
    """Run explicit Euler graph diffusion with decay and source production."""
    n = laplacian.shape[0]
    u = np.zeros(n)  # shape: (n,)
    for _ in range(n_steps):
        u = u + dt * (-alpha * laplacian @ u - beta * u + q)
    return u


def steady_state_solution(laplacian, q, alpha, beta):
    """Solve the analytical steady-state diffusion system."""
    if beta <= 0:
        raise ValueError("beta must be positive so the steady-state system is invertible")
    n = laplacian.shape[0]
    identity = np.eye(n)  # shape: (n, n)
    system_matrix = alpha * laplacian + beta * identity  # shape: (n, n)
    return np.linalg.solve(system_matrix, q)
