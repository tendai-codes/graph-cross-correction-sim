import numpy as np

from src.spectral import (
    build_geometry_graph,
    compute_laplacian_eigendecomposition,
    run_diffusion_simulation,
    stability_bound,
    steady_state_solution,
)


def test_geometry_laplacian_is_symmetric():
    """Geometry graph should produce a symmetric Laplacian for undirected diffusion."""
    positions = np.array([0.0, 0.5, 1.1, 3.0])  # shape: (4,)
    _, _, laplacian = build_geometry_graph(positions, threshold=1.0)

    assert np.allclose(laplacian, laplacian.T)


def test_laplacian_has_zero_row_sums():
    """Each Laplacian row should sum to zero."""
    positions = np.array([0.0, 0.5, 1.1, 3.0])  # shape: (4,)
    _, _, laplacian = build_geometry_graph(positions, threshold=1.0)

    assert np.allclose(laplacian.sum(axis=1), 0.0)


def test_laplacian_zero_eigenvalue():
    """Smallest Laplacian eigenvalue should be approximately zero."""
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])  # shape: (2, 2)
    degree = np.diag(adjacency.sum(axis=1))  # shape: (2, 2)
    laplacian = degree - adjacency  # shape: (2, 2)

    eigenvalues, _ = compute_laplacian_eigendecomposition(laplacian)

    assert np.isclose(eigenvalues[0], 0.0)


def test_stability_bound_positive():
    """Stability bound must be positive for positive alpha and lambda_max."""
    bound = stability_bound(alpha=0.3, lambda_max=2.0)

    assert bound > 0.0


def test_steady_state_matches_long_euler_simulation():
    """Analytical steady state should match a long stable Euler simulation."""
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])  # shape: (2, 2)
    degree = np.diag(adjacency.sum(axis=1))  # shape: (2, 2)
    laplacian = degree - adjacency  # shape: (2, 2)
    q = np.array([1.0, 0.0])  # shape: (2,)

    alpha = 0.3
    beta = 0.05
    dt = 0.05

    analytical = steady_state_solution(laplacian, q, alpha, beta)
    numerical = run_diffusion_simulation(
        laplacian=laplacian,
        q=q,
        alpha=alpha,
        beta=beta,
        dt=dt,
        n_steps=3000,
    )

    assert np.allclose(analytical, numerical, atol=1e-2)


def test_clustered_placement_rescues_at_least_as_many_as_split():
    """Clustered Fiedler placement should rescue at least as many nuclei as split placement."""
    np.random.seed(42)
    n = 20
    positions = np.sort(np.random.uniform(0, 10, n))  # shape: (n,)
    _, _, laplacian = build_geometry_graph(positions, threshold=1.2)
    eigenvalues, eigenvectors = compute_laplacian_eigendecomposition(laplacian)

    alpha = 0.3
    beta = 0.05
    rescue_threshold = 0.3
    n_corrected = 4

    fiedler = eigenvectors[:, 1]                        # shape: (n,)
    pos_idx = np.where(fiedler >= 0)[0]                 # shape: (n_positive,)
    neg_idx = np.where(fiedler <  0)[0]                 # shape: (n_negative,)

    q_clustered = np.zeros(n)                           # shape: (n,)
    q_clustered[pos_idx[:n_corrected]] = 1.0

    q_split = np.zeros(n)                               # shape: (n,)
    half = n_corrected // 2
    q_split[pos_idx[:half]]  = 1.0
    q_split[neg_idx[:half]]  = 1.0

    u_clustered = steady_state_solution(laplacian, q_clustered, alpha, beta)
    u_split     = steady_state_solution(laplacian, q_split,     alpha, beta)

    rescued_clustered = np.sum(u_clustered > rescue_threshold)
    rescued_split     = np.sum(u_split     > rescue_threshold)

    assert rescued_clustered >= rescued_split
