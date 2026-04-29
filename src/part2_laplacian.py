"""Part 2: graph Laplacian diffusion on an abstract fibre graph."""

from __future__ import annotations

import numpy as np


def line_adjacency(n_nodes: int) -> np.ndarray:
    """Create an undirected path-graph adjacency matrix."""
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def graph_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Return the combinatorial graph Laplacian L = D - A."""
    A = np.asarray(adjacency, dtype=float)
    degrees = A.sum(axis=1)
    return np.diag(degrees) - A


def diffusion_step(signal: np.ndarray, laplacian: np.ndarray, source: np.ndarray, alpha: float, beta: float, dt: float) -> np.ndarray:
    """Apply one explicit Euler step for graph diffusion with decay and source production."""
    signal = np.asarray(signal, dtype=float)
    L = np.asarray(laplacian, dtype=float)
    q = np.asarray(source, dtype=float)
    next_signal = signal + dt * (-alpha * (L @ signal) - beta * signal + q)
    return np.maximum(next_signal, 0.0)


def simulate_laplacian_diffusion(
    adjacency: np.ndarray,
    corrected: np.ndarray,
    alpha: float = 0.12,
    beta: float = 0.03,
    production_rate: float = 0.05,
    dt: float = 0.1,
    steps: int = 300,
) -> dict[str, np.ndarray]:
    """Run Laplacian diffusion with a fixed source term at corrected nuclei."""
    A = np.asarray(adjacency, dtype=float)
    corrected = np.asarray(corrected, dtype=bool)
    L = graph_laplacian(A)
    signal = np.zeros(A.shape[0], dtype=float)
    q = np.zeros(A.shape[0], dtype=float)
    q[corrected] = production_rate
    history = [signal.copy()]

    for _ in range(steps):
        signal = diffusion_step(signal, L, q, alpha=alpha, beta=beta, dt=dt)
        history.append(signal.copy())

    return {"signal": signal, "history": np.asarray(history), "laplacian": L, "source": q}
