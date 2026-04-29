"""Part 3: geometry-derived graph construction for nuclear positions."""

from __future__ import annotations

import numpy as np


def generate_positions(
    n_nodes: int = 30,
    length: float = 1.0,
    jitter: float = 0.01,
    seed: int = 7,
) -> np.ndarray:
    """Generate slightly irregular one-dimensional nuclear positions along a fibre."""
    rng = np.random.default_rng(seed)

    base = np.linspace(0.0, length, n_nodes)
    noise = rng.normal(0.0, jitter, size=n_nodes)

    positions = np.clip(base + noise, 0.0, length)
    positions.sort()

    return positions


def build_radius_adjacency(
    positions: np.ndarray,
    radius: float = 0.09,
) -> np.ndarray:
    """Create an undirected adjacency matrix by connecting nuclei within a distance radius.

    Supports both:

    - one-dimensional positions with shape ``(n_nodes,)``
    - two-dimensional positions with shape ``(n_nodes, 2)``
    """
    positions = np.asarray(positions, dtype=float)
    n_nodes = len(positions)

    A = np.zeros((n_nodes, n_nodes), dtype=float)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distance = np.linalg.norm(positions[i] - positions[j])

            if distance <= radius:
                A[i, j] = 1.0
                A[j, i] = 1.0

    return A


def build_geometric_graph(
    positions: np.ndarray,
    radius: float = 0.09,
) -> np.ndarray:
    """Alias for geometry-based adjacency construction.

    This keeps the animation code readable while reusing the same
    radius-based graph construction used in Part 3.
    """
    return build_radius_adjacency(positions=positions, radius=radius)


def corrected_pattern(
    n_nodes: int,
    corrected_fraction: float = 0.1,
    mode: str = "left_cluster",
    seed: int = 7,
) -> np.ndarray:
    """Create deterministic corrected-nucleus patterns for simulation experiments."""
    n_corrected = max(1, int(round(n_nodes * corrected_fraction)))
    corrected = np.zeros(n_nodes, dtype=bool)

    if mode == "left_cluster":
        corrected[:n_corrected] = True

    elif mode == "right_cluster":
        corrected[-n_corrected:] = True

    elif mode == "central_cluster":
        start = max(0, (n_nodes - n_corrected) // 2)
        corrected[start : start + n_corrected] = True

    elif mode == "evenly_spaced":
        indices = np.linspace(0, n_nodes - 1, n_corrected, dtype=int)
        corrected[indices] = True

    elif mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_nodes, size=n_corrected, replace=False)
        corrected[indices] = True

    else:
        raise ValueError(f"Unknown corrected pattern mode: {mode}")

    return corrected