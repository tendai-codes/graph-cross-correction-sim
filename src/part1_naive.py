"""Part 1: naive neighbour-averaging diffusion on a one-dimensional fibre."""

from __future__ import annotations

import numpy as np


def initialise_line_signal(n_nodes: int, corrected_indices: list[int] | np.ndarray, production_rate: float = 1.0) -> np.ndarray:
    """Create a 1D signal vector with production at corrected nuclei."""
    signal = np.zeros(n_nodes, dtype=float)
    signal[np.asarray(corrected_indices, dtype=int)] = production_rate
    return signal


def naive_neighbour_step(signal: np.ndarray, retention: float = 0.6, neighbour_share: float = 0.2) -> np.ndarray:
    """Apply one heuristic neighbour-averaging step along a 1D line."""
    signal = np.asarray(signal, dtype=float)
    next_signal = retention * signal.copy()

    if len(signal) > 1:
        next_signal[:-1] += neighbour_share * signal[1:]
        next_signal[1:] += neighbour_share * signal[:-1]

    return np.maximum(next_signal, 0.0)


def simulate_naive_line(
    n_nodes: int = 30,
    corrected_indices: list[int] | np.ndarray | None = None,
    production_rate: float = 1.0,
    retention: float = 0.6,
    neighbour_share: float = 0.2,
    steps: int = 50,
) -> dict[str, np.ndarray]:
    """Run the Part 1 naive line simulation."""
    if corrected_indices is None:
        corrected_indices = [0]

    signal = initialise_line_signal(n_nodes, corrected_indices, production_rate)
    history = [signal.copy()]

    for _ in range(steps):
        signal = naive_neighbour_step(signal, retention=retention, neighbour_share=neighbour_share)
        signal[np.asarray(corrected_indices, dtype=int)] += production_rate
        history.append(signal.copy())

    return {"signal": signal, "history": np.asarray(history)}
