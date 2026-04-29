"""Part 4: deterministic parameter sweeps for instantaneous rescue."""

from __future__ import annotations

import itertools
import numpy as np

from .part2_laplacian import simulate_laplacian_diffusion
from .part3_geometry import build_radius_adjacency, corrected_pattern


def instantaneous_rescue(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Classify nuclei as rescued when final signal exceeds an instantaneous threshold."""
    return np.asarray(signal, dtype=float) >= threshold


def simulate_instantaneous_rescue(
    positions: np.ndarray,
    corrected_fraction: float = 0.1,
    correction_mode: str = "left_cluster",
    radius: float = 0.09,
    alpha: float = 0.12,
    beta: float = 0.03,
    production_rate: float = 0.05,
    dt: float = 0.1,
    steps: int = 300,
    rescue_threshold: float = 0.5,
    seed: int = 7,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    """Run geometry-based diffusion and apply an instantaneous rescue threshold."""
    A = build_radius_adjacency(positions, radius=radius)
    corrected = corrected_pattern(len(positions), corrected_fraction, mode=correction_mode, seed=seed)
    diffusion = simulate_laplacian_diffusion(A, corrected, alpha, beta, production_rate, dt, steps)
    rescued = instantaneous_rescue(diffusion["signal"], rescue_threshold)

    return {
        "positions": np.asarray(positions, dtype=float),
        "adjacency": A,
        "corrected": corrected,
        "signal": diffusion["signal"],
        "signal_history": diffusion["history"],
        "rescued": rescued,
        "rescued_fraction": float(rescued.mean()),
        "parameters": {
            "corrected_fraction": corrected_fraction,
            "radius": radius,
            "alpha": alpha,
            "beta": beta,
            "production_rate": production_rate,
            "dt": dt,
            "steps": steps,
            "rescue_threshold": rescue_threshold,
        },
    }


def run_parameter_sweep(positions: np.ndarray, parameter_grid: dict[str, list[float]], base_parameters: dict | None = None) -> list[dict]:
    """Run a small deterministic grid search over instantaneous-rescue parameters."""
    base_parameters = dict(base_parameters or {})
    keys = list(parameter_grid.keys())
    results = []

    for values in itertools.product(*(parameter_grid[key] for key in keys)):
        params = base_parameters | dict(zip(keys, values))
        result = simulate_instantaneous_rescue(positions, **params)
        results.append({**params, "rescued_fraction": result["rescued_fraction"]})

    return results
