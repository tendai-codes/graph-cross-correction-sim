"""Part 5: cumulative exposure rescue model."""

from __future__ import annotations

import itertools
import numpy as np

from .part2_laplacian import diffusion_step, graph_laplacian
from .part3_geometry import build_radius_adjacency, corrected_pattern
from .part4_sweeps import simulate_instantaneous_rescue


def cumulative_rescue(exposure: np.ndarray, exposure_threshold: float) -> np.ndarray:
    """Classify nuclei as rescued once accumulated exposure reaches the threshold."""
    return np.asarray(exposure, dtype=float) >= exposure_threshold


def simulate_cumulative_exposure(
    positions: np.ndarray,
    corrected_fraction: float = 0.1,
    correction_mode: str = "left_cluster",
    radius: float = 0.09,
    alpha: float = 0.12,
    beta: float = 0.03,
    production_rate: float = 0.05,
    dt: float = 0.1,
    steps: int = 300,
    exposure_threshold: float = 1.5,
    seed: int = 7,
    exposure_timing: str = "post_update",
) -> dict[str, np.ndarray | float | dict[str, float | str]]:
    """Simulate graph diffusion and rescue using cumulative signal exposure.

    exposure_timing controls whether exposure integrates the signal before or after
    each Euler update. The default, "post_update", matches the original notebook.
    "pre_update" is useful for sensitivity checks of the numerical convention.
    """
    positions = np.asarray(positions, dtype=float)
    A = build_radius_adjacency(positions, radius=radius)
    L = graph_laplacian(A)
    corrected = corrected_pattern(len(positions), corrected_fraction, mode=correction_mode, seed=seed)

    signal = np.zeros(len(positions), dtype=float)
    exposure = np.zeros(len(positions), dtype=float)
    q = np.zeros(len(positions), dtype=float)
    q[corrected] = production_rate

    rescued = np.zeros(len(positions), dtype=bool)
    signal_history = [signal.copy()]
    exposure_history = [exposure.copy()]
    rescued_history = [rescued.copy()]

    if exposure_timing not in {"pre_update", "post_update"}:
        raise ValueError("exposure_timing must be 'pre_update' or 'post_update'")

    for _ in range(steps):
        if exposure_timing == "pre_update":
            exposure = exposure + signal * dt
            rescued = rescued | cumulative_rescue(exposure, exposure_threshold)
            signal = diffusion_step(signal, L, q, alpha=alpha, beta=beta, dt=dt)
        else:
            signal = diffusion_step(signal, L, q, alpha=alpha, beta=beta, dt=dt)
            exposure = exposure + signal * dt
            rescued = rescued | cumulative_rescue(exposure, exposure_threshold)

        signal_history.append(signal.copy())
        exposure_history.append(exposure.copy())
        rescued_history.append(rescued.copy())

    return {
        "positions": positions,
        "adjacency": A,
        "laplacian": L,
        "corrected": corrected,
        "source": q,
        "signal": signal,
        "exposure": exposure,
        "rescued": rescued,
        "signal_history": np.asarray(signal_history),
        "exposure_history": np.asarray(exposure_history),
        "rescued_history": np.asarray(rescued_history),
        "rescued_fraction": float(rescued.mean()),
        "parameters": {
            "corrected_fraction": corrected_fraction,
            "correction_mode": correction_mode,
            "radius": radius,
            "alpha": alpha,
            "beta": beta,
            "production_rate": production_rate,
            "dt": dt,
            "steps": steps,
            "exposure_threshold": exposure_threshold,
            "exposure_timing": exposure_timing,
        },
    }


def compare_instantaneous_and_cumulative(
    positions: np.ndarray,
    rescue_threshold: float = 0.5,
    exposure_threshold: float = 1.5,
    **shared_parameters,
) -> dict[str, dict]:
    """Run Part 4 and Part 5 rescue rules on the same geometry and transport parameters."""
    instantaneous = simulate_instantaneous_rescue(
        positions,
        rescue_threshold=rescue_threshold,
        **shared_parameters,
    )
    cumulative = simulate_cumulative_exposure(
        positions,
        exposure_threshold=exposure_threshold,
        **shared_parameters,
    )
    return {"instantaneous": instantaneous, "cumulative": cumulative}


def run_cumulative_sweep(positions: np.ndarray, parameter_grid: dict[str, list[float]], base_parameters: dict | None = None) -> list[dict]:
    """Run a deterministic grid search over cumulative-exposure parameters."""
    base_parameters = dict(base_parameters or {})
    keys = list(parameter_grid.keys())
    results = []

    for values in itertools.product(*(parameter_grid[key] for key in keys)):
        params = base_parameters | dict(zip(keys, values))
        result = simulate_cumulative_exposure(positions, **params)
        results.append({
            **params,
            "rescued_fraction": result["rescued_fraction"],
            "mean_exposure": float(np.mean(result["exposure"])),
            "max_exposure": float(np.max(result["exposure"])),
        })

    return results
