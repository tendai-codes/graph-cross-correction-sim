"""Biologically framed hypothesis experiments for graph-cross-correction-sim.

The functions in this module deliberately report model outputs as produced by the
simulation. They do not encode assumptions such as "clustered placement must be
better". Biological claims should be made only after inspecting the generated
results.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .part2_laplacian import graph_laplacian
from .part3_geometry import build_radius_adjacency, corrected_pattern, generate_positions


def source_from_corrected(corrected: np.ndarray, production_rate: float = 0.05) -> np.ndarray:
    """Create a source vector from a boolean corrected-nucleus mask."""
    source = np.zeros(len(corrected), dtype=float)  # shape: (n_nuclei,)
    source[np.asarray(corrected, dtype=bool)] = production_rate
    return source


def steady_state_signal(laplacian: np.ndarray, source: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Solve the steady-state graph diffusion equation."""
    if beta <= 0:
        raise ValueError("beta must be positive so the steady-state system is invertible")
    identity = np.eye(laplacian.shape[0])  # shape: (n_nuclei, n_nuclei)
    system_matrix = alpha * laplacian + beta * identity  # shape: (n_nuclei, n_nuclei)
    return np.linalg.solve(system_matrix, source)


def simulate_biological_rescue(
    positions: np.ndarray,
    corrected_fraction: float,
    correction_mode: str,
    radius: float = 0.09,
    alpha: float = 0.12,
    beta: float = 0.03,
    production_rate: float = 0.05,
    rescue_threshold: float = 0.5,
    seed: int = 7,
) -> dict[str, object]:
    """Run a steady-state rescue simulation with biological labels."""
    positions = np.asarray(positions, dtype=float)  # shape: (n_nuclei,)
    adjacency = build_radius_adjacency(positions, radius=radius)  # shape: (n_nuclei, n_nuclei)
    laplacian = graph_laplacian(adjacency)  # shape: (n_nuclei, n_nuclei)
    corrected = corrected_pattern(len(positions), corrected_fraction, correction_mode, seed=seed)  # shape: (n_nuclei,)
    source = source_from_corrected(corrected, production_rate=production_rate)  # shape: (n_nuclei,)
    signal = steady_state_signal(laplacian, source, alpha=alpha, beta=beta)  # shape: (n_nuclei,)
    rescued = signal >= rescue_threshold  # shape: (n_nuclei,)

    return {
        "positions": positions,
        "adjacency": adjacency,
        "laplacian": laplacian,
        "corrected": corrected,
        "source": source,
        "signal": signal,
        "rescued": rescued,
        "corrected_fraction": float(corrected.mean()),
        "rescued_fraction": float(rescued.mean()),
        "mean_signal": float(np.mean(signal)),
        "max_signal": float(np.max(signal)),
        "min_signal": float(np.min(signal)),
        "alpha": float(alpha),
        "beta": float(beta),
        "production_rate": float(production_rate),
        "rescue_threshold": float(rescue_threshold),
        "correction_mode": correction_mode,
        "n_nuclei": int(len(positions)),
    }


def graph_metrics(laplacian: np.ndarray) -> dict[str, float]:
    """Return simple spectral graph metrics used to interpret rescue behaviour."""
    eigenvalues = np.linalg.eigvalsh(laplacian)  # shape: (n_nuclei,)
    zero_count = int(np.sum(np.isclose(eigenvalues, 0.0, atol=1e-8)))
    fiedler_value = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    return {
        "components_estimate": float(zero_count),
        "algebraic_connectivity": fiedler_value,
        "lambda_max": float(eigenvalues[-1]),
    }


def run_clustered_vs_distributed_experiment(
    n_nuclei: int = 60,
    corrected_fraction: float = 0.15,
    seed: int = 7,
    **kwargs,
) -> list[dict[str, float | str]]:
    """Compare clustered and distributed correction patterns on the same fibre."""
    positions = generate_positions(n_nodes=n_nuclei, length=1.0, jitter=0.005, seed=seed)  # shape: (n_nuclei,)
    rows = []
    for mode in ["left_cluster", "central_cluster", "evenly_spaced", "random"]:
        result = simulate_biological_rescue(
            positions,
            corrected_fraction=corrected_fraction,
            correction_mode=mode,
            seed=seed,
            **kwargs,
        )
        metrics = graph_metrics(result["laplacian"])
        rows.append({
            "hypothesis": "clustered_vs_distributed_correction",
            "strategy": mode,
            "n_nuclei": result["n_nuclei"],
            "corrected_fraction_requested": corrected_fraction,
            "corrected_fraction_actual": result["corrected_fraction"],
            "rescued_fraction": result["rescued_fraction"],
            "mean_signal": result["mean_signal"],
            "max_signal": result["max_signal"],
            **metrics,
        })
    return rows


def run_transduction_efficiency_sweep(
    n_nuclei: int = 60,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60),
    correction_mode: str = "evenly_spaced",
    seed: int = 7,
    **kwargs,
) -> list[dict[str, float | str]]:
    """Measure rescue over increasing gene-correction efficiency."""
    positions = generate_positions(n_nodes=n_nuclei, length=1.0, jitter=0.005, seed=seed)  # shape: (n_nuclei,)
    rows = []
    for fraction in fractions:
        result = simulate_biological_rescue(
            positions,
            corrected_fraction=fraction,
            correction_mode=correction_mode,
            seed=seed,
            **kwargs,
        )
        rows.append({
            "hypothesis": "transduction_efficiency",
            "strategy": correction_mode,
            "n_nuclei": result["n_nuclei"],
            "corrected_fraction_requested": fraction,
            "corrected_fraction_actual": result["corrected_fraction"],
            "rescued_fraction": result["rescued_fraction"],
            "mean_signal": result["mean_signal"],
            "max_signal": result["max_signal"],
        })
    return rows


def run_protein_stability_sweep(
    n_nuclei: int = 60,
    beta_values: tuple[float, ...] = (0.01, 0.03, 0.07, 0.12),
    corrected_fraction: float = 0.15,
    correction_mode: str = "evenly_spaced",
    seed: int = 7,
    **kwargs,
) -> list[dict[str, float | str]]:
    """Test how degradation rate influences rescue potential."""
    positions = generate_positions(n_nodes=n_nuclei, length=1.0, jitter=0.005, seed=seed)  # shape: (n_nuclei,)
    rows = []
    for beta in beta_values:
        result = simulate_biological_rescue(
            positions,
            corrected_fraction=corrected_fraction,
            correction_mode=correction_mode,
            beta=beta,
            seed=seed,
            **kwargs,
        )
        rows.append({
            "hypothesis": "protein_stability",
            "strategy": correction_mode,
            "n_nuclei": result["n_nuclei"],
            "beta": beta,
            "half_life_proxy": math.log(2) / beta,
            "corrected_fraction_actual": result["corrected_fraction"],
            "rescued_fraction": result["rescued_fraction"],
            "mean_signal": result["mean_signal"],
            "max_signal": result["max_signal"],
        })
    return rows


def run_fibre_length_scaling_experiment(
    n_values: tuple[int, ...] = (20, 50, 100, 250),
    corrected_fraction: float = 0.15,
    correction_mode: str = "evenly_spaced",
    seed: int = 7,
    **kwargs,
) -> list[dict[str, float | str]]:
    """Test whether rescue behaviour changes as the number of nuclei increases."""
    rows = []
    for n_nuclei in n_values:
        positions = generate_positions(n_nodes=n_nuclei, length=1.0, jitter=0.003, seed=seed)  # shape: (n_nuclei,)
        result = simulate_biological_rescue(
            positions,
            corrected_fraction=corrected_fraction,
            correction_mode=correction_mode,
            seed=seed,
            **kwargs,
        )
        rows.append({
            "hypothesis": "fibre_length_scaling",
            "strategy": correction_mode,
            "n_nuclei": result["n_nuclei"],
            "corrected_fraction_actual": result["corrected_fraction"],
            "rescued_fraction": result["rescued_fraction"],
            "mean_signal": result["mean_signal"],
            "max_signal": result["max_signal"],
        })
    return rows


def bootstrap_confidence_interval(values: np.ndarray, n_bootstrap: int = 1000, seed: int = 7) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)  # shape: (n_samples,)
    if len(values) == 0:
        raise ValueError("values must contain at least one sample")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)  # shape: (n_bootstrap,)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = np.mean(sample)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def run_random_placement_statistics(
    n_nuclei: int = 60,
    corrected_fraction: float = 0.15,
    n_trials: int = 250,
    seed: int = 7,
    **kwargs,
) -> dict[str, float | int | str]:
    """Estimate rescue variability from random corrected-nucleus placement."""
    positions = generate_positions(n_nodes=n_nuclei, length=1.0, jitter=0.005, seed=seed)  # shape: (n_nuclei,)
    rescued_fractions = np.empty(n_trials, dtype=float)  # shape: (n_trials,)
    mean_signals = np.empty(n_trials, dtype=float)  # shape: (n_trials,)
    for trial in range(n_trials):
        result = simulate_biological_rescue(
            positions,
            corrected_fraction=corrected_fraction,
            correction_mode="random",
            seed=seed + trial,
            **kwargs,
        )
        rescued_fractions[trial] = result["rescued_fraction"]
        mean_signals[trial] = result["mean_signal"]
    ci_low, ci_high = bootstrap_confidence_interval(rescued_fractions, seed=seed)
    return {
        "hypothesis": "random_placement_statistics",
        "strategy": "random",
        "n_nuclei": n_nuclei,
        "n_trials": n_trials,
        "corrected_fraction_requested": corrected_fraction,
        "mean_rescued_fraction": float(np.mean(rescued_fractions)),
        "std_rescued_fraction": float(np.std(rescued_fractions, ddof=1)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mean_signal_across_trials": float(np.mean(mean_signals)),
    }


def fiedler_masks(laplacian: np.ndarray, n_corrected: int) -> tuple[np.ndarray, np.ndarray]:
    """Return clustered and split source masks using the Fiedler partition."""
    n_nuclei = laplacian.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, 1]  # shape: (n_nuclei,)
    positive = np.where(fiedler >= 0)[0]
    negative = np.where(fiedler < 0)[0]

    if len(positive) < len(negative):
        positive, negative = negative, positive

    clustered = np.zeros(n_nuclei, dtype=bool)  # shape: (n_nuclei,)
    clustered[positive[:n_corrected]] = True

    split = np.zeros(n_nuclei, dtype=bool)  # shape: (n_nuclei,)
    half = n_corrected // 2
    split[positive[:half]] = True
    split[negative[: n_corrected - half]] = True

    return clustered, split


def run_fiedler_validation(
    n_graphs: int = 100,
    n_nuclei: int = 60,
    corrected_fraction: float = 0.15,
    seed: int = 7,
    radius: float = 0.09,
    alpha: float = 0.12,
    beta: float = 0.03,
    production_rate: float = 0.05,
    rescue_threshold: float = 0.5,
) -> dict[str, float | int | str]:
    """Compare Fiedler-clustered and split placements across random graph realisations."""
    clustered_scores = []
    split_scores = []
    n_corrected = max(2, int(round(n_nuclei * corrected_fraction)))
    if n_corrected % 2 == 1:
        n_corrected += 1

    for graph_id in range(n_graphs):
        positions = generate_positions(n_nodes=n_nuclei, length=1.0, jitter=0.006, seed=seed + graph_id)
        adjacency = build_radius_adjacency(positions, radius=radius)
        laplacian = graph_laplacian(adjacency)
        clustered, split = fiedler_masks(laplacian, n_corrected=n_corrected)

        q_clustered = source_from_corrected(clustered, production_rate=production_rate)
        q_split = source_from_corrected(split, production_rate=production_rate)
        u_clustered = steady_state_signal(laplacian, q_clustered, alpha=alpha, beta=beta)
        u_split = steady_state_signal(laplacian, q_split, alpha=alpha, beta=beta)
        clustered_scores.append(float(np.mean(u_clustered >= rescue_threshold)))
        split_scores.append(float(np.mean(u_split >= rescue_threshold)))

    clustered_array = np.asarray(clustered_scores, dtype=float)
    split_array = np.asarray(split_scores, dtype=float)
    difference = clustered_array - split_array
    return {
        "hypothesis": "fiedler_placement_validation",
        "strategy": "fiedler_clustered_vs_split",
        "n_graphs": n_graphs,
        "n_nuclei": n_nuclei,
        "corrected_nuclei_used": n_corrected,
        "mean_clustered_rescue": float(np.mean(clustered_array)),
        "mean_split_rescue": float(np.mean(split_array)),
        "mean_difference_clustered_minus_split": float(np.mean(difference)),
        "clustered_better_fraction": float(np.mean(difference > 0)),
        "equal_fraction": float(np.mean(np.isclose(difference, 0.0))),
        "split_better_fraction": float(np.mean(difference < 0)),
    }


def write_rows_to_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write a list of dictionaries to CSV using the union of available columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
