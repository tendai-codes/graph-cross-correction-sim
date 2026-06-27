import numpy as np

from src.biological_experiments import (
    bootstrap_confidence_interval,
    run_clustered_vs_distributed_experiment,
    run_random_placement_statistics,
    run_transduction_efficiency_sweep,
    simulate_biological_rescue,
)
from src.part3_geometry import generate_positions


def test_biological_rescue_output_shapes():
    positions = generate_positions(n_nodes=20, length=1.0, jitter=0.001, seed=1)
    result = simulate_biological_rescue(
        positions,
        corrected_fraction=0.2,
        correction_mode="evenly_spaced",
        radius=0.12,
    )

    assert result["signal"].shape == (20,)
    assert result["rescued"].shape == (20,)
    assert 0.0 <= result["rescued_fraction"] <= 1.0


def test_transduction_efficiency_sweep_reports_requested_fractions():
    rows = run_transduction_efficiency_sweep(
        n_nuclei=30,
        fractions=(0.1, 0.2),
        radius=0.12,
        seed=3,
    )

    assert len(rows) == 2
    assert rows[0]["corrected_fraction_requested"] == 0.1
    assert rows[1]["corrected_fraction_requested"] == 0.2


def test_random_placement_statistics_are_reproducible_with_seed():
    first = run_random_placement_statistics(n_nuclei=25, n_trials=20, seed=11, radius=0.14)
    second = run_random_placement_statistics(n_nuclei=25, n_trials=20, seed=11, radius=0.14)

    assert first["mean_rescued_fraction"] == second["mean_rescued_fraction"]
    assert first["ci95_low"] == second["ci95_low"]
    assert first["ci95_high"] == second["ci95_high"]


def test_bootstrap_confidence_interval_is_ordered():
    low, high = bootstrap_confidence_interval(np.array([0.1, 0.2, 0.3, 0.4]), n_bootstrap=100, seed=5)

    assert low <= high


def test_clustered_vs_distributed_does_not_assume_winner():
    rows = run_clustered_vs_distributed_experiment(n_nuclei=30, corrected_fraction=0.2, radius=0.12)

    strategies = {row["strategy"] for row in rows}
    assert {"left_cluster", "central_cluster", "evenly_spaced", "random"}.issubset(strategies)
    assert all(0.0 <= row["rescued_fraction"] <= 1.0 for row in rows)
