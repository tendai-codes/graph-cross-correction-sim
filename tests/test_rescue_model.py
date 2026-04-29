import numpy as np

from src.part3_geometry import generate_positions
from src.part4_sweeps import instantaneous_rescue
from src.part5_cumulative import simulate_cumulative_exposure


def test_instantaneous_threshold_is_monotonic():
    signal = np.array([0.1, 0.3, 0.7, 1.2])

    low_threshold = instantaneous_rescue(signal, threshold=0.3)
    high_threshold = instantaneous_rescue(signal, threshold=0.8)

    assert low_threshold.sum() >= high_threshold.sum()


def test_cumulative_rescue_threshold_is_monotonic():
    positions = generate_positions(n_nodes=30, seed=7)

    low_threshold = simulate_cumulative_exposure(positions, exposure_threshold=0.5, steps=200)
    high_threshold = simulate_cumulative_exposure(positions, exposure_threshold=2.0, steps=200)

    assert low_threshold["rescued_fraction"] >= high_threshold["rescued_fraction"]


def test_rescue_history_never_decreases():
    positions = generate_positions(n_nodes=30, seed=7)
    result = simulate_cumulative_exposure(positions, exposure_threshold=1.0, steps=120)

    rescued_counts = result["rescued_history"].sum(axis=1)

    assert np.all(np.diff(rescued_counts) >= 0)


def test_simulation_is_reproducible_with_same_seed():
    positions = generate_positions(n_nodes=30, seed=7)
    first = simulate_cumulative_exposure(positions, correction_mode="random", seed=42)
    second = simulate_cumulative_exposure(positions, correction_mode="random", seed=42)

    assert np.allclose(first["signal"], second["signal"])
    assert np.allclose(first["exposure"], second["exposure"])
    assert np.array_equal(first["rescued"], second["rescued"])
