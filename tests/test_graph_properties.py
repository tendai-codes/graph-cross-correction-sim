import numpy as np

from src.part2_laplacian import graph_laplacian, line_adjacency
from src.part3_geometry import build_radius_adjacency, corrected_pattern, generate_positions


def test_radius_adjacency_is_symmetric_and_has_zero_diagonal():
    positions = generate_positions(n_nodes=20, seed=11)
    A = build_radius_adjacency(positions, radius=0.12)

    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 0.0)


def test_laplacian_has_zero_row_sums():
    A = line_adjacency(10)
    L = graph_laplacian(A)

    assert np.allclose(L.sum(axis=1), 0.0)


def test_corrected_pattern_is_deterministic_for_fixed_seed():
    first = corrected_pattern(30, corrected_fraction=0.2, mode="random", seed=7)
    second = corrected_pattern(30, corrected_fraction=0.2, mode="random", seed=7)

    assert np.array_equal(first, second)
    assert first.sum() == 6
