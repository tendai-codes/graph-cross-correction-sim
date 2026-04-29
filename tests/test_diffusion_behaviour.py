import numpy as np

from src.part2_laplacian import diffusion_step, graph_laplacian, line_adjacency, simulate_laplacian_diffusion


def test_no_source_keeps_zero_signal_zero():
    A = line_adjacency(8)
    L = graph_laplacian(A)
    signal = np.zeros(8)
    source = np.zeros(8)

    next_signal = diffusion_step(signal, L, source, alpha=0.1, beta=0.03, dt=0.1)

    assert np.allclose(next_signal, 0.0)


def test_corrected_source_generates_signal():
    A = line_adjacency(8)
    corrected = np.zeros(8, dtype=bool)
    corrected[0] = True

    result = simulate_laplacian_diffusion(A, corrected, production_rate=0.05, steps=20)

    assert result["signal"][0] > 0.0
    assert result["signal"].sum() > 0.0


def test_higher_production_increases_total_signal():
    A = line_adjacency(8)
    corrected = np.zeros(8, dtype=bool)
    corrected[0] = True

    low = simulate_laplacian_diffusion(A, corrected, production_rate=0.02, steps=80)["signal"].sum()
    high = simulate_laplacian_diffusion(A, corrected, production_rate=0.08, steps=80)["signal"].sum()

    assert high > low
