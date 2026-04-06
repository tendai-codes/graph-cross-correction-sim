# week1_sim.py
from __future__ import annotations

def run_naive_line(n_nodes=20, corrected_fraction=0.1, steps=50):
    signal = [0.0 for _ in range(n_nodes)]
    corrected = [False for _ in range(n_nodes)]
    n_corrected = max(1, round(n_nodes * corrected_fraction))
    for i in range(n_corrected):
        corrected[i] = True

    rescued = [False for _ in range(n_nodes)]

    for _ in range(steps):
        next_signal = signal[:]
        for i in range(n_nodes):
            left = signal[i - 1] if i > 0 else signal[i]
            right = signal[i + 1] if i < n_nodes - 1 else signal[i]
            production = 1.0 if corrected[i] else 0.0
            next_signal[i] = 0.6 * signal[i] + 0.2 * left + 0.2 * right + production * 0.05
            if next_signal[i] > 1.0:
                rescued[i] = True
        signal = next_signal

    return {"signal": signal, "rescued": rescued}
