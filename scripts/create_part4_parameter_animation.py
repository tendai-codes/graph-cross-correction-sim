# This script creates a GIF comparing how different decay rates affect
# signal spread and rescue outcomes. It is intended as a visual explanation of
# Part 4: parameter sensitivity experiments.

"""Create Part 4 parameter sensitivity animation."""

import numpy as np

from src.part4_animation import animate_parameter_sensitivity


def main():
    np.random.seed(42)

    n_nuclei = 40

    positions = np.column_stack(
        [
            np.linspace(0, 1, n_nuclei),
            0.15 * np.random.randn(n_nuclei),
        ]
    )

    corrected_indices = np.array([5, 18, 31])

    output_path = animate_parameter_sensitivity(
        positions=positions,
        corrected_indices=corrected_indices,
        beta_values=(0.01, 0.04, 0.10),
        radius=0.18,
        alpha=0.25,
        dt=0.1,
        steps=120,
        threshold=0.35,
    )

    print(f"Saved animation to: {output_path}")


if __name__ == "__main__":
    main()