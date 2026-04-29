# This script creates a GIF comparing instantaneous rescue with cumulative
# exposure rescue. It is intended as a visual explanation of
# Part 5: cumulative exposure rescue modelling.


"""Create Part 5 instantaneous vs cumulative rescue animation."""

import numpy as np

from src.part5_animation import animate_instantaneous_vs_cumulative_rescue


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

    output_path = animate_instantaneous_vs_cumulative_rescue(
        positions=positions,
        corrected_indices=corrected_indices,
        radius=0.18,
        alpha=0.25,
        beta=0.04,
        dt=0.1,
        steps=120,
        instantaneous_threshold=0.35,
        cumulative_threshold=2.5,
    )

    print(f"Saved animation to: {output_path}")


if __name__ == "__main__":
    main()