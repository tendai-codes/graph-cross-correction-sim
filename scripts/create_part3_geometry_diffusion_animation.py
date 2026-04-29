# This script creates a GIF showing how signal spreads from corrected nuclei
# across a geometry-derived graph. It is intended as a visual explanation of
# Part 3: spatial graph construction and graph-based diffusion.

"""Create geometry-based signal diffusion animation for Part 3."""

import numpy as np

from src.part3_animation import (
    animate_geometry_diffusion,
    simulate_geometry_diffusion,
)


def main():
    np.random.seed(42)

    n_nuclei = 40

    positions = np.column_stack(
        [
            np.linspace(0, 1, n_nuclei),
            0.15 * np.random.randn(n_nuclei),
        ]
    )

    corrected_indices = [5, 18, 31]

    A, history, _ = simulate_geometry_diffusion(
        positions=positions,
        corrected_indices=corrected_indices,
        radius=0.18,
        alpha=0.25,
        beta=0.04,
        dt=0.1,
        steps=120,
        source_strength=1.0,
    )

    output_path = animate_geometry_diffusion(
        positions=positions,
        A=A,
        history=history,
        corrected_indices=corrected_indices,
        output_path="outputs/part3_geometry_diffusion.gif",
        interval=80,
        threshold=0.35,
    )

    print(f"Saved animation to: {output_path}")


if __name__ == "__main__":
    main()