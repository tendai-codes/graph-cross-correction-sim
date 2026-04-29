"""Animation utilities for Part 4 parameter sensitivity experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.part3_animation import simulate_geometry_diffusion


def animate_parameter_sensitivity(
    positions,
    corrected_indices,
    beta_values=(0.01, 0.04, 0.10),
    radius=0.18,
    alpha=0.25,
    dt=0.1,
    steps=120,
    threshold=0.35,
    output_path="outputs/part4_decay_sensitivity.gif",
    interval=90,
):
    """Animate how different decay rates affect rescue behaviour."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.asarray(positions, dtype=float)

    if positions.ndim == 1:
        positions = np.column_stack([positions, np.zeros_like(positions)])

    simulations = []

    for beta in beta_values:
        A, history, _ = simulate_geometry_diffusion(
            positions=positions,
            corrected_indices=corrected_indices,
            radius=radius,
            alpha=alpha,
            beta=beta,
            dt=dt,
            steps=steps,
        )
        simulations.append((beta, A, history))

    vmax = max(float(np.max(history)) for _, _, history in simulations)
    vmax = max(vmax, 1e-8)

    x = positions[:, 0]
    y = positions[:, 1]

    fig, axes = plt.subplots(1, len(beta_values), figsize=(5 * len(beta_values), 5))

    if len(beta_values) == 1:
        axes = [axes]

    def draw_edges(ax, A):
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[1]):
                if A[i, j] > 0:
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        linewidth=0.5,
                        alpha=0.30,
                    )

    def update(frame):
        for ax, (beta, A, history) in zip(axes, simulations):
            ax.clear()
            draw_edges(ax, A)

            values = history[frame]
            rescued = values >= threshold

            ax.scatter(
                x,
                y,
                c=values,
                s=85,
                vmin=0,
                vmax=vmax,
                edgecolors="black",
                linewidths=0.35,
            )

            ax.scatter(
                x[list(corrected_indices)],
                y[list(corrected_indices)],
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidths=1.6,
                label="Corrected",
            )

            ax.scatter(
                x[rescued],
                y[rescued],
                s=220,
                facecolors="none",
                edgecolors="black",
                linewidths=1.1,
                alpha=0.5,
                label="Rescued",
            )

            rescue_fraction = rescued.mean()

            ax.set_title(
                f"β = {beta:.2f}\n"
                f"Rescue fraction = {rescue_fraction:.2f}"
            )
            ax.set_xlabel("x-position")
            ax.set_ylabel("y-position")
            ax.set_aspect("equal", adjustable="box")

        fig.suptitle(f"Part 4: Decay sensitivity | step {frame}", fontsize=14)

        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=steps + 1,
        interval=interval,
        blit=False,
    )

    animation.save(output_path, writer=PillowWriter(fps=max(1, 1000 // interval)))
    plt.close(fig)

    return output_path