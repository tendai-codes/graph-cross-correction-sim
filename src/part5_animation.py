"""Animation utilities for Part 5 cumulative exposure rescue."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.part3_animation import simulate_geometry_diffusion


def compute_cumulative_exposure(history, dt=0.1):
    """Compute cumulative signal exposure over time."""
    return np.cumsum(history * dt, axis=0)


def animate_instantaneous_vs_cumulative_rescue(
    positions,
    corrected_indices,
    radius=0.18,
    alpha=0.25,
    beta=0.04,
    dt=0.1,
    steps=120,
    instantaneous_threshold=0.35,
    cumulative_threshold=2.5,
    output_path="outputs/part5_instant_vs_cumulative.gif",
    interval=90,
):
    """Compare instantaneous rescue against cumulative exposure rescue."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.asarray(positions, dtype=float)

    if positions.ndim == 1:
        positions = np.column_stack([positions, np.zeros_like(positions)])

    A, history, _ = simulate_geometry_diffusion(
        positions=positions,
        corrected_indices=corrected_indices,
        radius=radius,
        alpha=alpha,
        beta=beta,
        dt=dt,
        steps=steps,
    )

    exposure = compute_cumulative_exposure(history, dt=dt)

    x = positions[:, 0]
    y = positions[:, 1]

    vmax_signal = max(float(np.max(history)), 1e-8)
    vmax_exposure = max(float(np.max(exposure)), 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def draw_edges(ax):
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[1]):
                if A[i, j] > 0:
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        linewidth=0.5,
                        alpha=0.30,
                    )

    def draw_panel(ax, values, rescued, title, vmax):
        draw_edges(ax)

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

        ax.set_title(f"{title}\nRescue fraction = {rescued.mean():.2f}")
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        ax.set_aspect("equal", adjustable="box")

    def update(frame):
        for ax in axes:
            ax.clear()

        signal_values = history[frame]
        exposure_values = exposure[frame]

        instant_rescued = signal_values >= instantaneous_threshold
        cumulative_rescued = exposure_values >= cumulative_threshold

        draw_panel(
            axes[0],
            signal_values,
            instant_rescued,
            "Instantaneous threshold",
            vmax_signal,
        )

        draw_panel(
            axes[1],
            exposure_values,
            cumulative_rescued,
            "Cumulative exposure threshold",
            vmax_exposure,
        )

        fig.suptitle(f"Part 5: Instantaneous vs cumulative rescue | step {frame}")

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