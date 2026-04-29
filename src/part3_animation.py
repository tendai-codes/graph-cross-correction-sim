"""Animation utilities for geometry-based signal diffusion."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.part2_laplacian import graph_laplacian
from src.part3_geometry import build_geometric_graph


def simulate_geometry_diffusion(
    positions,
    corrected_indices,
    radius=0.25,
    alpha=0.15,
    beta=0.03,
    dt=0.1,
    steps=100,
    source_strength=1.0,
):
    """Simulate signal diffusion over a geometry-based nuclear graph."""
    positions = np.asarray(positions, dtype=float)

    A = build_geometric_graph(positions, radius=radius)
    L = graph_laplacian(A)

    n_nodes = len(positions)

    q = np.zeros(n_nodes)
    q[corrected_indices] = source_strength

    u = np.zeros(n_nodes)
    history = [u.copy()]

    for _ in range(steps):
        diffusion = -alpha * (L @ u)
        decay = -beta * u

        u = u + dt * (diffusion + decay + q)
        u = np.maximum(u, 0.0)

        history.append(u.copy())

    return A, np.asarray(history), q


def animate_geometry_diffusion(
    positions,
    A,
    history,
    corrected_indices,
    output_path="outputs/create_part3_geometry_diffusion.gif",
    interval=80,
    threshold=None,
):
    """Create a GIF showing signal diffusion over spatially placed nuclei."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.asarray(positions, dtype=float)

    if positions.ndim == 1:
        positions = np.column_stack([positions, np.zeros_like(positions)])

    corrected_indices = set(corrected_indices)

    x = positions[:, 0]
    y = positions[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = max(float(np.max(history)), 1e-8)

    def draw_edges():
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[1]):
                if A[i, j] > 0:
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        linewidth=0.6,
                        alpha=0.35,
                    )

    def update(frame):
        ax.clear()
        draw_edges()

        values = history[frame]

        scatter = ax.scatter(
            x,
            y,
            c=values,
            s=90,
            vmin=0,
            vmax=vmax,
            edgecolors="black",
            linewidths=0.4,
        )

        corrected_x = [positions[i, 0] for i in corrected_indices]
        corrected_y = [positions[i, 1] for i in corrected_indices]

        ax.scatter(
            corrected_x,
            corrected_y,
            s=160,
            facecolors="none",
            edgecolors="black",
            linewidths=1.6,
            label="Corrected nuclei",
        )

        if threshold is not None:
            rescued = values >= threshold
            ax.scatter(
                x[rescued],
                y[rescued],
                s=230,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                alpha=0.5,
                label="Above rescue threshold",
            )

        ax.set_title(f"Geometry-based signal diffusion | step {frame}")
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")

        return scatter,

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval,
        blit=False,
    )

    animation.save(output_path, writer=PillowWriter(fps=max(1, 1000 // interval)))
    plt.close(fig)

    return output_path