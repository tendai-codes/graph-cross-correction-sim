"""Visualisation helpers for biologically framed rescue experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_metric_by_category(rows, category_key, metric_key, title, output_path):
    """Create a bar plot from experiment rows."""
    labels = [str(row[category_key]) for row in rows]
    values = [float(row[metric_key]) for row in rows]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
