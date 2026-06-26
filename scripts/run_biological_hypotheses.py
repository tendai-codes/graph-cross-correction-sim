"""Run biologically framed hypothesis experiments and export CSV/figure outputs."""

from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.biological_experiments import (
    run_clustered_vs_distributed_experiment,
    run_fibre_length_scaling_experiment,
    run_fiedler_validation,
    run_protein_stability_sweep,
    run_random_placement_statistics,
    run_transduction_efficiency_sweep,
    write_rows_to_csv,
)
from src.biological_visualisations import plot_metric_by_category

OUTPUT_DIR = Path("outputs/biological_hypotheses")


def main() -> None:
    """Run all biological hypothesis experiments and save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clustered_rows = run_clustered_vs_distributed_experiment()
    efficiency_rows = run_transduction_efficiency_sweep()
    stability_rows = run_protein_stability_sweep()
    length_rows = run_fibre_length_scaling_experiment()
    random_stats = run_random_placement_statistics(n_trials=250)
    fiedler_stats = run_fiedler_validation(n_graphs=100)

    write_rows_to_csv(clustered_rows, OUTPUT_DIR / "clustered_vs_distributed.csv")
    write_rows_to_csv(efficiency_rows, OUTPUT_DIR / "transduction_efficiency.csv")
    write_rows_to_csv(stability_rows, OUTPUT_DIR / "protein_stability.csv")
    write_rows_to_csv(length_rows, OUTPUT_DIR / "fibre_length_scaling.csv")
    write_rows_to_csv([random_stats], OUTPUT_DIR / "random_placement_statistics.csv")
    write_rows_to_csv([fiedler_stats], OUTPUT_DIR / "fiedler_validation.csv")

    summary_rows = clustered_rows + efficiency_rows + stability_rows + length_rows + [random_stats, fiedler_stats]
    write_rows_to_csv(summary_rows, OUTPUT_DIR / "summary_all_experiments.csv")

    plot_metric_by_category(
        clustered_rows,
        category_key="strategy",
        metric_key="rescued_fraction",
        title="Clustered vs Distributed Correction",
        output_path=OUTPUT_DIR / "clustered_vs_distributed.png",
    )
    plot_metric_by_category(
        efficiency_rows,
        category_key="corrected_fraction_requested",
        metric_key="rescued_fraction",
        title="Gene Correction Efficiency Sweep",
        output_path=OUTPUT_DIR / "transduction_efficiency.png",
    )
    plot_metric_by_category(
        stability_rows,
        category_key="beta",
        metric_key="rescued_fraction",
        title="Protein Stability / Decay Sweep",
        output_path=OUTPUT_DIR / "protein_stability.png",
    )
    plot_metric_by_category(
        length_rows,
        category_key="n_nuclei",
        metric_key="rescued_fraction",
        title="Fibre Size Scaling",
        output_path=OUTPUT_DIR / "fibre_length_scaling.png",
    )

    print("Biological hypothesis experiments complete.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
