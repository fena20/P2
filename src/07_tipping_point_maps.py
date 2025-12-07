#!/usr/bin/env python3
"""Step 7 – Tipping point heatmaps and division maps."""

from __future__ import annotations

import argparse

import pandas as pd

from recs2020_heatpump import default_pipeline_config
from recs2020_heatpump.io_utils import load_dataset
from recs2020_heatpump.retrofit import build_archetypes
from recs2020_heatpump.tipping import (
    evaluate_viability,
    plot_division_map,
    plot_tipping_heatmaps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tipping point visualizations.")
    parser.add_argument(
        "--nsga-results",
        type=str,
        default="nsga_pareto_points.csv",
        help="Filename of NSGA results produced in step 6.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()

    dataset_path = config.paths.data_cache / "recs2020_gas_heating.parquet"
    df = load_dataset(dataset_path)

    nsga_path = config.paths.tables_dir / args.nsga_results
    if not nsga_path.exists():
        raise FileNotFoundError(
            f"NSGA results not found at {nsga_path}. Run 06_nsga2_optimization.py first."
        )
    nsga_df = pd.read_csv(nsga_path)

    archetypes = build_archetypes(df, config)
    tipping_df = evaluate_viability(nsga_df, archetypes, config)
    table7_path = config.paths.tables_dir / "table7_tipping_points.csv"
    tipping_df.to_csv(table7_path, index=False)

    figure9_path = config.paths.figures_dir / "figure9_tipping_heatmap.png"
    figure10_path = config.paths.figures_dir / "figure10_tipping_map.png"

    plot_tipping_heatmaps(tipping_df, figure9_path)
    map_path = plot_division_map(tipping_df, figure10_path)

    print("✅ Tipping point evaluation complete.")
    print(f"   Table 7 -> {table7_path}")
    print(f"   Figure 9 -> {figure9_path}")
    if map_path:
        print(f"   Figure 10 -> {map_path}")
    else:
        print("   Figure 10 skipped (missing data or kaleido).")


if __name__ == "__main__":
    main()
