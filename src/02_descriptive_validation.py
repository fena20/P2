#!/usr/bin/env python3
"""Step 2 – Descriptive statistics and macro validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from recs2020_heatpump import default_pipeline_config
from recs2020_heatpump.descriptives import (
    build_table2,
    compare_with_official,
    macro_validation_metrics,
    plot_climate_envelope_overview,
    plot_macro_validation,
    plot_thermal_intensity_distribution,
)
from recs2020_heatpump.io_utils import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate descriptive tables/figures for the RECS workflow."
    )
    parser.add_argument(
        "--official-table",
        type=Path,
        default=None,
        help=(
            "Optional CSV with columns [metric, official_value] for macro validation. "
            "Defaults to `data/recs2020_official_validation.csv` if present."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()

    dataset_path = config.paths.data_cache / "recs2020_gas_heating.parquet"
    df = load_dataset(dataset_path)

    table2 = build_table2(df, config)
    table2_path = config.paths.tables_dir / "table2_weighted_characteristics.csv"
    table2.to_csv(table2_path, index=False)

    derived = macro_validation_metrics(df, config)
    official_path = (
        args.official_table
        if args.official_table
        else config.paths.data_dir / "recs2020_official_validation.csv"
    )
    comparison, used_official = compare_with_official(derived, official_path)
    table8_path = config.paths.tables_dir / "table8_macro_validation.csv"
    comparison.to_csv(table8_path, index=False)

    figure2_path = config.paths.figures_dir / "figure2_climate_envelope.png"
    figure3_path = config.paths.figures_dir / "figure3_thermal_intensity.png"
    figure4_path = config.paths.figures_dir / "figure4_macro_validation.png"

    plot_climate_envelope_overview(df, config, figure2_path)
    plot_thermal_intensity_distribution(df, config, figure3_path)
    if not comparison["official_value"].isna().all():
        plot_macro_validation(comparison, figure4_path)

    print("✅ Descriptive statistics generated.")
    print(f"   Table 2 -> {table2_path}")
    print(f"   Table 8 -> {table8_path}")
    generated_figs = [figure2_path.name, figure3_path.name]
    if not comparison["official_value"].isna().all():
        generated_figs.append(figure4_path.name)
    print(f"   Figures -> {generated_figs}")
    if used_official:
        print(f"   Official comparison used file: {used_official}")
    else:
        print("   Official reference file not found; Figure 4 skipped.")


if __name__ == "__main__":
    main()
