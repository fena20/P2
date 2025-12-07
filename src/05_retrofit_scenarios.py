#!/usr/bin/env python3
"""Step 5 – Retrofit and heat pump scenario assumptions."""

from __future__ import annotations

import argparse

from recs2020_heatpump import default_pipeline_config
from recs2020_heatpump.io_utils import load_dataset
from recs2020_heatpump.retrofit import (
    default_heat_pumps,
    default_retrofit_measures,
    generate_retrofit_results,
    heatpumps_to_table,
    measures_to_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate retrofit/heat pump scenario tables and outputs."
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional limit on number of scenario rows (useful for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()
    df = load_dataset(config.paths.data_cache / "recs2020_gas_heating.parquet")

    measures = default_retrofit_measures()
    hp_options = default_heat_pumps()

    scenarios = generate_retrofit_results(df, config, measures, hp_options)
    if args.limit_rows:
        scenarios = scenarios.head(args.limit_rows)

    measures_table = measures_to_table(measures)
    hp_table = heatpumps_to_table(hp_options)

    measures_path = config.paths.tables_dir / "table5_retrofit_measures.csv"
    hp_path = config.paths.tables_dir / "table5_heatpump_options.csv"
    scenarios_path = config.paths.tables_dir / "retrofit_scenarios_results.csv"

    measures_table.to_csv(measures_path, index=False)
    hp_table.to_csv(hp_path, index=False)
    scenarios.to_csv(scenarios_path, index=False)

    print("✅ Retrofit scenarios evaluated.")
    print(f"   Table 5 (measures) -> {measures_path}")
    print(f"   Table 5 (HP)       -> {hp_path}")
    print(f"   Scenario results   -> {scenarios_path}")


if __name__ == "__main__":
    main()
