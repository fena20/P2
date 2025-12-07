#!/usr/bin/env python3
"""Step 1 – Data preparation for the RECS 2020 heat pump workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from recs2020_heatpump import default_pipeline_config
from recs2020_heatpump.io_utils import (
    discover_microdata_path,
    load_microdata,
    save_dataset,
    save_json,
    variable_table_from_meta,
)
from recs2020_heatpump.preprocessing import preprocess_recs_microdata


REQUIRED_COLUMNS = [
    "DOEID",
    "NWEIGHT",
    "FUELHEAT",
    "HDD65",
    "TOTSQFT_EN",
    "TOTSQFT",
    "HEATEDSQFT",
    "BTUNG",
    "BTUHEAT",
    "BTUSE",
    "BTU_SPACE_HEAT",
    "TYPEHUQ",
    "TYPEHUQ1",
    "YEARMADE",
    "DRAFTY",
    "WINDOWS",
    "EQUIPM",
    "EQUIPAGE",
    "MONEYPY",
    "HINCP",
    "NUMHHS",
    "NHSLDMEM",
    "DIVISION",
    "REGIONC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RECS 2020 gas-heated dataset with engineered features."
    )
    parser.add_argument(
        "--microdata",
        type=Path,
        help="Path to recs2020_public CSV. Defaults to scanning the data directory.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="recs2020_public",
        help="Substring used to detect the microdata CSV (default: recs2020_public).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()

    micro_path = (
        args.microdata
        if args.microdata
        else discover_microdata_path(config.paths.data_dir, pattern=args.pattern)
    )

    columns_to_load = sorted(set(REQUIRED_COLUMNS))
    raw_df = load_microdata(micro_path, columns=columns_to_load)

    prep_result = preprocess_recs_microdata(raw_df, config)
    dataset_path = config.paths.data_cache / "recs2020_gas_heating.parquet"
    save_dataset(prep_result.dataframe, dataset_path)

    summary = {
        "microdata_path": str(micro_path),
        "rows_source": len(raw_df),
        "rows_clean": len(prep_result.dataframe),
        "dropped_fraction": prep_result.dropped_fraction,
        "columns_used": prep_result.microdata_columns,
    }
    save_json(summary, config.paths.data_cache / "data_prep_summary.json")

    table1 = variable_table_from_meta(config)
    table1_path = config.paths.tables_dir / "table1_variable_definitions.csv"
    table1.to_csv(table1_path, index=False)

    print("✅ Data preparation complete.")
    print(f"   Clean dataset -> {dataset_path}")
    print(f"   Table 1       -> {table1_path}")


if __name__ == "__main__":
    main()
