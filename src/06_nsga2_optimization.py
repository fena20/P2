#!/usr/bin/env python3
"""Step 6 – NSGA-II optimization."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from recs2020_heatpump import default_pipeline_config
from recs2020_heatpump.io_utils import load_dataset
from recs2020_heatpump.optimization import run_nsga_pipeline
from recs2020_heatpump.retrofit import default_heat_pumps, default_retrofit_measures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSGA-II retrofit optimization.")
    parser.add_argument(
        "--limit-divisions",
        type=int,
        default=None,
        help="Optionally limit number of divisions processed (for quick tests).",
    )
    return parser.parse_args()


def plot_pareto_examples(nsga_df: pd.DataFrame, output_path):
    if nsga_df.empty:
        return None

    divisions = nsga_df["DIVISION"].unique()
    selected = divisions[:2]
    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, division in zip(axes, selected):
        subset = nsga_df[nsga_df["DIVISION"] == division]
        sc = ax.scatter(
            subset["annual_emissions_kg"],
            subset["annual_cost_usd"],
            c=subset["load_reduction_pct"],
            cmap="viridis",
            alpha=0.6,
        )
        ax.set_title(f"{division}")
        ax.set_xlabel("Annual CO₂ emissions (kg)")
        ax.set_ylabel("Annualized cost (USD)")
        fig.colorbar(sc, ax=ax, label="Load reduction share")

    fig.suptitle("Figure 8 – Example Pareto fronts")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()
    df = load_dataset(config.paths.data_cache / "recs2020_gas_heating.parquet")

    if args.limit_divisions:
        allowed_divisions = df["DIVISION"].dropna().unique()[: args.limit_divisions]
        df = df[df["DIVISION"].isin(allowed_divisions)]

    measures = default_retrofit_measures()
    hp_options = default_heat_pumps()

    nsga_results = run_nsga_pipeline(df, config, measures, hp_options)
    nsga_path = config.paths.tables_dir / "nsga_pareto_points.csv"
    if not nsga_results.empty:
        nsga_results.to_csv(nsga_path, index=False)

    config_rows = [
        {"parameter": "population", "value": config.nsga.population},
        {"parameter": "generations", "value": config.nsga.generations},
        {"parameter": "crossover_prob", "value": config.nsga.crossover_prob},
        {"parameter": "mutation_prob", "value": config.nsga.mutation_prob},
        {"parameter": "seed", "value": config.nsga.seed},
        {"parameter": "electricity_price", "value": config.scenario.electricity_price_per_kwh},
        {"parameter": "gas_price", "value": config.scenario.gas_price_per_mmbtu},
    ]
    table6_path = config.paths.tables_dir / "table6_nsga_config.csv"
    pd.DataFrame(config_rows).to_csv(table6_path, index=False)

    figure8_path = config.paths.figures_dir / "figure8_pareto_fronts.png"
    plot_pareto_examples(nsga_results, figure8_path)

    print("✅ NSGA-II optimization complete.")
    print(f"   Table 6 -> {table6_path}")
    if not nsga_results.empty:
        print(f"   Pareto points -> {nsga_path}")
        print(f"   Figure 8 -> {figure8_path}")
    else:
        print("   Warning: no NSGA solutions (check input data).")


if __name__ == "__main__":
    main()
