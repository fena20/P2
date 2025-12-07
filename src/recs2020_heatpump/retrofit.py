from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence

import pandas as pd

from .config import PipelineConfig
from .descriptives import weighted_mean

MMBTU_TO_KWH = 293.07107


@dataclass(frozen=True)
class RetrofitMeasure:
    name: str
    load_reduction_pct: float  # expressed as fraction (0.05 = 5%)
    capex_per_sqft: float
    lifetime_years: int
    notes: str = ""


@dataclass(frozen=True)
class HeatPumpOption:
    name: str
    cop_heating: float
    capex_per_sqft: float
    lifetime_years: int
    min_temp_f: float
    backup_fraction: float = 0.0
    notes: str = ""


def default_retrofit_measures() -> List[RetrofitMeasure]:
    return [
        RetrofitMeasure(
            name="air_sealing",
            load_reduction_pct=0.08,
            capex_per_sqft=1.5,
            lifetime_years=12,
            notes="Envelope tightening and duct sealing",
        ),
        RetrofitMeasure(
            name="attic_insulation",
            load_reduction_pct=0.12,
            capex_per_sqft=2.8,
            lifetime_years=30,
            notes="Bring attic to R-49 equivalent",
        ),
        RetrofitMeasure(
            name="wall_insulation",
            load_reduction_pct=0.09,
            capex_per_sqft=3.5,
            lifetime_years=35,
            notes="Dense-pack or exterior insulation board",
        ),
        RetrofitMeasure(
            name="window_upgrade",
            load_reduction_pct=0.07,
            capex_per_sqft=6.0,
            lifetime_years=25,
            notes="Double/triple pane retrofit kits",
        ),
    ]


def default_heat_pumps() -> List[HeatPumpOption]:
    return [
        HeatPumpOption(
            name="baseline_HP",
            cop_heating=3.0,
            capex_per_sqft=12.0,
            lifetime_years=15,
            min_temp_f=25,
            backup_fraction=0.2,
            notes="Standard cold climate minisplit with aux gas",
        ),
        HeatPumpOption(
            name="cold_climate_HP",
            cop_heating=3.5,
            capex_per_sqft=15.0,
            lifetime_years=18,
            min_temp_f=5,
            backup_fraction=0.05,
            notes="Cold climate variable-speed heat pump",
        ),
    ]


def capital_recovery_factor(rate: float, years: int) -> float:
    """Return CRF used to annualize capital investments."""

    if rate == 0:
        return 1 / years
    numerator = rate * (1 + rate) ** years
    denominator = (1 + rate) ** years - 1
    return numerator / denominator


def build_archetypes(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Aggregate households into division x envelope archetypes."""

    records = []
    for (division, envelope), frame in df.groupby(["DIVISION", "envelope_class"]):
        weights = frame[config.model.weight_col]
        records.append(
            {
                "DIVISION": division,
                "envelope_class": envelope,
                "representative_households": weights.sum(),
                "heated_sqft": weighted_mean(frame["heated_floor_area_sqft"], weights),
                "baseline_intensity": weighted_mean(
                    frame[config.model.target_col], weights
                ),
                "baseline_mmbtu": weighted_mean(
                    frame["baseline_heating_energy_mmbtu"], weights
                ),
                "HDD65": weighted_mean(frame["HDD65"], weights),
                "electricity_price": config.scenario.electricity_price_per_kwh,
                "gas_price": config.scenario.gas_price_per_mmbtu,
            }
        )
    return pd.DataFrame(records)


def measure_combinations(
    measures: Sequence[RetrofitMeasure],
) -> List[List[RetrofitMeasure]]:
    """Return all measure combinations (including empty set)."""

    combos: List[List[RetrofitMeasure]] = [[]]
    for r in range(1, len(measures) + 1):
        for combo in combinations(measures, r):
            combos.append(list(combo))
    return combos


def combined_reduction(measures: Sequence[RetrofitMeasure]) -> float:
    """Multiplicative energy reduction of combined measures."""

    remaining = 1.0
    for measure in measures:
        remaining *= 1 - measure.load_reduction_pct
    return 1 - remaining


def evaluate_scenario(
    archetype: pd.Series,
    measures: Sequence[RetrofitMeasure],
    hp_option: HeatPumpOption | None,
    config: PipelineConfig,
    electricity_price: float | None = None,
    gas_price: float | None = None,
) -> Dict[str, float | str]:
    """
    Compute annualized cost & emissions for a retrofit + HP bundle.
    """

    electricity_price = electricity_price or archetype["electricity_price"]
    gas_price = gas_price or archetype["gas_price"]

    baseline_load_mmbtu = archetype["baseline_mmbtu"]
    area = archetype["heated_sqft"]

    reduction = combined_reduction(measures)
    load_after_retrofit = baseline_load_mmbtu * (1 - reduction)

    retrofit_annualized = sum(
        capital_recovery_factor(config.scenario.discount_rate, m.lifetime_years)
        * m.capex_per_sqft
        * area
        for m in measures
    )

    if hp_option:
        hp_capex = hp_option.capex_per_sqft * area
        hp_annualized = (
            capital_recovery_factor(config.scenario.discount_rate, hp_option.lifetime_years)
            * hp_capex
        )

        electric_kwh = (load_after_retrofit * MMBTU_TO_KWH) / hp_option.cop_heating
        backup_mmbtu = load_after_retrofit * hp_option.backup_fraction
        gas_mmbtu = backup_mmbtu

        operating_cost = (
            electric_kwh * electricity_price
            + gas_mmbtu * gas_price
        )
        annualized_capex = retrofit_annualized + hp_annualized

        co2 = (
            electric_kwh * config.scenario.grid_co2_kg_per_kwh
            + gas_mmbtu * config.scenario.gas_co2_kg_per_mmbtu
        )
        option_label = hp_option.name
    else:
        electric_kwh = 0.0
        gas_mmbtu = load_after_retrofit
        operating_cost = gas_mmbtu * gas_price
        annualized_capex = retrofit_annualized
        co2 = gas_mmbtu * config.scenario.gas_co2_kg_per_mmbtu
        option_label = "gas_baseline"

    total_cost = operating_cost + annualized_capex

    return {
        "DIVISION": archetype["DIVISION"],
        "envelope_class": archetype["envelope_class"],
        "measures": "+".join(m.name for m in measures) if measures else "none",
        "heat_pump": option_label,
        "load_reduction_pct": reduction,
        "annual_cost_usd": total_cost,
        "operating_cost_usd": operating_cost,
        "annualized_capex_usd": annualized_capex,
        "annual_emissions_kg": co2,
        "electricity_kwh": electric_kwh,
        "gas_mmbtu": gas_mmbtu,
        "baseline_mmbtu": baseline_load_mmbtu,
        "load_after_retrofit_mmbtu": load_after_retrofit,
    }


def generate_retrofit_results(
    df: pd.DataFrame,
    config: PipelineConfig,
    measures: Sequence[RetrofitMeasure] | None = None,
    hp_options: Sequence[HeatPumpOption] | None = None,
) -> pd.DataFrame:
    """Evaluate all archetypes across retrofit + HP combinations."""

    measures = measures or default_retrofit_measures()
    hp_options = hp_options or default_heat_pumps()

    archetypes = build_archetypes(df, config)
    combos = measure_combinations(measures)

    rows = []
    for _, archetype in archetypes.iterrows():
        for combo in combos:
            for hp_option in [None] + list(hp_options):
                rows.append(
                    evaluate_scenario(
                        archetype=archetype,
                        measures=combo,
                        hp_option=hp_option,
                        config=config,
                    )
                )

    return pd.DataFrame(rows)


def measures_to_table(measures: Sequence[RetrofitMeasure]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "measure": m.name,
                "load_reduction_pct": m.load_reduction_pct * 100,
                "capex_per_sqft_usd": m.capex_per_sqft,
                "lifetime_years": m.lifetime_years,
                "notes": m.notes,
            }
            for m in measures
        ]
    )


def heatpumps_to_table(options: Sequence[HeatPumpOption]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "option": hp.name,
                "cop_heating": hp.cop_heating,
                "capex_per_sqft_usd": hp.capex_per_sqft,
                "lifetime_years": hp.lifetime_years,
                "backup_fraction": hp.backup_fraction,
                "notes": hp.notes,
            }
            for hp in options
        ]
    )
