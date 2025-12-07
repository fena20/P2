from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import EnvelopeConfig, PipelineConfig
from .io_utils import first_existing_column

BTU_PER_MMBTU = 1_000_000


@dataclass
class PreprocessResult:
    dataframe: pd.DataFrame
    microdata_columns: list[str]
    dropped_fraction: float


def _select_series(df: pd.DataFrame, candidates: list[str], label: str) -> pd.Series:
    column = first_existing_column(df, candidates)
    if column is None:
        raise KeyError(
            f"None of the candidate columns for {label} were found. "
            f"Tried: {', '.join(candidates)}"
        )
    return df[column]


def compute_envelope_score(row: pd.Series, cfg: EnvelopeConfig) -> Tuple[str, int]:
    """Return envelope class and raw score."""

    score = 0

    year = row.get("year_built")
    if pd.notnull(year):
        if year >= cfg.recent_year_threshold:
            score += 2
        elif year >= cfg.mid_year_threshold:
            score += 1
        else:
            score -= 1

    draftiness = row.get("draftiness")
    if pd.notnull(draftiness):
        if int(draftiness) in cfg.drafty_good:
            score += 1
        elif int(draftiness) in cfg.drafty_poor:
            score -= 1

    windows = row.get("window_quality")
    if pd.notnull(windows):
        if int(windows) in cfg.window_good_codes:
            score += 1
        elif int(windows) in cfg.window_poor_codes:
            score -= 1

    floor_area = row.get("heated_floor_area_sqft", 0)
    if pd.notnull(floor_area) and floor_area > 2500:
        score -= 1

    for label, bin_range in cfg.score_bins.items():
        if score in bin_range:
            return label, score
    return "medium", score


def categorize_hdd(series: pd.Series) -> pd.Series:
    """Categorize HDD65 into low/medium/high bands."""

    quantiles = series.dropna().quantile([0.33, 0.66])
    bins = [-np.inf, quantiles.iloc[0], quantiles.iloc[1], np.inf]
    labels = ["low", "medium", "high"]
    hdd_band = pd.cut(series, bins=bins, labels=labels)
    return hdd_band.astype(str).fillna("unknown")


def preprocess_recs_microdata(df: pd.DataFrame, config: PipelineConfig) -> PreprocessResult:
    """
    Apply all preprocessing steps needed for downstream analysis.
    """

    original_rows = len(df)
    df = df.copy()

    if "FUELHEAT" not in df.columns:
        raise KeyError(
            "Column 'FUELHEAT' not found. Ensure you loaded the full RECS microdata."
        )
    if config.model.weight_col not in df.columns:
        raise KeyError(
            f"Column '{config.model.weight_col}' not found. "
            "Refer to the RECS 2020 documentation for weighting variables."
        )

    # rename for clarity
    df["HOUSING_TYPE"] = df.get("TYPEHUQ", df.get("TYPEHUQ1", np.nan))
    df["year_built"] = df.get("YEARMADE")
    df["draftiness"] = df.get("DRAFTY")
    df["window_quality"] = df.get("WINDOWS")
    df["household_income"] = df.get("MONEYPY", df.get("HINCP", np.nan))

    # Filter to natural gas heated dwellings
    gas_mask = df["FUELHEAT"].isin(["Natural gas", "NATURAL GAS", 2, 3])
    df = df[gas_mask].copy()

    # Floor area
    floor_series = _select_series(df, config.floor_area_candidates, "floor area")
    df["heated_floor_area_sqft"] = pd.to_numeric(floor_series, errors="coerce")

    # Heating energy (BTU) -> convert to MMBtu
    heat_series = _select_series(df, config.heat_energy_candidates, "heating energy")
    df["baseline_heating_energy_mmbtu"] = (
        pd.to_numeric(heat_series, errors="coerce") / BTU_PER_MMBTU
    )

    # Occupants and income proxies
    occupant_series = _select_series(df, config.occupant_candidates, "occupants")
    df["occupants"] = pd.to_numeric(occupant_series, errors="coerce")
    income_series = _select_series(df, config.income_candidates, "income")

    # Some files encode income as brackets; attempt to coerce numeric
    df["household_income"] = pd.to_numeric(income_series, errors="coerce")

    # Derived metrics
    df["thermal_intensity_btu_per_sqft_hdd"] = (
        df["baseline_heating_energy_mmbtu"] * BTU_PER_MMBTU
    ) / (
        df["heated_floor_area_sqft"] * df["HDD65"].replace(0, np.nan)
    )
    df["thermal_intensity_btu_per_sqft_hdd"] = df[
        "thermal_intensity_btu_per_sqft_hdd"
    ].replace([np.inf, -np.inf], np.nan)

    # Envelope classes
    envelope_results = df.apply(
        lambda row: compute_envelope_score(row, config.envelope), axis=1
    )
    df["envelope_class"] = [cls for cls, _ in envelope_results]
    df["envelope_score"] = [score for _, score in envelope_results]

    df["hdd_band"] = categorize_hdd(df["HDD65"])

    # Drop rows with missing fundamentals
    df = df.dropna(
        subset=[
            "heated_floor_area_sqft",
            "baseline_heating_energy_mmbtu",
            "thermal_intensity_btu_per_sqft_hdd",
            "HDD65",
            config.model.weight_col,
        ]
    )

    dropped_fraction = 1 - len(df) / max(original_rows, 1)

    return PreprocessResult(
        dataframe=df,
        microdata_columns=list(df.columns),
        dropped_fraction=dropped_fraction,
    )
