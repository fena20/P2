from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PipelineConfig


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_share(df: pd.DataFrame, column: str, weight_col: str) -> Dict[str, float]:
    data = df[[column, weight_col]].dropna()
    if data.empty:
        return {}
    grouped = (
        data.groupby(column)[weight_col].sum().sort_values(ascending=False)
    )
    total = grouped.sum()
    shares = (grouped / total).to_dict()
    return shares


HOUSING_AGG_MAP = {
    1: "single_family",
    2: "single_family_attached",
    3: "multi_family_lowrise",
    4: "multi_family_highrise",
    5: "manufactured",
    "Single-Family Detached": "single_family",
    "Single-Family Attached": "single_family_attached",
    "Apartment (2-4 units)": "multi_family_lowrise",
    "Apartment (5+ units)": "multi_family_highrise",
    "Mobile home": "manufactured",
}


def map_housing_category(value) -> str:
    return HOUSING_AGG_MAP.get(value, "other")


def build_table2(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Weighted descriptive statistics by division and envelope class."""

    records = []
    for (division, envelope), frame in df.groupby(["DIVISION", "envelope_class"]):
        weights = frame[config.model.weight_col]
        housing = frame["HOUSING_TYPE"].apply(map_housing_category)
        frame = frame.assign(HOUSING_AGG=housing)
        housing_shares = weighted_share(frame, "HOUSING_AGG", config.model.weight_col)

        record = {
            "DIVISION": division,
            "envelope_class": envelope,
            "weighted_dwellings": weights.sum(),
            "mean_heated_sqft": weighted_mean(frame["heated_floor_area_sqft"], weights),
            "mean_HDD65": weighted_mean(frame["HDD65"], weights),
        }

        for category in ["single_family", "single_family_attached", "multi_family_lowrise", "multi_family_highrise", "manufactured", "other"]:
            record[f"share_{category}"] = housing_shares.get(category, 0.0)

        records.append(record)

    result = pd.DataFrame(records).sort_values(["DIVISION", "envelope_class"])
    return result


def macro_validation_metrics(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Aggregate metrics to compare with official tables."""

    weights = df[config.model.weight_col]
    total_weight = weights.sum()

    metrics = {
        "avg_heated_sqft": weighted_mean(df["heated_floor_area_sqft"], weights),
        "avg_heating_mmbtu": weighted_mean(df["baseline_heating_energy_mmbtu"], weights),
        "avg_HDD65": weighted_mean(df["HDD65"], weights),
        "share_poor_envelope": (weights[df["envelope_class"] == "poor"].sum() / total_weight),
        "share_good_envelope": (weights[df["envelope_class"] == "good"].sum() / total_weight),
        "share_single_family": (
            weights[df["HOUSING_TYPE"].apply(map_housing_category) == "single_family"].sum()
            / total_weight
        ),
    }

    return pd.DataFrame(
        [{"metric": name, "value": float(val)} for name, val in metrics.items()]
    )


def compare_with_official(
    derived_metrics: pd.DataFrame,
    official_path: Path,
) -> Tuple[pd.DataFrame, Path | None]:
    """
    Join derived metrics with official ones if provided.

    The official CSV is expected to have columns: metric, official_value.
    """

    if not official_path.exists():
        return derived_metrics.assign(official_value=np.nan, pct_diff=np.nan), None

    official = pd.read_csv(official_path)
    merged = derived_metrics.merge(official, on="metric", how="left")
    merged["pct_diff"] = (
        (merged["value"] - merged["official_value"]) / merged["official_value"]
    ) * 100
    return merged, official_path


def plot_climate_envelope_overview(
    df: pd.DataFrame, config: PipelineConfig, output_path: Path
) -> Path:
    """Generate Figure 2 (HDD distribution + envelope shares)."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(
        data=df,
        x="DIVISION",
        y="HDD65",
        ax=axes[0],
        showfliers=False,
    )
    axes[0].set_title("HDD65 Distribution by Division")
    axes[0].tick_params(axis="x", rotation=45)

    envelope_weights = (
        df.groupby("envelope_class")[config.model.weight_col].sum().reset_index()
    )
    axes[1].pie(
        envelope_weights[config.model.weight_col],
        labels=envelope_weights["envelope_class"],
        autopct="%1.1f%%",
    )
    axes[1].set_title("Envelope Class Shares")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_thermal_intensity_distribution(
    df: pd.DataFrame, config: PipelineConfig, output_path: Path
) -> Path:
    """Generate Figure 3 (thermal intensity distribution)."""

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="envelope_class",
        y=config.model.target_col,
        hue="hdd_band",
        ax=ax,
    )
    ax.set_ylabel("Thermal Intensity [BTU / (ft²·HDD)]")
    ax.set_xlabel("Envelope Class")
    ax.set_title("Thermal Intensity by Envelope Class and Climate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_macro_validation(
    comparison_df: pd.DataFrame, output_path: Path
) -> Path | None:
    """Generate Figure 4 if official data exists."""

    if comparison_df["official_value"].isna().all():
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    idx = np.arange(len(comparison_df))
    ax.bar(idx - width / 2, comparison_df["value"], width, label="This study")
    ax.bar(
        idx + width / 2,
        comparison_df["official_value"],
        width,
        label="Official RECS",
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(comparison_df["metric"], rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Macro Validation vs RECS Tables")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
