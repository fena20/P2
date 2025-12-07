from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from .config import PipelineConfig


PRICE_TIERS = {
    "low": 0.85,
    "medium": 1.0,
    "high": 1.2,
}

HDD_TIERS = {
    "mild": 0.9,
    "base": 1.0,
    "severe": 1.15,
}

STATUS_PRIORITY = {
    "double_win": 3,
    "economic_only": 2,
    "emissions_only": 1,
    "no_go": 0,
}

DIVISION_TO_STATES = {
    "New England": ["ME", "NH", "VT", "MA", "RI", "CT"],
    "Middle Atlantic": ["NY", "NJ", "PA"],
    "East North Central": ["WI", "MI", "IL", "IN", "OH"],
    "West North Central": ["ND", "SD", "NE", "KS", "MN", "IA", "MO"],
    "South Atlantic": ["DE", "MD", "DC", "VA", "WV", "NC", "SC", "GA", "FL"],
    "East South Central": ["KY", "TN", "MS", "AL"],
    "West South Central": ["OK", "AR", "TX", "LA"],
    "Mountain": ["MT", "ID", "WY", "NV", "UT", "CO", "AZ", "NM"],
    "Pacific": ["WA", "OR", "CA", "AK", "HI"],
}


def baseline_cost_emissions(archetypes: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Compute baseline cost and emissions for each archetype."""

    data = archetypes.copy()
    data["baseline_cost_usd"] = (
        data["baseline_mmbtu"] * config.scenario.gas_price_per_mmbtu
    )
    data["baseline_emissions_kg"] = (
        data["baseline_mmbtu"] * config.scenario.gas_co2_kg_per_mmbtu
    )
    return data[
        [
            "DIVISION",
            "envelope_class",
            "baseline_mmbtu",
            "baseline_cost_usd",
            "baseline_emissions_kg",
        ]
    ]


def evaluate_viability(
    nsga_df: pd.DataFrame,
    archetypes: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Return tipping status for every division/envelope across price + HDD tiers."""

    archetype_baseline = baseline_cost_emissions(archetypes, config)

    records = []
    for _, baseline_row in archetype_baseline.iterrows():
        division = baseline_row["DIVISION"]
        envelope = baseline_row["envelope_class"]
        subset = nsga_df[
            (nsga_df["DIVISION"] == division) & (nsga_df["envelope_class"] == envelope)
        ]
        if subset.empty:
            continue

        for price_label, price_factor in PRICE_TIERS.items():
            for hdd_label, hdd_factor in HDD_TIERS.items():
                elec_price = config.scenario.electricity_price_per_kwh * price_factor
                gas_price = config.scenario.gas_price_per_mmbtu * price_factor

                baseline_cost = baseline_row["baseline_cost_usd"] * price_factor * hdd_factor
                baseline_emissions = baseline_row["baseline_emissions_kg"] * hdd_factor

                status = "no_go"

                for _, scenario in subset.iterrows():
                    annualized_capex = scenario["annualized_capex_usd"]
                    electricity_kwh = scenario["electricity_kwh"] * hdd_factor
                    gas_mmbtu = scenario["gas_mmbtu"] * hdd_factor

                    cost = (
                        annualized_capex
                        + electricity_kwh * elec_price
                        + gas_mmbtu * gas_price
                    )
                    emissions = (
                        electricity_kwh * config.scenario.grid_co2_kg_per_kwh
                        + gas_mmbtu * config.scenario.gas_co2_kg_per_mmbtu
                    )

                    cost_better = cost <= baseline_cost
                    emission_better = emissions <= baseline_emissions

                    scenario_status = "no_go"
                    if cost_better and emission_better:
                        scenario_status = "double_win"
                    elif cost_better:
                        scenario_status = "economic_only"
                    elif emission_better:
                        scenario_status = "emissions_only"

                    if STATUS_PRIORITY[scenario_status] > STATUS_PRIORITY[status]:
                        status = scenario_status

                records.append(
                    {
                        "DIVISION": division,
                        "envelope_class": envelope,
                        "electricity_price_tier": price_label,
                        "hdd_tier": hdd_label,
                        "status": status,
                    }
                )

    return pd.DataFrame(records)


def plot_tipping_heatmaps(
    tipping_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Figure 9: heatmap panels by envelope class."""

    envelope_classes = sorted(tipping_df["envelope_class"].unique())
    fig, axes = plt.subplots(1, len(envelope_classes), figsize=(5 * len(envelope_classes), 4))

    if len(envelope_classes) == 1:
        axes = [axes]

    status_to_value = {name: idx for idx, name in enumerate(STATUS_PRIORITY.keys())}
    cmap = plt.get_cmap("RdYlGn")

    for ax, envelope in zip(axes, envelope_classes):
        subset = tipping_df[tipping_df["envelope_class"] == envelope]
        pivot = subset.pivot_table(
            index="hdd_tier",
            columns="electricity_price_tier",
            values="status",
            aggfunc="first",
        )
        numeric = pivot.replace(status_to_value)
        im = ax.imshow(numeric, cmap=cmap, vmin=0, vmax=3)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Envelope: {envelope}")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                status = pivot.iloc[i, j]
                ax.text(j, i, status or "-", ha="center", va="center", color="black")

    fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], label="Status priority")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_division_map(
    tipping_df: pd.DataFrame,
    output_path: Path,
    price_tier: str = "medium",
    hdd_tier: str = "base",
) -> Path | None:
    """Figure 10: US choropleth by division viability."""

    subset = tipping_df[
        (tipping_df["electricity_price_tier"] == price_tier)
        & (tipping_df["hdd_tier"] == hdd_tier)
    ]
    if subset.empty:
        return None

    rows = []
    for _, row in subset.iterrows():
        division = row["DIVISION"]
        status = row["status"]
        for state in DIVISION_TO_STATES.get(division, []):
            rows.append({"state": state, "status": status})

    state_df = pd.DataFrame(rows)
    if state_df.empty:
        return None

    fig = px.choropleth(
        state_df,
        locations="state",
        color="status",
        locationmode="USA-states",
        color_discrete_map={
            "double_win": "#1a9641",
            "economic_only": "#a6d96a",
            "emissions_only": "#fdae61",
            "no_go": "#d7191c",
        },
        scope="usa",
        title=f"Heat Pump Viability ({price_tier.title()} price, {hdd_tier} HDD)",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), scale=2)
    return output_path
