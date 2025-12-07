"""
07_tipping_point_maps.py
=========================
Tipping Point Analysis and Mapping for Heat Pump Viability

This module:
1. Builds scenario grids (HDD × electricity price × envelope class)
2. Identifies economic and environmental tipping points
3. Generates visualization maps and heatmaps

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import from previous modules
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from retrofit_scenarios import (
        RETROFIT_MEASURES, HEAT_PUMP_OPTIONS,
        EnergyPrices, GridEmissionFactor, GRID_EMISSIONS,
        evaluate_scenario
    )
except ImportError:
    # Define locally if import fails
    pass


@dataclass
class TippingPointResult:
    """Result of tipping point analysis for a single scenario."""
    hdd: float
    electricity_price: float
    gas_price: float
    envelope_class: str
    baseline_cost: float
    baseline_emissions: float
    best_hp_cost: float
    best_hp_emissions: float
    best_hp_retrofit: str
    best_hp_type: str
    cost_savings: float
    emissions_reduction: float
    is_economically_viable: bool
    is_environmentally_beneficial: bool
    viability_class: str


def define_scenario_grid() -> Dict:
    """Define the scenario grid for tipping point analysis."""
    
    scenario_grid = {
        # HDD bands
        'hdd_values': [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500],
        
        # Electricity prices ($/kWh)
        'electricity_prices': [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28],
        
        # Gas prices ($/therm)
        'gas_prices': [0.80, 1.00, 1.20, 1.50],
        
        # Envelope classes
        'envelope_classes': ['poor', 'medium', 'good'],
        
        # Grid emission scenarios
        'grid_emissions': {
            'current': 0.42,  # kg CO2/kWh
            'moderate_decarb': 0.25,
            'high_decarb': 0.10,
        },
        
        # Representative floor area (sqft)
        'reference_sqft': 2000,
        
        # Reference thermal intensity by envelope class (BTU/sqft/HDD)
        'reference_intensity': {
            'poor': 12.0,
            'medium': 8.0,
            'good': 5.0,
        }
    }
    
    return scenario_grid


def evaluate_tipping_point(
    hdd: float,
    electricity_price: float,
    gas_price: float,
    envelope_class: str,
    grid_emission_factor: float,
    reference_sqft: float,
    reference_intensity: float
) -> TippingPointResult:
    """
    Evaluate whether heat pump is viable for given conditions.
    
    Returns TippingPointResult with viability assessment.
    """
    from src.retrofit_scenarios import (
        RETROFIT_MEASURES, HEAT_PUMP_OPTIONS,
        EnergyPrices, GridEmissionFactor,
        evaluate_scenario
    )
    
    prices = EnergyPrices(
        electricity_per_kwh=electricity_price,
        natural_gas_per_therm=gas_price
    )
    
    grid_emissions = GridEmissionFactor(
        kg_co2_per_kwh=grid_emission_factor
    )
    
    # Evaluate baseline (no retrofit, gas furnace)
    baseline = evaluate_scenario(
        baseline_intensity=reference_intensity,
        heated_sqft=reference_sqft,
        hdd=hdd,
        retrofit=RETROFIT_MEASURES['none'],
        heat_pump=None,
        prices=prices,
        grid_emissions=grid_emissions
    )
    
    # Evaluate all HP + retrofit combinations
    best_hp_cost = float('inf')
    best_hp_emissions = float('inf')
    best_hp_retrofit = None
    best_hp_type = None
    
    # Filter retrofits by envelope class
    applicable_retrofits = [
        (key, m) for key, m in RETROFIT_MEASURES.items()
        if envelope_class in m.applicable_envelope_classes
    ]
    
    for retrofit_key, retrofit in applicable_retrofits:
        for hp_key, hp in HEAT_PUMP_OPTIONS.items():
            if hp is None:
                continue
            
            result = evaluate_scenario(
                baseline_intensity=reference_intensity,
                heated_sqft=reference_sqft,
                hdd=hdd,
                retrofit=retrofit,
                heat_pump=hp,
                prices=prices,
                grid_emissions=grid_emissions
            )
            
            # Track best by cost
            if result['total_annual_cost'] < best_hp_cost:
                best_hp_cost = result['total_annual_cost']
                best_hp_emissions = result['annual_emissions_kg']
                best_hp_retrofit = retrofit.name
                best_hp_type = hp.name
    
    # Calculate savings
    cost_savings = baseline['total_annual_cost'] - best_hp_cost
    emissions_reduction = baseline['annual_emissions_kg'] - best_hp_emissions
    
    # Determine viability
    is_economically_viable = cost_savings > 0
    is_environmentally_beneficial = emissions_reduction > 0
    
    # Classify viability
    if is_economically_viable and is_environmentally_beneficial:
        viability_class = 'viable'
    elif not is_economically_viable and is_environmentally_beneficial:
        viability_class = 'env_only'
    elif is_economically_viable and not is_environmentally_beneficial:
        viability_class = 'econ_only'
    else:
        viability_class = 'not_viable'
    
    return TippingPointResult(
        hdd=hdd,
        electricity_price=electricity_price,
        gas_price=gas_price,
        envelope_class=envelope_class,
        baseline_cost=baseline['total_annual_cost'],
        baseline_emissions=baseline['annual_emissions_kg'],
        best_hp_cost=best_hp_cost,
        best_hp_emissions=best_hp_emissions,
        best_hp_retrofit=best_hp_retrofit,
        best_hp_type=best_hp_type,
        cost_savings=cost_savings,
        emissions_reduction=emissions_reduction,
        is_economically_viable=is_economically_viable,
        is_environmentally_beneficial=is_environmentally_beneficial,
        viability_class=viability_class
    )


def run_tipping_point_analysis(scenario_grid: Dict) -> pd.DataFrame:
    """
    Run tipping point analysis across the entire scenario grid.
    """
    logger.info("Running tipping point analysis...")
    
    results = []
    
    total_scenarios = (
        len(scenario_grid['hdd_values']) *
        len(scenario_grid['electricity_prices']) *
        len(scenario_grid['gas_prices']) *
        len(scenario_grid['envelope_classes']) *
        len(scenario_grid['grid_emissions'])
    )
    
    logger.info(f"Evaluating {total_scenarios} scenarios...")
    
    count = 0
    for hdd in scenario_grid['hdd_values']:
        for elec_price in scenario_grid['electricity_prices']:
            for gas_price in scenario_grid['gas_prices']:
                for env_class in scenario_grid['envelope_classes']:
                    for grid_name, grid_ef in scenario_grid['grid_emissions'].items():
                        
                        ref_intensity = scenario_grid['reference_intensity'][env_class]
                        
                        try:
                            result = evaluate_tipping_point(
                                hdd=hdd,
                                electricity_price=elec_price,
                                gas_price=gas_price,
                                envelope_class=env_class,
                                grid_emission_factor=grid_ef,
                                reference_sqft=scenario_grid['reference_sqft'],
                                reference_intensity=ref_intensity
                            )
                            
                            result_dict = {
                                'hdd': result.hdd,
                                'electricity_price': result.electricity_price,
                                'gas_price': result.gas_price,
                                'envelope_class': result.envelope_class,
                                'grid_scenario': grid_name,
                                'grid_emission_factor': grid_ef,
                                'baseline_cost': result.baseline_cost,
                                'baseline_emissions': result.baseline_emissions,
                                'best_hp_cost': result.best_hp_cost,
                                'best_hp_emissions': result.best_hp_emissions,
                                'best_hp_retrofit': result.best_hp_retrofit,
                                'best_hp_type': result.best_hp_type,
                                'cost_savings': result.cost_savings,
                                'emissions_reduction': result.emissions_reduction,
                                'is_economically_viable': result.is_economically_viable,
                                'is_environmentally_beneficial': result.is_environmentally_beneficial,
                                'viability_class': result.viability_class,
                            }
                            results.append(result_dict)
                            
                        except Exception as e:
                            logger.warning(f"Error evaluating scenario: {e}")
                        
                        count += 1
                        if count % 500 == 0:
                            logger.info(f"Processed {count}/{total_scenarios} scenarios")
    
    results_df = pd.DataFrame(results)
    logger.info(f"Completed {len(results_df)} scenario evaluations")
    
    return results_df


def generate_figure9_heatmap(
    results_df: pd.DataFrame,
    envelope_class: str = 'medium',
    grid_scenario: str = 'current',
    gas_price: float = 1.20
):
    """
    Generate Figure 9: Tipping point heatmap in HDD × electricity price space.
    """
    logger.info(f"Generating Figure 9: Heatmap for {envelope_class}, {grid_scenario}")
    
    # Filter data
    mask = (
        (results_df['envelope_class'] == envelope_class) &
        (results_df['grid_scenario'] == grid_scenario) &
        (np.abs(results_df['gas_price'] - gas_price) < 0.01)
    )
    subset = results_df[mask]
    
    if len(subset) == 0:
        logger.warning("No data for this combination")
        return
    
    # Create pivot table for cost savings
    pivot = subset.pivot_table(
        index='hdd',
        columns='electricity_price',
        values='cost_savings',
        aggfunc='mean'
    )
    
    # Create viability matrix
    viability_pivot = subset.pivot_table(
        index='hdd',
        columns='electricity_price',
        values='viability_class',
        aggfunc='first'
    )
    
    # Map viability to numeric
    viability_map = {
        'viable': 2,
        'env_only': 1,
        'econ_only': 0.5,
        'not_viable': 0
    }
    viability_numeric = viability_pivot.applymap(lambda x: viability_map.get(x, 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # (a) Cost savings heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(
        pivot.values,
        cmap='RdYlGn',
        aspect='auto',
        origin='lower',
        vmin=-500, vmax=500
    )
    
    ax1.set_xticks(range(len(pivot.columns)))
    ax1.set_xticklabels([f'${x:.2f}' for x in pivot.columns])
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels(pivot.index)
    ax1.set_xlabel('Electricity Price ($/kWh)', fontsize=12)
    ax1.set_ylabel('Heating Degree Days (HDD65)', fontsize=12)
    ax1.set_title(f'(a) Annual Cost Savings (HP vs Gas)\n{envelope_class} envelope', fontsize=14)
    
    plt.colorbar(im1, ax=ax1, label='Annual Savings ($)')
    
    # Add contour at break-even
    contour = ax1.contour(
        pivot.values, levels=[0],
        colors='white', linewidths=2, linestyles='--'
    )
    
    # (b) Viability classification
    ax2 = axes[1]
    
    cmap = LinearSegmentedColormap.from_list(
        'viability',
        ['#d62728', '#ff7f0e', '#9467bd', '#2ca02c'],
        N=4
    )
    
    im2 = ax2.imshow(
        viability_numeric.values,
        cmap=cmap,
        aspect='auto',
        origin='lower',
        vmin=0, vmax=2
    )
    
    ax2.set_xticks(range(len(viability_pivot.columns)))
    ax2.set_xticklabels([f'${x:.2f}' for x in viability_pivot.columns])
    ax2.set_yticks(range(len(viability_pivot.index)))
    ax2.set_yticklabels(viability_pivot.index)
    ax2.set_xlabel('Electricity Price ($/kWh)', fontsize=12)
    ax2.set_ylabel('Heating Degree Days (HDD65)', fontsize=12)
    ax2.set_title(f'(b) HP Viability Classification\n{envelope_class} envelope', fontsize=14)
    
    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ca02c', label='Viable (cost + emissions)'),
        mpatches.Patch(facecolor='#9467bd', label='Emissions benefit only'),
        mpatches.Patch(facecolor='#ff7f0e', label='Cost benefit only'),
        mpatches.Patch(facecolor='#d62728', label='Not viable'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"figure9_heatmap_{envelope_class}.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / f"figure9_heatmap_{envelope_class}.pdf", bbox_inches='tight')
    plt.close()


def generate_figure10_us_map(results_df: pd.DataFrame):
    """
    Generate Figure 10: U.S. map of HP viability by census division.
    
    Note: This creates a simplified representation. For publication-quality
    maps, consider using geopandas with actual shapefiles.
    """
    logger.info("Generating Figure 10: U.S. viability map")
    
    # Census division approximate positions and HDD values
    divisions = {
        'New England': {'hdd': 6500, 'pos': (0.85, 0.85), 'abbr': 'NE'},
        'Middle Atlantic': {'hdd': 5500, 'pos': (0.80, 0.70), 'abbr': 'MA'},
        'East North Central': {'hdd': 6500, 'pos': (0.55, 0.70), 'abbr': 'ENC'},
        'West North Central': {'hdd': 7000, 'pos': (0.40, 0.65), 'abbr': 'WNC'},
        'South Atlantic': {'hdd': 3500, 'pos': (0.75, 0.45), 'abbr': 'SA'},
        'East South Central': {'hdd': 3500, 'pos': (0.55, 0.45), 'abbr': 'ESC'},
        'West South Central': {'hdd': 2500, 'pos': (0.40, 0.35), 'abbr': 'WSC'},
        'Mountain': {'hdd': 5500, 'pos': (0.25, 0.50), 'abbr': 'MTN'},
        'Pacific': {'hdd': 3000, 'pos': (0.10, 0.55), 'abbr': 'PAC'},
    }
    
    # Determine viability for each division under central scenario
    mask = (
        (results_df['envelope_class'] == 'medium') &
        (results_df['grid_scenario'] == 'current') &
        (np.abs(results_df['gas_price'] - 1.20) < 0.01) &
        (np.abs(results_df['electricity_price'] - 0.15) < 0.01)
    )
    
    viability_colors = {
        'viable': '#2ca02c',
        'env_only': '#9467bd',
        'econ_only': '#ff7f0e',
        'not_viable': '#d62728',
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw divisions as circles
    for div_name, div_info in divisions.items():
        hdd = div_info['hdd']
        
        # Find closest HDD in results
        closest_hdd = results_df.loc[mask, 'hdd'].unique()
        if len(closest_hdd) > 0:
            closest_hdd = min(closest_hdd, key=lambda x: abs(x - hdd))
            
            div_mask = mask & (results_df['hdd'] == closest_hdd)
            if div_mask.sum() > 0:
                viability = results_df.loc[div_mask, 'viability_class'].iloc[0]
                color = viability_colors.get(viability, 'gray')
            else:
                color = 'gray'
        else:
            color = 'gray'
        
        # Draw circle
        circle = plt.Circle(
            div_info['pos'], 0.08,
            color=color, alpha=0.7
        )
        ax.add_patch(circle)
        
        # Add label
        ax.text(
            div_info['pos'][0], div_info['pos'][1],
            div_info['abbr'],
            ha='center', va='center',
            fontsize=10, fontweight='bold'
        )
    
    # Set limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ca02c', label='Viable', alpha=0.7),
        mpatches.Patch(facecolor='#9467bd', label='Env. benefit only', alpha=0.7),
        mpatches.Patch(facecolor='#ff7f0e', label='Cost benefit only', alpha=0.7),
        mpatches.Patch(facecolor='#d62728', label='Not viable', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    ax.set_title('Heat Pump Viability by Census Division\n(Medium envelope, current grid, $0.15/kWh, $1.20/therm)',
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure10_us_viability_map.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure10_us_viability_map.pdf", bbox_inches='tight')
    plt.close()


def generate_table7_tipping_points(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 7: Summary of tipping points by division and envelope.
    """
    logger.info("Generating Table 7: Tipping point summary")
    
    # Filter for central gas price scenario
    mask = (
        (results_df['grid_scenario'] == 'current') &
        (np.abs(results_df['gas_price'] - 1.20) < 0.01)
    )
    subset = results_df[mask]
    
    summary = []
    
    for env_class in ['poor', 'medium', 'good']:
        for hdd in sorted(subset['hdd'].unique()):
            env_hdd_mask = (
                (subset['envelope_class'] == env_class) &
                (subset['hdd'] == hdd)
            )
            env_hdd_data = subset[env_hdd_mask]
            
            if len(env_hdd_data) == 0:
                continue
            
            # Find electricity price threshold where HP becomes viable
            viable_prices = env_hdd_data[
                env_hdd_data['is_economically_viable']
            ]['electricity_price']
            
            if len(viable_prices) > 0:
                threshold_price = viable_prices.max()  # Max price still viable
            else:
                threshold_price = None
            
            # Get emissions reduction at that threshold
            if threshold_price:
                at_threshold = env_hdd_data[
                    np.abs(env_hdd_data['electricity_price'] - threshold_price) < 0.01
                ]
                if len(at_threshold) > 0:
                    emissions_red = at_threshold['emissions_reduction'].mean()
                else:
                    emissions_red = None
            else:
                emissions_red = None
            
            summary.append({
                'Envelope Class': env_class,
                'HDD Band': f'{hdd:,}',
                'Viable Price Threshold ($/kWh)': threshold_price,
                'Avg Emissions Reduction (kg)': emissions_red,
                'N Scenarios Evaluated': len(env_hdd_data),
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(TABLES_DIR / "table7_tipping_points.csv", index=False)
    
    # LaTeX version
    summary_df.to_latex(
        TABLES_DIR / "table7_tipping_points.tex",
        index=False,
        float_format="%.2f",
        caption="Heat pump economic tipping points by HDD and envelope class.",
        label="tab:tipping_points",
        na_rep='-'
    )
    
    return summary_df


def generate_figure11_sensitivity(results_df: pd.DataFrame):
    """
    Generate Figure 11: Sensitivity of tipping points to prices and grid decarbonization.
    """
    logger.info("Generating Figure 11: Sensitivity analysis")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) Sensitivity to electricity price
    ax1 = axes[0]
    
    for env_class in ['poor', 'medium', 'good']:
        mask = (
            (results_df['envelope_class'] == env_class) &
            (results_df['grid_scenario'] == 'current') &
            (np.abs(results_df['gas_price'] - 1.20) < 0.01) &
            (results_df['hdd'] == 5500)  # Representative HDD
        )
        subset = results_df[mask].sort_values('electricity_price')
        
        ax1.plot(
            subset['electricity_price'],
            subset['cost_savings'],
            marker='o',
            label=env_class.capitalize()
        )
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_xlabel('Electricity Price ($/kWh)', fontsize=12)
    ax1.set_ylabel('Annual Cost Savings ($)', fontsize=12)
    ax1.set_title('(a) Cost Savings vs. Electricity Price\n(HDD=5500, Gas=$1.20/therm)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # (b) Sensitivity to grid decarbonization
    ax2 = axes[1]
    
    for grid_scenario in ['current', 'moderate_decarb', 'high_decarb']:
        mask = (
            (results_df['envelope_class'] == 'medium') &
            (results_df['grid_scenario'] == grid_scenario) &
            (np.abs(results_df['gas_price'] - 1.20) < 0.01) &
            (np.abs(results_df['electricity_price'] - 0.15) < 0.01)
        )
        subset = results_df[mask].sort_values('hdd')
        
        label_map = {
            'current': 'Current Grid (0.42 kg/kWh)',
            'moderate_decarb': 'Moderate Decarb (0.25 kg/kWh)',
            'high_decarb': 'High Decarb (0.10 kg/kWh)',
        }
        
        ax2.plot(
            subset['hdd'],
            subset['emissions_reduction'],
            marker='s',
            label=label_map.get(grid_scenario, grid_scenario)
        )
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No reduction')
    ax2.set_xlabel('Heating Degree Days (HDD65)', fontsize=12)
    ax2.set_ylabel('Annual Emissions Reduction (kg CO₂)', fontsize=12)
    ax2.set_title('(b) Emissions Reduction vs. Climate\n(Medium envelope, $0.15/kWh)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure11_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure11_sensitivity.pdf", bbox_inches='tight')
    plt.close()


def run_tipping_point_pipeline() -> Dict:
    """Main function to run the tipping point analysis pipeline."""
    logger.info("=" * 60)
    logger.info("Tipping Point Analysis and Mapping")
    logger.info("=" * 60)
    
    # Define scenario grid
    scenario_grid = define_scenario_grid()
    
    # Run analysis
    results_df = run_tipping_point_analysis(scenario_grid)
    
    # Save results
    results_df.to_csv(RESULTS_DIR / "tipping_point_results.csv", index=False)
    
    # Generate Table 7
    table7 = generate_table7_tipping_points(results_df)
    
    # Generate Figure 9 - Heatmaps for each envelope class
    for env_class in ['poor', 'medium', 'good']:
        generate_figure9_heatmap(results_df, envelope_class=env_class)
    
    # Generate Figure 10 - U.S. map
    generate_figure10_us_map(results_df)
    
    # Generate Figure 11 - Sensitivity analysis
    generate_figure11_sensitivity(results_df)
    
    logger.info("=" * 60)
    logger.info("Tipping point analysis complete!")
    logger.info("=" * 60)
    
    return {
        'scenario_grid': scenario_grid,
        'results': results_df,
        'table7': table7,
    }


if __name__ == "__main__":
    results = run_tipping_point_pipeline()
    
    print("\n" + "=" * 60)
    print("TIPPING POINT ANALYSIS SUMMARY")
    print("=" * 60)
    print("\nTable 7: Tipping Points Summary")
    print(results['table7'].to_string())
