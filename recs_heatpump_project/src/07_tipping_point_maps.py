"""
Step 7: Tipping Point Maps
===========================

Goal: Identify conditions where heat pump retrofits become economically and 
environmentally preferable to natural gas heating.

This script:
1. Builds a scenario grid (electricity price, HDD bands, envelope classes)
2. Calculates cost and emissions for baseline vs HP scenarios across grid
3. Identifies tipping points (where HP becomes preferable)
4. Visualizes results as maps and contour plots
5. Generates summary tables of tipping point conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import functions from previous steps
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("retrofit_scenarios", 
    Path(__file__).parent / "05_retrofit_scenarios.py")
retrofit_scenarios = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrofit_scenarios)

# Import needed functions
define_retrofit_measures = retrofit_scenarios.define_retrofit_measures
define_heat_pump_options = retrofit_scenarios.define_heat_pump_options
calculate_operational_costs = retrofit_scenarios.calculate_operational_costs
calculate_emissions = retrofit_scenarios.calculate_emissions
calculate_retrofit_intensity_reduction = retrofit_scenarios.calculate_retrofit_intensity_reduction

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CLEANED_DATA = OUTPUT_DIR / "recs2020_gas_heated_cleaned.csv"
SCENARIO_DATA = OUTPUT_DIR / "retrofit_scenarios.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
DISCOUNT_RATE = 0.03
ANALYSIS_PERIOD = 20

# Scenario grid parameters
ELECTRICITY_PRICE_RANGE = np.linspace(0.08, 0.20, 25)  # $/kWh
HDD_BANDS = [
    (0, 2000, "Very Mild"),
    (2000, 4000, "Mild"),
    (4000, 6000, "Moderate"),
    (6000, 8000, "Cold"),
    (8000, 10000, "Very Cold"),
    (10000, 15000, "Extreme")
]
ENVELOPE_CLASSES = ["poor", "medium", "good"]

# Fuel prices (baseline)
GAS_PRICE_BASELINE = 1.20  # $/therm
ELECTRICITY_PRICE_BASELINE = 0.13  # $/kWh

# Emission factors
GAS_EMISSION_FACTOR = 5.31  # kg CO2/therm
GRID_EMISSION_FACTOR_BASELINE = 0.4  # kg CO2/kWh (national average, will vary by region)


def load_archetype_data():
    """Load cleaned data and create representative archetypes."""
    df = pd.read_csv(CLEANED_DATA)
    
    # Create archetypes by envelope class and HDD band
    archetypes = []
    
    for env_class in ENVELOPE_CLASSES:
        env_data = df[df['envelope_class'] == env_class].copy()
        if len(env_data) == 0:
            continue
            
        # Calculate median characteristics for this envelope class
        archetype = {
            'envelope_class': env_class,
            'heated_area_sqft': env_data['heated_area_sqft'].median(),
            'thermal_intensity': env_data['thermal_intensity'].median(),
            'hdd65': env_data['hdd65'].median(),
            'division': env_data['division'].mode()[0] if len(env_data['division'].mode()) > 0 else 'Middle Atlantic',
            'housing_type': env_data['housing_type'].mode()[0] if len(env_data['housing_type'].mode()) > 0 else 'Single-family detached'
        }
        archetypes.append(archetype)
    
    return pd.DataFrame(archetypes)


def calculate_scenario_costs_emissions(archetype, electricity_price, hdd_value, 
                                      retrofit_measures=None, hp_option=None):
    """
    Calculate annualized total cost and annual CO2 emissions for a scenario.
    
    Parameters
    ----------
    archetype : dict
        Representative dwelling characteristics
    electricity_price : float
        Electricity price in $/kWh
    hdd_value : float
        Heating degree days (HDD65)
    retrofit_measures : list, optional
        List of RetrofitMeasure objects to apply
    hp_option : HeatPumpOption, optional
        Heat pump option (if None, uses natural gas baseline)
    
    Returns
    -------
    dict
        Dictionary with 'total_cost', 'operational_cost', 'capital_cost', 
        'co2_emissions', 'thermal_intensity'
    """
    # Get baseline thermal intensity
    thermal_intensity = archetype['thermal_intensity']
    
    # Apply retrofit reductions if specified
    if retrofit_measures:
        total_reduction = 0
        total_capital_cost = 0
        
        for measure in retrofit_measures:
            reduction = measure.load_reduction_pct
            total_reduction += reduction * (1 - total_reduction)  # Cumulative reduction
            
            # Estimate retrofit area (simplified - using heated area as proxy)
            area = archetype['heated_area_sqft']
            if 'attic' in measure.name.lower():
                area = area * 0.8  # Attic area approximation
            elif 'wall' in measure.name.lower():
                area = archetype['heated_area_sqft'] ** 0.5 * 8 * 2  # Wall area approximation
            
            annualized_cost = measure.annualized_cost(area=area)
            total_capital_cost += annualized_cost
        
        thermal_intensity *= (1 - total_reduction)
    else:
        total_capital_cost = 0
    
    # Calculate heating load
    heating_load_btu = thermal_intensity * archetype['heated_area_sqft'] * hdd_value
    
    # Determine HP COP based on HDD (proxy for climate severity)
    if hp_option:
        if hdd_value > 6000:
            cop = hp_option.cop_cold
        else:
            cop = hp_option.cop_mild
        
        # Convert heating load to electricity consumption
        # 1 BTU = 0.000293 kWh
        electricity_kwh = (heating_load_btu * 0.000293) / cop
        
        operational_cost = electricity_kwh * electricity_price * 100  # Annual (simplified)
        
        # HP capital cost (annualized)
        hp_capital_cost = hp_option.annualized_cost()
        total_capital_cost += hp_capital_cost
        
        # CO2 emissions (electricity)
        # Use regional grid factor if available, else baseline
        grid_emission_factor = GRID_EMISSION_FACTOR_BASELINE
        co2_emissions = electricity_kwh * grid_emission_factor * 100  # Annual
        
    else:
        # Natural gas baseline
        # 1 therm = 100,000 BTU
        gas_therms = heating_load_btu / 100000
        
        operational_cost = gas_therms * GAS_PRICE_BASELINE * 100  # Annual (simplified)
        co2_emissions = gas_therms * GAS_EMISSION_FACTOR * 100  # Annual
    
    total_cost = operational_cost + total_capital_cost
    
    return {
        'total_cost': total_cost,
        'operational_cost': operational_cost,
        'capital_cost': total_capital_cost,
        'co2_emissions': co2_emissions,
        'thermal_intensity': thermal_intensity
    }


def build_scenario_grid():
    """
    Build a comprehensive scenario grid and evaluate all combinations.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: electricity_price, hdd_band, hdd_midpoint, 
        envelope_class, scenario_type, total_cost, co2_emissions, 
        cost_premium, emissions_reduction
    """
    archetypes = load_archetype_data()
    retrofit_measures = define_retrofit_measures()
    hp_options = define_heat_pump_options()
    
    # Use standard HP for main analysis
    standard_hp = hp_options[0] if len(hp_options) > 0 else None
    
    scenario_results = []
    
    for elec_price in ELECTRICITY_PRICE_RANGE:
        for hdd_min, hdd_max, hdd_label in HDD_BANDS:
            hdd_midpoint = (hdd_min + hdd_max) / 2
            
            for _, archetype in archetypes.iterrows():
                env_class = archetype['envelope_class']
                
                # Baseline (natural gas, no retrofit)
                baseline = calculate_scenario_costs_emissions(
                    archetype, elec_price, hdd_midpoint
                )
                
                # HP only (no retrofit)
                hp_only = calculate_scenario_costs_emissions(
                    archetype, elec_price, hdd_midpoint,
                    hp_option=standard_hp
                )
                
                # Retrofit + HP
                retrofit_hp = calculate_scenario_costs_emissions(
                    archetype, elec_price, hdd_midpoint,
                    retrofit_measures=retrofit_measures[:2],  # Attic + wall insulation
                    hp_option=standard_hp
                )
                
                # Calculate metrics
                cost_premium_hp = hp_only['total_cost'] - baseline['total_cost']
                cost_premium_retrofit_hp = retrofit_hp['total_cost'] - baseline['total_cost']
                
                emissions_reduction_hp = baseline['co2_emissions'] - hp_only['co2_emissions']
                emissions_reduction_retrofit_hp = baseline['co2_emissions'] - retrofit_hp['co2_emissions']
                
                # Determine if HP is preferable (cost-effective AND lower emissions)
                hp_preferable = (cost_premium_hp <= 0) and (emissions_reduction_hp > 0)
                retrofit_hp_preferable = (cost_premium_retrofit_hp <= 0) and (emissions_reduction_retrofit_hp > 0)
                
                scenario_results.append({
                    'electricity_price': elec_price,
                    'hdd_band': hdd_label,
                    'hdd_min': hdd_min,
                    'hdd_max': hdd_max,
                    'hdd_midpoint': hdd_midpoint,
                    'envelope_class': env_class,
                    'baseline_cost': baseline['total_cost'],
                    'baseline_emissions': baseline['co2_emissions'],
                    'hp_only_cost': hp_only['total_cost'],
                    'hp_only_emissions': hp_only['co2_emissions'],
                    'retrofit_hp_cost': retrofit_hp['total_cost'],
                    'retrofit_hp_emissions': retrofit_hp['co2_emissions'],
                    'cost_premium_hp': cost_premium_hp,
                    'cost_premium_retrofit_hp': cost_premium_retrofit_hp,
                    'emissions_reduction_hp': emissions_reduction_hp,
                    'emissions_reduction_retrofit_hp': emissions_reduction_retrofit_hp,
                    'hp_preferable': hp_preferable,
                    'retrofit_hp_preferable': retrofit_hp_preferable
                })
    
    return pd.DataFrame(scenario_results)


def identify_tipping_points(scenario_grid):
    """
    Identify tipping point conditions where HP becomes preferable.
    
    Parameters
    ----------
    scenario_grid : pd.DataFrame
        Output from build_scenario_grid()
    
    Returns
    -------
    pd.DataFrame
        Summary of tipping points by envelope class and HDD band
    """
    tipping_points = []
    
    for env_class in ENVELOPE_CLASSES:
        for hdd_label in [band[2] for band in HDD_BANDS]:
            subset = scenario_grid[
                (scenario_grid['envelope_class'] == env_class) &
                (scenario_grid['hdd_band'] == hdd_label)
            ]
            
            if len(subset) == 0:
                continue
            
            # Find minimum electricity price where HP becomes preferable
            hp_preferable_subset = subset[subset['hp_preferable']]
            retrofit_hp_preferable_subset = subset[subset['retrofit_hp_preferable']]
            
            if len(hp_preferable_subset) > 0:
                min_elec_price_hp = hp_preferable_subset['electricity_price'].min()
            else:
                min_elec_price_hp = None
            
            if len(retrofit_hp_preferable_subset) > 0:
                min_elec_price_retrofit_hp = retrofit_hp_preferable_subset['electricity_price'].min()
            else:
                min_elec_price_retrofit_hp = None
            
            # Calculate average cost premium and emissions reduction at tipping point
            if min_elec_price_hp is not None:
                tp_data = subset[subset['electricity_price'] == min_elec_price_hp].iloc[0]
                avg_cost_premium_hp = tp_data['cost_premium_hp']
                avg_emissions_reduction_hp = tp_data['emissions_reduction_hp']
            else:
                avg_cost_premium_hp = None
                avg_emissions_reduction_hp = None
            
            if min_elec_price_retrofit_hp is not None:
                tp_data = subset[subset['electricity_price'] == min_elec_price_retrofit_hp].iloc[0]
                avg_cost_premium_retrofit_hp = tp_data['cost_premium_retrofit_hp']
                avg_emissions_reduction_retrofit_hp = tp_data['emissions_reduction_retrofit_hp']
            else:
                avg_cost_premium_retrofit_hp = None
                avg_emissions_reduction_retrofit_hp = None
            
            tipping_points.append({
                'envelope_class': env_class,
                'hdd_band': hdd_label,
                'min_elec_price_hp_only': min_elec_price_hp,
                'min_elec_price_retrofit_hp': min_elec_price_retrofit_hp,
                'cost_premium_hp_at_tp': avg_cost_premium_hp,
                'emissions_reduction_hp_at_tp': avg_emissions_reduction_hp,
                'cost_premium_retrofit_hp_at_tp': avg_cost_premium_retrofit_hp,
                'emissions_reduction_retrofit_hp_at_tp': avg_emissions_reduction_retrofit_hp
            })
    
    return pd.DataFrame(tipping_points)


def plot_tipping_point_heatmap(scenario_grid, tipping_points):
    """
    Create heatmap showing tipping point conditions.
    
    Parameters
    ----------
    scenario_grid : pd.DataFrame
        Full scenario grid results
    tipping_points : pd.DataFrame
        Tipping point summary
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, env_class in enumerate(ENVELOPE_CLASSES):
        ax = axes[idx]
        
        # Prepare data for heatmap
        heatmap_data = []
        hdd_labels = [band[2] for band in HDD_BANDS]
        
        for hdd_label in hdd_labels:
            tp_row = tipping_points[
                (tipping_points['envelope_class'] == env_class) &
                (tipping_points['hdd_band'] == hdd_label)
            ]
            
            if len(tp_row) > 0 and pd.notna(tp_row.iloc[0]['min_elec_price_hp_only']):
                min_price = tp_row.iloc[0]['min_elec_price_hp_only']
            else:
                min_price = np.nan
            
            heatmap_data.append(min_price)
        
        # Create heatmap
        if not all(pd.isna(heatmap_data)):
            im = ax.imshow([heatmap_data], aspect='auto', cmap='RdYlGn_r', 
                          vmin=ELECTRICITY_PRICE_RANGE.min(), 
                          vmax=ELECTRICITY_PRICE_RANGE.max())
            ax.set_yticks([0])
            ax.set_yticklabels([env_class.title()])
            ax.set_xticks(range(len(hdd_labels)))
            ax.set_xticklabels(hdd_labels, rotation=45, ha='right')
            ax.set_title(f'Tipping Point Electricity Price\n({env_class.title()} Envelope)')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Electricity Price ($/kWh)')
            
            # Add text annotations
            for i, val in enumerate(heatmap_data):
                if pd.notna(val):
                    ax.text(i, 0, f'{val:.3f}', ha='center', va='center', 
                           color='white' if val > 0.14 else 'black', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No tipping point\nfound', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{env_class.title()} Envelope')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure9_tipping_point_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cost_emissions_tradeoff(scenario_grid):
    """
    Plot cost vs emissions tradeoff across scenarios.
    
    Parameters
    ----------
    scenario_grid : pd.DataFrame
        Full scenario grid results
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, env_class in enumerate(ENVELOPE_CLASSES):
        ax = axes[idx]
        
        subset = scenario_grid[scenario_grid['envelope_class'] == env_class]
        
        # Plot baseline
        ax.scatter(subset['baseline_emissions'], subset['baseline_cost'], 
                  alpha=0.3, label='Baseline (Gas)', s=20, color='red')
        
        # Plot HP scenarios
        hp_preferable = subset[subset['hp_preferable']]
        hp_not_preferable = subset[~subset['hp_preferable']]
        
        if len(hp_preferable) > 0:
            ax.scatter(hp_preferable['hp_only_emissions'], hp_preferable['hp_only_cost'],
                      alpha=0.5, label='HP Preferable', s=30, color='green', marker='^')
        
        if len(hp_not_preferable) > 0:
            ax.scatter(hp_not_preferable['hp_only_emissions'], hp_not_preferable['hp_only_cost'],
                      alpha=0.3, label='HP Not Preferable', s=20, color='orange', marker='^')
        
        ax.set_xlabel('Annual CO₂ Emissions (kg)')
        ax.set_ylabel('Annualized Total Cost ($)')
        ax.set_title(f'Cost vs Emissions: {env_class.title()} Envelope')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure10_cost_emissions_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_electricity_price_sensitivity(scenario_grid):
    """
    Plot how HP preference changes with electricity price.
    
    Parameters
    ----------
    scenario_grid : pd.DataFrame
        Full scenario grid results
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, env_class in enumerate(ENVELOPE_CLASSES):
        ax = axes[idx]
        
        # Aggregate by electricity price and HDD band
        for hdd_label in ['Mild', 'Moderate', 'Cold', 'Very Cold']:
            subset = scenario_grid[
                (scenario_grid['envelope_class'] == env_class) &
                (scenario_grid['hdd_band'] == hdd_label)
            ]
            
            if len(subset) == 0:
                continue
            
            # Calculate percentage of scenarios where HP is preferable
            hp_preferable_pct = []
            elec_prices = []
            
            for elec_price in ELECTRICITY_PRICE_RANGE:
                price_subset = subset[subset['electricity_price'] == elec_price]
                if len(price_subset) > 0:
                    pct = price_subset['hp_preferable'].mean() * 100
                    hp_preferable_pct.append(pct)
                    elec_prices.append(elec_price)
            
            if len(hp_preferable_pct) > 0:
                ax.plot(elec_prices, hp_preferable_pct, marker='o', label=hdd_label, linewidth=2)
        
        ax.axvline(ELECTRICITY_PRICE_BASELINE, color='red', linestyle='--', 
                  label='Baseline Price', linewidth=2)
        ax.set_xlabel('Electricity Price ($/kWh)')
        ax.set_ylabel('% Scenarios Where HP is Preferable')
        ax.set_title(f'Electricity Price Sensitivity: {env_class.title()} Envelope')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure11_electricity_price_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    print("Step 7: Building Tipping Point Maps...")
    print("=" * 60)
    
    # Check if cleaned data exists
    if not CLEANED_DATA.exists():
        print(f"ERROR: Cleaned data not found at {CLEANED_DATA}")
        print("Please run Step 1 (01_data_prep.py) first.")
        return
    
    # Build scenario grid
    print("\n1. Building scenario grid...")
    scenario_grid = build_scenario_grid()
    print(f"   Generated {len(scenario_grid)} scenarios")
    
    # Identify tipping points
    print("\n2. Identifying tipping points...")
    tipping_points = identify_tipping_points(scenario_grid)
    print(f"   Found tipping points for {len(tipping_points)} combinations")
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    plot_tipping_point_heatmap(scenario_grid, tipping_points)
    print("   ✓ Created figure9_tipping_point_heatmap.png")
    
    plot_cost_emissions_tradeoff(scenario_grid)
    print("   ✓ Created figure10_cost_emissions_tradeoff.png")
    
    plot_electricity_price_sensitivity(scenario_grid)
    print("   ✓ Created figure11_electricity_price_sensitivity.png")
    
    # Save results
    print("\n4. Saving results...")
    scenario_grid.to_csv(OUTPUT_DIR / 'tipping_point_scenario_grid.csv', index=False)
    print(f"   ✓ Saved tipping_point_scenario_grid.csv ({len(scenario_grid)} rows)")
    
    tipping_points.to_csv(TABLES_DIR / 'table7_tipping_points.csv', index=False)
    print(f"   ✓ Saved table7_tipping_points.csv")
    
    # Generate summary statistics
    summary_stats = {
        'total_scenarios': len(scenario_grid),
        'hp_preferable_count': scenario_grid['hp_preferable'].sum(),
        'hp_preferable_pct': scenario_grid['hp_preferable'].mean() * 100,
        'retrofit_hp_preferable_count': scenario_grid['retrofit_hp_preferable'].sum(),
        'retrofit_hp_preferable_pct': scenario_grid['retrofit_hp_preferable'].mean() * 100,
        'avg_cost_premium_hp': scenario_grid['cost_premium_hp'].mean(),
        'avg_emissions_reduction_hp': scenario_grid['emissions_reduction_hp'].mean(),
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(TABLES_DIR / 'table8_tipping_point_summary.csv', index=False)
    print(f"   ✓ Saved table8_tipping_point_summary.csv")
    
    print("\n" + "=" * 60)
    print("Step 7 Complete!")
    print(f"\nResults saved to:")
    print(f"  - {OUTPUT_DIR / 'tipping_point_scenario_grid.csv'}")
    print(f"  - {TABLES_DIR / 'table7_tipping_points.csv'}")
    print(f"  - {TABLES_DIR / 'table8_tipping_point_summary.csv'}")
    print(f"  - {FIGURES_DIR / 'figure9_tipping_point_heatmap.png'}")
    print(f"  - {FIGURES_DIR / 'figure10_cost_emissions_tradeoff.png'}")
    print(f"  - {FIGURES_DIR / 'figure11_electricity_price_sensitivity.png'}")


if __name__ == "__main__":
    main()
