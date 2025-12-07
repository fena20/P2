"""
Step 5: Retrofit & Heat Pump Scenarios
=======================================

Goal: Define retrofit measures and heat pump options, calculate costs and emissions.

This script:
1. Defines retrofit measures (attic insulation, wall insulation, windows, etc.)
2. Defines heat pump options (standard vs cold-climate)
3. Calculates intensity adjustments based on retrofit measures
4. Computes capital costs (annualized)
5. Computes operational costs using fuel prices
6. Computes CO₂ emissions using regional grid + gas emission factors
7. Prepares scenario data for optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CLEANED_DATA = OUTPUT_DIR / "recs2020_gas_heated_cleaned.csv"

# Configuration
DISCOUNT_RATE = 0.03  # 3% discount rate for annualization
ANALYSIS_PERIOD = 20  # years


class RetrofitMeasure:
    """Represents a retrofit measure with cost and performance characteristics."""
    
    def __init__(self, name, unit_cost, lifetime_years, load_reduction_pct, 
                 description=""):
        """
        Parameters
        ----------
        name : str
            Name of retrofit measure
        unit_cost : float
            Cost per unit (e.g., per sqft)
        lifetime_years : int
            Expected lifetime in years
        load_reduction_pct : float
            Percentage reduction in heating load (0-1)
        description : str
            Description of the measure
        """
        self.name = name
        self.unit_cost = unit_cost
        self.lifetime_years = lifetime_years
        self.load_reduction_pct = load_reduction_pct
        self.description = description
    
    def annualized_cost(self, area=None, total_cost=None):
        """
        Calculate annualized cost using capital recovery factor.
        
        Parameters
        ----------
        area : float, optional
            Area to retrofit (for unit-based costing)
        total_cost : float, optional
            Total cost (if area not provided)
            
        Returns
        -------
        float
            Annualized cost
        """
        if total_cost is None:
            if area is None:
                raise ValueError("Must provide either area or total_cost")
            total_cost = self.unit_cost * area
        
        # Capital recovery factor
        crf = (DISCOUNT_RATE * (1 + DISCOUNT_RATE)**self.lifetime_years) / \
              ((1 + DISCOUNT_RATE)**self.lifetime_years - 1)
        
        return total_cost * crf


class HeatPumpOption:
    """Represents a heat pump option with performance and cost characteristics."""
    
    def __init__(self, name, capacity_btu_per_h, cop_cold, cop_mild, 
                 installed_cost, lifetime_years, description=""):
        """
        Parameters
        ----------
        name : str
            Name of heat pump type
        capacity_btu_per_h : float
            Heating capacity in BTU/hour
        cop_cold : float
            Coefficient of Performance in cold conditions (< 32°F)
        cop_mild : float
            Coefficient of Performance in mild conditions (> 32°F)
        installed_cost : float
            Installed cost (total)
        lifetime_years : int
            Expected lifetime
        description : str
            Description
        """
        self.name = name
        self.capacity_btu_per_h = capacity_btu_per_h
        self.cop_cold = cop_cold
        self.cop_mild = cop_mild
        self.installed_cost = installed_cost
        self.lifetime_years = lifetime_years
        self.description = description
    
    def annualized_cost(self):
        """Calculate annualized capital cost."""
        crf = (DISCOUNT_RATE * (1 + DISCOUNT_RATE)**self.lifetime_years) / \
              ((1 + DISCOUNT_RATE)**self.lifetime_years - 1)
        return self.installed_cost * crf
    
    def seasonal_cop(self, hdd65, cold_threshold_hdd=3000):
        """
        Estimate seasonal COP based on climate.
        
        Parameters
        ----------
        hdd65 : float
            Heating degree days
        cold_threshold_hdd : float
            HDD threshold for "cold" climate
            
        Returns
        -------
        float
            Weighted average COP
        """
        # Simple weighting: more HDD = more cold weather operation
        cold_fraction = min(1.0, hdd65 / cold_threshold_hdd)
        return cold_fraction * self.cop_cold + (1 - cold_fraction) * self.cop_mild


def define_retrofit_measures():
    """
    Define standard retrofit measures based on literature.
    
    Returns
    -------
    dict
        Dictionary of RetrofitMeasure objects
    """
    measures = {
        'attic_insulation': RetrofitMeasure(
            name='Attic Insulation Upgrade',
            unit_cost=1.50,  # $/sqft
            lifetime_years=30,
            load_reduction_pct=0.15,  # 15% reduction
            description='Add insulation to attic to R-49'
        ),
        'wall_insulation': RetrofitMeasure(
            name='Wall Insulation Upgrade',
            unit_cost=3.00,  # $/sqft
            lifetime_years=30,
            load_reduction_pct=0.20,  # 20% reduction
            description='Add wall insulation'
        ),
        'window_replacement': RetrofitMeasure(
            name='Window Replacement',
            unit_cost=25.00,  # $/sqft
            lifetime_years=25,
            load_reduction_pct=0.10,  # 10% reduction
            description='Replace single-pane with double-pane windows'
        ),
        'air_sealing': RetrofitMeasure(
            name='Air Sealing',
            unit_cost=0.50,  # $/sqft
            lifetime_years=20,
            load_reduction_pct=0.12,  # 12% reduction
            description='Seal air leaks in building envelope'
        ),
    }
    
    return measures


def define_heat_pump_options():
    """
    Define heat pump options.
    
    Returns
    -------
    dict
        Dictionary of HeatPumpOption objects
    """
    options = {
        'standard_hp': HeatPumpOption(
            name='Standard Air-Source Heat Pump',
            capacity_btu_per_h=48000,  # ~4 tons
            cop_cold=2.0,  # COP at cold temps
            cop_mild=3.5,  # COP at mild temps
            installed_cost=8000,  # $ installed
            lifetime_years=15,
            description='Standard ASHP for moderate climates'
        ),
        'cold_climate_hp': HeatPumpOption(
            name='Cold-Climate Heat Pump',
            capacity_btu_per_h=48000,
            cop_cold=2.5,  # Better cold-weather performance
            cop_mild=4.0,
            installed_cost=12000,  # Higher cost
            lifetime_years=15,
            description='Cold-climate ASHP with better low-temp performance'
        ),
    }
    
    return options


def calculate_retrofit_intensity_reduction(baseline_intensity, retrofit_combination, measures):
    """
    Calculate new thermal intensity after retrofits.
    
    Parameters
    ----------
    baseline_intensity : float
        Original thermal intensity
    retrofit_combination : list
        List of retrofit measure names to apply
    measures : dict
        Dictionary of RetrofitMeasure objects
        
    Returns
    -------
    float
        New thermal intensity
    """
    total_reduction = 1.0
    
    for measure_name in retrofit_combination:
        if measure_name in measures:
            # Apply reduction multiplicatively (conservative)
            total_reduction *= (1 - measures[measure_name].load_reduction_pct)
    
    # Cap maximum reduction at 50% (realistic limit)
    total_reduction = max(0.5, total_reduction)
    
    return baseline_intensity * total_reduction


def calculate_operational_costs(heating_load_btu, fuel_type='gas', 
                                fuel_price=None, cop=None, hdd65=None):
    """
    Calculate annual operational costs.
    
    Parameters
    ----------
    heating_load_btu : float
        Annual heating load in BTU
    fuel_type : str
        'gas' or 'electricity'
    fuel_price : float
        Price per unit (therm for gas, kWh for electricity)
    cop : float, optional
        COP for heat pump (if fuel_type='electricity')
    hdd65 : float, optional
        HDD65 for COP estimation
        
    Returns
    -------
    float
        Annual operational cost
    """
    if fuel_type == 'gas':
        # Natural gas: 1 therm = 100,000 BTU
        therms = heating_load_btu / 100000
        if fuel_price is None:
            fuel_price = 1.20  # $/therm (default)
        return therms * fuel_price
    
    elif fuel_type == 'electricity':
        # Electricity: need COP to convert BTU to kWh
        if cop is None:
            if hdd65 is not None:
                # Estimate COP from climate
                hp_options = define_heat_pump_options()
                cop = hp_options['standard_hp'].seasonal_cop(hdd65)
            else:
                cop = 3.0  # Default
        
        # 1 kWh = 3412 BTU
        # With COP, input energy = output / COP
        input_kwh = (heating_load_btu / 3412) / cop
        
        if fuel_price is None:
            fuel_price = 0.13  # $/kWh (default)
        
        return input_kwh * fuel_price
    
    else:
        raise ValueError(f"Unknown fuel type: {fuel_type}")


def calculate_emissions(heating_load_btu, fuel_type='gas', 
                        emission_factor_gas=None, emission_factor_grid=None,
                        cop=None, hdd65=None):
    """
    Calculate annual CO₂ emissions.
    
    Parameters
    ----------
    heating_load_btu : float
        Annual heating load in BTU
    fuel_type : str
        'gas' or 'electricity'
    emission_factor_gas : float
        CO₂ emissions per therm of gas (kg CO₂/therm)
    emission_factor_grid : float
        CO₂ emissions per kWh of electricity (kg CO₂/kWh)
    cop : float, optional
        COP for heat pump
    hdd65 : float, optional
        HDD65 for COP estimation
        
    Returns
    -------
    float
        Annual CO₂ emissions (kg)
    """
    if fuel_type == 'gas':
        therms = heating_load_btu / 100000
        if emission_factor_gas is None:
            emission_factor_gas = 5.3  # kg CO₂/therm (default)
        return therms * emission_factor_gas
    
    elif fuel_type == 'electricity':
        if cop is None:
            if hdd65 is not None:
                hp_options = define_heat_pump_options()
                cop = hp_options['standard_hp'].seasonal_cop(hdd65)
            else:
                cop = 3.0
        
        input_kwh = (heating_load_btu / 3412) / cop
        
        if emission_factor_grid is None:
            emission_factor_grid = 0.4  # kg CO₂/kWh (default, varies by region)
        
        return input_kwh * emission_factor_grid
    
    else:
        raise ValueError(f"Unknown fuel type: {fuel_type}")


def create_scenario_dataframe(df, measures, hp_options):
    """
    Create a dataframe with scenario calculations for each dwelling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with thermal intensity
    measures : dict
        Retrofit measures
    hp_options : dict
        Heat pump options
        
    Returns
    -------
    pd.DataFrame
        Scenario dataframe with costs and emissions
    """
    print("Creating scenario calculations...")
    
    scenario_data = []
    
    # Get required columns
    intensity_col = 'thermal_intensity'
    area_col = 'TOTSQFT_EN'
    hdd_col = 'HDD65'
    
    if intensity_col not in df.columns:
        raise ValueError(f"Missing column: {intensity_col}")
    
    # Sample for testing (remove in production)
    n_sample = min(1000, len(df))
    df_sample = df.sample(n=n_sample, random_state=42) if len(df) > n_sample else df
    
    for idx, row in df_sample.iterrows():
        baseline_intensity = row[intensity_col]
        area = row[area_col] if area_col in row else 2000  # Default
        hdd65 = row[hdd_col] if hdd_col in row else 4000  # Default
        
        if pd.isna(baseline_intensity):
            continue
        
        # Baseline: gas heating, no retrofits
        baseline_load = baseline_intensity * area * hdd65
        
        baseline_opex = calculate_operational_costs(baseline_load, 'gas')
        baseline_emissions = calculate_emissions(baseline_load, 'gas')
        
        scenario_data.append({
            'dwelling_id': idx,
            'scenario': 'baseline_gas',
            'retrofits': 'none',
            'heat_pump': 'none',
            'intensity': baseline_intensity,
            'annual_load_btu': baseline_load,
            'capex_annualized': 0.0,
            'opex_annual': baseline_opex,
            'total_cost_annual': baseline_opex,
            'emissions_kg_co2': baseline_emissions,
            'area': area,
            'hdd65': hdd65,
        })
        
        # Scenarios with retrofits only (still gas)
        retrofit_combinations = [
            ['attic_insulation'],
            ['wall_insulation'],
            ['window_replacement'],
            ['air_sealing'],
            ['attic_insulation', 'air_sealing'],
            ['attic_insulation', 'wall_insulation', 'air_sealing'],
        ]
        
        for retrofit_combo in retrofit_combinations:
            new_intensity = calculate_retrofit_intensity_reduction(
                baseline_intensity, retrofit_combo, measures
            )
            new_load = new_intensity * area * hdd65
            
            # Calculate retrofit costs
            retrofit_capex = 0.0
            for measure_name in retrofit_combo:
                if measure_name in measures:
                    measure = measures[measure_name]
                    retrofit_capex += measure.annualized_cost(area=area)
            
            opex = calculate_operational_costs(new_load, 'gas')
            emissions = calculate_emissions(new_load, 'gas')
            
            scenario_data.append({
                'dwelling_id': idx,
                'scenario': f'retrofit_{"+".join(retrofit_combo)}',
                'retrofits': '+'.join(retrofit_combo),
                'heat_pump': 'none',
                'intensity': new_intensity,
                'annual_load_btu': new_load,
                'capex_annualized': retrofit_capex,
                'opex_annual': opex,
                'total_cost_annual': retrofit_capex + opex,
                'emissions_kg_co2': emissions,
                'area': area,
                'hdd65': hdd65,
            })
        
        # Scenarios with heat pumps (with/without retrofits)
        for hp_name, hp_option in hp_options.items():
            # HP only (no retrofits)
            hp_capex = hp_option.annualized_cost()
            cop = hp_option.seasonal_cop(hdd65)
            
            opex_hp = calculate_operational_costs(baseline_load, 'electricity', cop=cop)
            emissions_hp = calculate_emissions(baseline_load, 'electricity', cop=cop)
            
            scenario_data.append({
                'dwelling_id': idx,
                'scenario': f'{hp_name}_only',
                'retrofits': 'none',
                'heat_pump': hp_name,
                'intensity': baseline_intensity,
                'annual_load_btu': baseline_load,
                'capex_annualized': hp_capex,
                'opex_annual': opex_hp,
                'total_cost_annual': hp_capex + opex_hp,
                'emissions_kg_co2': emissions_hp,
                'area': area,
                'hdd65': hdd65,
            })
            
            # HP + retrofits (best combination)
            best_retrofit = ['attic_insulation', 'wall_insulation', 'air_sealing']
            new_intensity = calculate_retrofit_intensity_reduction(
                baseline_intensity, best_retrofit, measures
            )
            new_load = new_intensity * area * hdd65
            
            retrofit_capex = sum(
                measures[m].annualized_cost(area=area) 
                for m in best_retrofit if m in measures
            )
            
            opex_hp_retrofit = calculate_operational_costs(new_load, 'electricity', cop=cop)
            emissions_hp_retrofit = calculate_emissions(new_load, 'electricity', cop=cop)
            
            scenario_data.append({
                'dwelling_id': idx,
                'scenario': f'{hp_name}_with_retrofits',
                'retrofits': '+'.join(best_retrofit),
                'heat_pump': hp_name,
                'intensity': new_intensity,
                'annual_load_btu': new_load,
                'capex_annualized': hp_capex + retrofit_capex,
                'opex_annual': opex_hp_retrofit,
                'total_cost_annual': hp_capex + retrofit_capex + opex_hp_retrofit,
                'emissions_kg_co2': emissions_hp_retrofit,
                'area': area,
                'hdd65': hdd65,
            })
    
    scenario_df = pd.DataFrame(scenario_data)
    print(f"Created {len(scenario_df):,} scenario records")
    
    return scenario_df


def save_assumptions_table(measures, hp_options, output_dir):
    """Save assumptions table (Table 5)."""
    print("\nSaving assumptions table...")
    
    rows = []
    
    # Retrofit measures
    for name, measure in measures.items():
        rows.append({
            'measure_type': 'Retrofit',
            'name': measure.name,
            'unit_cost': measure.unit_cost,
            'lifetime_years': measure.lifetime_years,
            'load_reduction_pct': measure.load_reduction_pct * 100,
            'description': measure.description,
        })
    
    # Heat pump options
    for name, hp in hp_options.items():
        rows.append({
            'measure_type': 'Heat Pump',
            'name': hp.name,
            'unit_cost': hp.installed_cost,
            'lifetime_years': hp.lifetime_years,
            'load_reduction_pct': np.nan,  # N/A for HP
            'description': f'COP cold: {hp.cop_cold}, COP mild: {hp.cop_mild}',
        })
    
    assumptions_df = pd.DataFrame(rows)
    output_file = output_dir / "table5_retrofit_hp_assumptions.csv"
    assumptions_df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")


def main():
    """Main scenario generation pipeline."""
    print("=" * 70)
    print("RECS 2020 Heat Pump Retrofit Project - Retrofit Scenarios")
    print("=" * 70)
    
    # Load cleaned data
    df = pd.read_csv(CLEANED_DATA)
    print(f"Loaded {len(df):,} dwellings")
    
    # Define measures and options
    measures = define_retrofit_measures()
    hp_options = define_heat_pump_options()
    
    print(f"\nDefined {len(measures)} retrofit measures and {len(hp_options)} HP options")
    
    # Save assumptions table
    save_assumptions_table(measures, hp_options, OUTPUT_DIR)
    
    # Create scenario dataframe
    scenario_df = create_scenario_dataframe(df, measures, hp_options)
    
    # Save scenarios
    scenario_file = OUTPUT_DIR / "retrofit_scenarios.csv"
    scenario_df.to_csv(scenario_file, index=False)
    print(f"\n✓ Saved scenarios to {scenario_file}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Scenario Summary")
    print("=" * 70)
    print(f"\nTotal scenarios: {len(scenario_df):,}")
    print(f"Unique dwellings: {scenario_df['dwelling_id'].nunique():,}")
    print(f"Scenarios per dwelling: {len(scenario_df) / scenario_df['dwelling_id'].nunique():.1f}")
    
    print("\nCost and emissions ranges:")
    print(f"  Total cost (annual): ${scenario_df['total_cost_annual'].min():.0f} - ${scenario_df['total_cost_annual'].max():.0f}")
    print(f"  Emissions (kg CO₂/year): {scenario_df['emissions_kg_co2'].min():.0f} - {scenario_df['emissions_kg_co2'].max():.0f}")
    
    print("\n" + "=" * 70)
    print("Scenario generation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review scenario assumptions")
    print("2. Proceed to Step 6: NSGA-II Optimization")


if __name__ == "__main__":
    main()
