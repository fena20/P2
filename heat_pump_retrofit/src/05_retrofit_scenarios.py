"""
05_retrofit_scenarios.py
=========================
Retrofit and Heat Pump Scenario Analysis

This module defines retrofit measures and heat pump options,
calculates cost and emissions implications, and prepares
scenarios for optimization.

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Energy conversion constants
BTU_PER_KWH = 3412
BTU_PER_THERM = 100000
KG_CO2_PER_THERM = 5.3  # Natural gas CO2 emissions
TYPICAL_GAS_FURNACE_EFFICIENCY = 0.80  # 80% AFUE
TYPICAL_GAS_FURNACE_NEW_EFFICIENCY = 0.95  # High-efficiency condensing


@dataclass
class RetrofitMeasure:
    """Definition of an envelope retrofit measure."""
    name: str
    description: str
    intensity_reduction_pct: float  # Reduction in thermal intensity (%)
    cost_per_sqft: float  # Installation cost ($/sqft)
    lifetime_years: int
    applicable_envelope_classes: List[str]
    
    def get_reduction_factor(self) -> float:
        """Return multiplier for heating intensity after retrofit."""
        return 1.0 - (self.intensity_reduction_pct / 100.0)


@dataclass
class HeatPumpOption:
    """Definition of a heat pump system option."""
    name: str
    hp_type: str  # 'standard', 'cold_climate'
    cop_rating: float  # Rated COP at 47°F
    cop_low_temp: float  # COP at 17°F (for cold climate)
    hspf: float  # Heating Seasonal Performance Factor (BTU/Wh)
    capacity_tons: float  # Heating capacity
    cost_per_ton: float  # Equipment + installation cost
    lifetime_years: int
    backup_required: bool
    min_operating_temp: float  # Minimum operating temperature (°F)
    
    def get_effective_cop(self, outdoor_temp: float) -> float:
        """Estimate COP at a given outdoor temperature."""
        # Simple linear interpolation between rating points
        if outdoor_temp >= 47:
            return self.cop_rating
        elif outdoor_temp <= 17:
            return self.cop_low_temp
        else:
            slope = (self.cop_rating - self.cop_low_temp) / (47 - 17)
            return self.cop_low_temp + slope * (outdoor_temp - 17)
    
    def get_seasonal_cop(self, hdd: float) -> float:
        """
        Estimate seasonal average COP based on HDD.
        
        Higher HDD = colder climate = lower average COP.
        """
        # Rough approximation based on HDD
        # HDD < 3000: mild climate -> use rated COP
        # HDD 3000-6000: mixed -> interpolate
        # HDD > 6000: cold -> use low-temp COP
        if hdd <= 3000:
            return self.cop_rating
        elif hdd >= 6000:
            return self.cop_low_temp
        else:
            frac = (hdd - 3000) / 3000
            return self.cop_rating - frac * (self.cop_rating - self.cop_low_temp)


# Define standard retrofit measures (literature-based)
RETROFIT_MEASURES = {
    'none': RetrofitMeasure(
        name='No Retrofit',
        description='Baseline - no envelope improvements',
        intensity_reduction_pct=0.0,
        cost_per_sqft=0.0,
        lifetime_years=0,
        applicable_envelope_classes=['poor', 'medium', 'good']
    ),
    'air_seal': RetrofitMeasure(
        name='Air Sealing',
        description='Seal air leaks around windows, doors, and penetrations',
        intensity_reduction_pct=10.0,
        cost_per_sqft=0.50,
        lifetime_years=20,
        applicable_envelope_classes=['poor', 'medium']
    ),
    'attic_insulation': RetrofitMeasure(
        name='Attic Insulation',
        description='Add/upgrade attic insulation to R-49',
        intensity_reduction_pct=15.0,
        cost_per_sqft=1.50,
        lifetime_years=30,
        applicable_envelope_classes=['poor', 'medium']
    ),
    'wall_insulation': RetrofitMeasure(
        name='Wall Insulation',
        description='Add blown-in wall cavity insulation',
        intensity_reduction_pct=12.0,
        cost_per_sqft=2.50,
        lifetime_years=30,
        applicable_envelope_classes=['poor']
    ),
    'windows': RetrofitMeasure(
        name='Window Replacement',
        description='Replace with double-pane low-E windows',
        intensity_reduction_pct=8.0,
        cost_per_sqft=3.00,  # Amortized over whole house
        lifetime_years=25,
        applicable_envelope_classes=['poor', 'medium']
    ),
    'comprehensive': RetrofitMeasure(
        name='Comprehensive Retrofit',
        description='Air sealing + attic + windows (bundled)',
        intensity_reduction_pct=30.0,
        cost_per_sqft=5.00,
        lifetime_years=25,
        applicable_envelope_classes=['poor']
    ),
}

# Define heat pump options
HEAT_PUMP_OPTIONS = {
    'none': None,  # Keep existing gas system
    'standard_hp': HeatPumpOption(
        name='Standard Heat Pump',
        hp_type='standard',
        cop_rating=3.5,
        cop_low_temp=2.0,
        hspf=9.5,
        capacity_tons=3.0,
        cost_per_ton=4000,
        lifetime_years=15,
        backup_required=True,
        min_operating_temp=30
    ),
    'cold_climate_hp': HeatPumpOption(
        name='Cold Climate Heat Pump',
        hp_type='cold_climate',
        cop_rating=4.0,
        cop_low_temp=2.5,
        hspf=11.0,
        capacity_tons=3.0,
        cost_per_ton=6000,
        lifetime_years=15,
        backup_required=True,
        min_operating_temp=5
    ),
    'high_perf_hp': HeatPumpOption(
        name='High-Performance Cold Climate HP',
        hp_type='cold_climate',
        cop_rating=4.5,
        cop_low_temp=3.0,
        hspf=13.0,
        capacity_tons=3.0,
        cost_per_ton=8000,
        lifetime_years=18,
        backup_required=False,  # Can operate down to very low temps
        min_operating_temp=-10
    ),
}


@dataclass
class EnergyPrices:
    """Regional energy prices."""
    electricity_per_kwh: float  # $/kWh
    natural_gas_per_therm: float  # $/therm
    region: str = "national"


# Default energy prices (2023 U.S. averages)
DEFAULT_PRICES = EnergyPrices(
    electricity_per_kwh=0.15,
    natural_gas_per_therm=1.20,
    region="national"
)

# Regional price scenarios
REGIONAL_PRICES = {
    'Northeast': EnergyPrices(0.22, 1.50, 'Northeast'),
    'Midwest': EnergyPrices(0.14, 0.95, 'Midwest'),
    'South': EnergyPrices(0.12, 1.10, 'South'),
    'West': EnergyPrices(0.18, 1.30, 'West'),
    'Pacific': EnergyPrices(0.20, 1.40, 'Pacific'),
}


@dataclass
class GridEmissionFactor:
    """Grid CO2 emission factor for electricity."""
    kg_co2_per_kwh: float
    region: str = "national"
    year: int = 2023


# Grid emission factors by region (approximate, kg CO2/kWh)
GRID_EMISSIONS = {
    'national': GridEmissionFactor(0.42, 'national', 2023),
    'Northeast': GridEmissionFactor(0.30, 'Northeast', 2023),
    'Midwest': GridEmissionFactor(0.60, 'Midwest', 2023),
    'South': GridEmissionFactor(0.45, 'South', 2023),
    'West': GridEmissionFactor(0.35, 'West', 2023),
    'Pacific': GridEmissionFactor(0.25, 'Pacific', 2023),
    # Future scenarios
    'decarbonized_2030': GridEmissionFactor(0.25, 'national', 2030),
    'decarbonized_2040': GridEmissionFactor(0.10, 'national', 2040),
}


def calculate_gas_heating_cost(
    heating_energy_btu: float,
    gas_price_per_therm: float,
    furnace_efficiency: float = TYPICAL_GAS_FURNACE_EFFICIENCY
) -> float:
    """
    Calculate annual gas heating cost.
    
    Parameters
    ----------
    heating_energy_btu : float
        Annual heating energy requirement (BTU delivered to space)
    gas_price_per_therm : float
        Natural gas price ($/therm)
    furnace_efficiency : float
        Gas furnace efficiency (AFUE)
    
    Returns
    -------
    float
        Annual heating cost ($)
    """
    gas_consumption_btu = heating_energy_btu / furnace_efficiency
    gas_consumption_therms = gas_consumption_btu / BTU_PER_THERM
    annual_cost = gas_consumption_therms * gas_price_per_therm
    return annual_cost


def calculate_hp_heating_cost(
    heating_energy_btu: float,
    electricity_price_per_kwh: float,
    seasonal_cop: float
) -> float:
    """
    Calculate annual heat pump heating cost.
    
    Parameters
    ----------
    heating_energy_btu : float
        Annual heating energy requirement (BTU delivered to space)
    electricity_price_per_kwh : float
        Electricity price ($/kWh)
    seasonal_cop : float
        Seasonal average coefficient of performance
    
    Returns
    -------
    float
        Annual heating cost ($)
    """
    heating_energy_kwh = heating_energy_btu / BTU_PER_KWH
    electricity_consumption_kwh = heating_energy_kwh / seasonal_cop
    annual_cost = electricity_consumption_kwh * electricity_price_per_kwh
    return annual_cost


def calculate_gas_emissions(
    heating_energy_btu: float,
    furnace_efficiency: float = TYPICAL_GAS_FURNACE_EFFICIENCY
) -> float:
    """
    Calculate annual CO2 emissions from gas heating.
    
    Returns
    -------
    float
        Annual emissions (kg CO2)
    """
    gas_consumption_therms = (heating_energy_btu / furnace_efficiency) / BTU_PER_THERM
    emissions = gas_consumption_therms * KG_CO2_PER_THERM
    return emissions


def calculate_hp_emissions(
    heating_energy_btu: float,
    seasonal_cop: float,
    grid_emission_factor: float
) -> float:
    """
    Calculate annual CO2 emissions from heat pump heating.
    
    Returns
    -------
    float
        Annual emissions (kg CO2)
    """
    heating_energy_kwh = heating_energy_btu / BTU_PER_KWH
    electricity_consumption_kwh = heating_energy_kwh / seasonal_cop
    emissions = electricity_consumption_kwh * grid_emission_factor
    return emissions


def annualize_capital_cost(
    capital_cost: float,
    lifetime_years: int,
    discount_rate: float = 0.05
) -> float:
    """
    Convert capital cost to annualized cost using CRF.
    
    Parameters
    ----------
    capital_cost : float
        Upfront capital cost ($)
    lifetime_years : int
        Equipment/measure lifetime
    discount_rate : float
        Discount rate (default 5%)
    
    Returns
    -------
    float
        Annualized capital cost ($/year)
    """
    if lifetime_years <= 0:
        return 0.0
    
    crf = (discount_rate * (1 + discount_rate)**lifetime_years) / \
          ((1 + discount_rate)**lifetime_years - 1)
    return capital_cost * crf


def evaluate_scenario(
    baseline_intensity: float,
    heated_sqft: float,
    hdd: float,
    retrofit: RetrofitMeasure,
    heat_pump: Optional[HeatPumpOption],
    prices: EnergyPrices,
    grid_emissions: GridEmissionFactor,
    discount_rate: float = 0.05
) -> Dict:
    """
    Evaluate a single retrofit + heat pump scenario.
    
    Returns dictionary with costs and emissions.
    """
    # Calculate baseline heating energy
    baseline_heating_btu = baseline_intensity * heated_sqft * hdd
    
    # Apply retrofit reduction
    post_retrofit_intensity = baseline_intensity * retrofit.get_reduction_factor()
    post_retrofit_btu = post_retrofit_intensity * heated_sqft * hdd
    
    # Capital costs
    retrofit_cost = retrofit.cost_per_sqft * heated_sqft
    retrofit_annual = annualize_capital_cost(retrofit_cost, retrofit.lifetime_years, discount_rate)
    
    if heat_pump is not None:
        hp_cost = heat_pump.cost_per_ton * heat_pump.capacity_tons
        hp_annual_capital = annualize_capital_cost(hp_cost, heat_pump.lifetime_years, discount_rate)
        
        # Seasonal COP based on climate
        seasonal_cop = heat_pump.get_seasonal_cop(hdd)
        
        # Operating costs (heat pump)
        annual_op_cost = calculate_hp_heating_cost(
            post_retrofit_btu, prices.electricity_per_kwh, seasonal_cop
        )
        
        # Emissions (heat pump)
        annual_emissions = calculate_hp_emissions(
            post_retrofit_btu, seasonal_cop, grid_emissions.kg_co2_per_kwh
        )
        
        heating_system = heat_pump.name
    else:
        hp_cost = 0
        hp_annual_capital = 0
        
        # Operating costs (gas furnace)
        annual_op_cost = calculate_gas_heating_cost(
            post_retrofit_btu, prices.natural_gas_per_therm
        )
        
        # Emissions (gas)
        annual_emissions = calculate_gas_emissions(post_retrofit_btu)
        
        heating_system = 'Gas Furnace'
    
    # Total costs
    total_capital = retrofit_cost + hp_cost
    total_annual_capital = retrofit_annual + hp_annual_capital
    total_annual_cost = total_annual_capital + annual_op_cost
    
    return {
        'retrofit': retrofit.name,
        'heating_system': heating_system,
        'baseline_heating_btu': baseline_heating_btu,
        'post_retrofit_btu': post_retrofit_btu,
        'intensity_reduction_pct': retrofit.intensity_reduction_pct,
        'retrofit_cost': retrofit_cost,
        'hp_cost': hp_cost,
        'total_capital': total_capital,
        'retrofit_annual': retrofit_annual,
        'hp_annual_capital': hp_annual_capital,
        'annual_op_cost': annual_op_cost,
        'total_annual_cost': total_annual_cost,
        'annual_emissions_kg': annual_emissions,
        'annual_emissions_tons': annual_emissions / 1000,
    }


def evaluate_all_scenarios(
    df: pd.DataFrame,
    prices: EnergyPrices = DEFAULT_PRICES,
    grid_emissions: GridEmissionFactor = GRID_EMISSIONS['national']
) -> pd.DataFrame:
    """
    Evaluate all retrofit × heat pump combinations for each household.
    """
    logger.info("Evaluating all scenarios...")
    
    results = []
    
    retrofit_options = ['none', 'air_seal', 'attic_insulation', 'comprehensive']
    hp_options = ['none', 'standard_hp', 'cold_climate_hp']
    
    for idx, row in df.iterrows():
        household_id = row.get('DOEID', idx)
        baseline_intensity = row['Thermal_Intensity_I']
        heated_sqft = row['A_heated']
        hdd = row['HDD65']
        envelope_class = row.get('envelope_class', 'medium')
        
        for retrofit_key in retrofit_options:
            retrofit = RETROFIT_MEASURES[retrofit_key]
            
            # Check if retrofit is applicable
            if envelope_class not in retrofit.applicable_envelope_classes:
                continue
            
            for hp_key in hp_options:
                hp = HEAT_PUMP_OPTIONS.get(hp_key)
                
                scenario_result = evaluate_scenario(
                    baseline_intensity, heated_sqft, hdd,
                    retrofit, hp, prices, grid_emissions
                )
                
                scenario_result['household_id'] = household_id
                scenario_result['envelope_class'] = envelope_class
                scenario_result['hdd'] = hdd
                scenario_result['heated_sqft'] = heated_sqft
                scenario_result['scenario_id'] = f"{retrofit_key}_{hp_key}"
                
                results.append(scenario_result)
    
    results_df = pd.DataFrame(results)
    logger.info(f"Evaluated {len(results_df)} scenarios for {len(df)} households")
    
    return results_df


def generate_table5_assumptions():
    """
    Generate Table 5: Retrofit and heat pump option assumptions.
    """
    logger.info("Generating Table 5: Assumptions")
    
    # Retrofit measures
    retrofit_data = []
    for key, measure in RETROFIT_MEASURES.items():
        retrofit_data.append({
            'Measure': measure.name,
            'Description': measure.description,
            'Intensity Reduction (%)': measure.intensity_reduction_pct,
            'Cost ($/sqft)': measure.cost_per_sqft,
            'Lifetime (years)': measure.lifetime_years,
        })
    
    retrofit_df = pd.DataFrame(retrofit_data)
    retrofit_df.to_csv(TABLES_DIR / "table5a_retrofit_assumptions.csv", index=False)
    
    # Heat pump options
    hp_data = []
    for key, hp in HEAT_PUMP_OPTIONS.items():
        if hp is not None:
            hp_data.append({
                'Option': hp.name,
                'Type': hp.hp_type,
                'COP (47°F)': hp.cop_rating,
                'COP (17°F)': hp.cop_low_temp,
                'HSPF': hp.hspf,
                'Cost ($/ton)': hp.cost_per_ton,
                'Lifetime (years)': hp.lifetime_years,
            })
    
    hp_df = pd.DataFrame(hp_data)
    hp_df.to_csv(TABLES_DIR / "table5b_heat_pump_assumptions.csv", index=False)
    
    # Energy prices and emissions
    price_data = []
    for region, prices in REGIONAL_PRICES.items():
        price_data.append({
            'Region': region,
            'Electricity ($/kWh)': prices.electricity_per_kwh,
            'Natural Gas ($/therm)': prices.natural_gas_per_therm,
            'Grid CO2 (kg/kWh)': GRID_EMISSIONS.get(region, GRID_EMISSIONS['national']).kg_co2_per_kwh,
        })
    
    price_df = pd.DataFrame(price_data)
    price_df.to_csv(TABLES_DIR / "table5c_energy_prices.csv", index=False)
    
    return retrofit_df, hp_df, price_df


def identify_dominant_scenarios(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Pareto-dominant scenarios for each household.
    
    A scenario is Pareto-dominant if no other scenario is better
    on both cost AND emissions.
    """
    logger.info("Identifying Pareto-dominant scenarios...")
    
    dominant_scenarios = []
    
    for household_id in results_df['household_id'].unique():
        household_df = results_df[results_df['household_id'] == household_id]
        
        for idx, scenario in household_df.iterrows():
            is_dominated = False
            
            for _, other in household_df.iterrows():
                if (other['total_annual_cost'] < scenario['total_annual_cost'] and
                    other['annual_emissions_kg'] < scenario['annual_emissions_kg']):
                    is_dominated = True
                    break
            
            if not is_dominated:
                dominant_scenarios.append({
                    **scenario.to_dict(),
                    'is_pareto_optimal': True
                })
    
    dominant_df = pd.DataFrame(dominant_scenarios)
    logger.info(f"Found {len(dominant_df)} Pareto-optimal scenarios")
    
    return dominant_df


def summarize_retrofit_potential(df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize retrofit potential by envelope class and climate.
    """
    logger.info("Summarizing retrofit potential...")
    
    # Get baseline (no retrofit, gas) scenario
    baseline_mask = (results_df['retrofit'] == 'No Retrofit') & \
                    (results_df['heating_system'] == 'Gas Furnace')
    baseline_df = results_df[baseline_mask].set_index('household_id')
    
    summaries = []
    
    for env_class in ['poor', 'medium', 'good']:
        for climate in ['mild', 'mixed', 'cold']:
            mask = (df['envelope_class'] == env_class) & (df['climate_zone'] == climate)
            subset = df[mask]
            
            if len(subset) > 10:
                hh_ids = subset['DOEID'].values if 'DOEID' in subset.columns else subset.index
                
                # Best HP scenario for this group
                hp_scenarios = results_df[
                    (results_df['household_id'].isin(hh_ids)) &
                    (results_df['heating_system'] != 'Gas Furnace')
                ]
                
                if len(hp_scenarios) > 0:
                    # Calculate average savings vs baseline
                    for hh_id in hh_ids:
                        if hh_id in baseline_df.index:
                            baseline_cost = baseline_df.loc[hh_id, 'total_annual_cost']
                            baseline_emissions = baseline_df.loc[hh_id, 'annual_emissions_kg']
                            
                            hp_rows = hp_scenarios[hp_scenarios['household_id'] == hh_id]
                            if len(hp_rows) > 0:
                                best_hp = hp_rows.loc[hp_rows['total_annual_cost'].idxmin()]
                                
                                summaries.append({
                                    'envelope_class': env_class,
                                    'climate_zone': climate,
                                    'household_id': hh_id,
                                    'baseline_cost': baseline_cost,
                                    'best_hp_cost': best_hp['total_annual_cost'],
                                    'cost_savings': baseline_cost - best_hp['total_annual_cost'],
                                    'baseline_emissions': baseline_emissions,
                                    'best_hp_emissions': best_hp['annual_emissions_kg'],
                                    'emissions_reduction': baseline_emissions - best_hp['annual_emissions_kg'],
                                })
    
    summary_df = pd.DataFrame(summaries)
    
    # Aggregate by envelope class and climate
    agg_summary = summary_df.groupby(['envelope_class', 'climate_zone']).agg({
        'cost_savings': 'mean',
        'emissions_reduction': 'mean',
        'household_id': 'count'
    }).rename(columns={'household_id': 'n_households'})
    
    agg_summary.to_csv(TABLES_DIR / "retrofit_potential_summary.csv")
    
    return agg_summary


def run_retrofit_analysis() -> dict:
    """
    Main function to run retrofit scenario analysis.
    """
    logger.info("=" * 60)
    logger.info("Retrofit and Heat Pump Scenario Analysis")
    logger.info("=" * 60)
    
    # Load data
    data_path = OUTPUT_DIR / "03_gas_heated_clean.csv"
    df = pd.read_csv(data_path)
    
    # Sample for initial analysis (full dataset can be slow)
    sample_size = min(1000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    logger.info(f"Analyzing {sample_size} sample households")
    
    # Generate assumption tables
    generate_table5_assumptions()
    
    # Evaluate all scenarios
    results_df = evaluate_all_scenarios(df_sample)
    
    # Save full results
    results_df.to_csv(RESULTS_DIR / "scenario_results.csv", index=False)
    
    # Identify Pareto-optimal scenarios
    pareto_df = identify_dominant_scenarios(results_df)
    pareto_df.to_csv(RESULTS_DIR / "pareto_optimal_scenarios.csv", index=False)
    
    # Summarize retrofit potential
    summary = summarize_retrofit_potential(df_sample, results_df)
    
    logger.info("=" * 60)
    logger.info("Retrofit analysis complete!")
    logger.info("=" * 60)
    
    return {
        'sample_data': df_sample,
        'scenario_results': results_df,
        'pareto_optimal': pareto_df,
        'summary': summary,
    }


if __name__ == "__main__":
    results = run_retrofit_analysis()
    
    print("\n" + "=" * 60)
    print("RETROFIT ANALYSIS SUMMARY")
    print("=" * 60)
    print("\nRetrofit Potential by Envelope Class and Climate:")
    print(results['summary'].to_string())
