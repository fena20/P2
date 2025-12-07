"""
Step 5 - Retrofit and Heat Pump Scenarios

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology

This script:
1. Defines retrofit measures (insulation, windows, etc.) and costs
2. Defines heat pump options (standard, cold-climate) and performance
3. Adjusts thermal intensity based on retrofit combinations
4. Calculates capital and operational costs
5. Computes CO2 emissions for scenarios
6. Generates Table 5 (retrofit and HP assumptions)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RetrofitScenarios:
    """
    Retrofit and heat pump scenario modeling
    """
    
    def __init__(self, output_dir='../recs_output'):
        """
        Initialize retrofit scenario generator
        
        Parameters
        ----------
        output_dir : str
            Output directory for tables
        """
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / 'tables'
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Define retrofit measures and heat pump options
        self.retrofit_measures = {}
        self.heat_pump_options = {}
        self.fuel_prices = {}
        self.emission_factors = {}
        
    def define_retrofit_measures(self):
        """
        Define envelope retrofit measures with costs and energy savings
        
        Based on literature values (DOE, NREL, etc.)
        """
        print("=" * 80)
        print("Defining Retrofit Measures")
        print("=" * 80)
        
        # Retrofit measure definitions
        # Format: {measure_name: {cost, lifetime, savings_fraction, ...}}
        
        self.retrofit_measures = {
            'none': {
                'description': 'No envelope retrofits',
                'cost_per_sqft': 0.0,
                'lifetime_years': 0,
                'intensity_reduction_pct': 0.0,
                'applicable_to': 'all'
            },
            'attic_insulation': {
                'description': 'Upgrade attic insulation to R-49',
                'cost_per_sqft': 1.50,  # USD per sqft of floor area
                'lifetime_years': 30,
                'intensity_reduction_pct': 15.0,  # 15% reduction in heating load
                'applicable_to': 'all'
            },
            'wall_insulation': {
                'description': 'Add wall insulation (blown-in or injection)',
                'cost_per_sqft': 3.00,
                'lifetime_years': 30,
                'intensity_reduction_pct': 20.0,
                'applicable_to': 'poor,medium'  # Most beneficial for leaky homes
            },
            'window_replacement': {
                'description': 'Replace with double/triple pane low-E windows',
                'cost_per_sqft': 8.00,  # Higher cost per sqft
                'lifetime_years': 25,
                'intensity_reduction_pct': 12.0,
                'applicable_to': 'poor,medium'
            },
            'air_sealing': {
                'description': 'Professional air sealing and weatherization',
                'cost_per_sqft': 0.50,
                'lifetime_years': 20,
                'intensity_reduction_pct': 10.0,
                'applicable_to': 'poor'
            },
            'comprehensive': {
                'description': 'Comprehensive retrofit (attic + walls + windows + sealing)',
                'cost_per_sqft': 12.0,
                'lifetime_years': 25,
                'intensity_reduction_pct': 45.0,  # Combined effect (not fully additive)
                'applicable_to': 'poor'
            }
        }
        
        print(f"Defined {len(self.retrofit_measures)} retrofit measures:")
        for name, specs in self.retrofit_measures.items():
            print(f"  - {name:25s}: {specs['description']}")
        
        return self
    
    def define_heat_pump_options(self):
        """
        Define heat pump technology options
        """
        print("\n" + "=" * 80)
        print("Defining Heat Pump Options")
        print("=" * 80)
        
        self.heat_pump_options = {
            'none': {
                'description': 'No heat pump (continue with natural gas)',
                'capital_cost_per_ton': 0,
                'lifetime_years': 0,
                'cop_at_47F': 0,
                'cop_at_17F': 0,
                'cop_at_5F': 0,
                'backup_required_temp_F': None,
                'applicable_climate': 'all'
            },
            'standard_ashp': {
                'description': 'Standard air-source heat pump (good for mild/moderate climates)',
                'capital_cost_per_ton': 3500,  # USD per ton of cooling capacity
                'lifetime_years': 15,
                'cop_at_47F': 3.5,
                'cop_at_17F': 2.0,
                'cop_at_5F': 1.2,
                'backup_required_temp_F': 20,  # Need backup below 20°F
                'applicable_climate': 'mild,moderate'
            },
            'cold_climate_hp': {
                'description': 'Cold-climate heat pump (maintains performance in extreme cold)',
                'capital_cost_per_ton': 5000,
                'lifetime_years': 18,
                'cop_at_47F': 3.8,
                'cop_at_17F': 2.5,
                'cop_at_5F': 2.0,
                'backup_required_temp_F': 0,  # Can operate down to 0°F or lower
                'applicable_climate': 'all'
            },
            'ground_source_hp': {
                'description': 'Ground-source (geothermal) heat pump',
                'capital_cost_per_ton': 8000,
                'lifetime_years': 25,
                'cop_at_47F': 4.0,
                'cop_at_17F': 3.8,
                'cop_at_5F': 3.5,
                'backup_required_temp_F': -40,  # Very reliable
                'applicable_climate': 'all'
            }
        }
        
        print(f"Defined {len(self.heat_pump_options)} heat pump options:")
        for name, specs in self.heat_pump_options.items():
            print(f"  - {name:20s}: {specs['description']}")
        
        return self
    
    def define_fuel_prices_and_emissions(self):
        """
        Define fuel price scenarios and emission factors
        """
        print("\n" + "=" * 80)
        print("Defining Fuel Prices and Emission Factors")
        print("=" * 80)
        
        # Fuel prices (representative values)
        self.fuel_prices = {
            'natural_gas': {
                'low': 8.0,      # USD per million BTU (or per 1000 cubic feet ~ 10 therms)
                'medium': 12.0,
                'high': 16.0,
                'unit': 'USD/MMBtu'
            },
            'electricity': {
                'low': 0.10,     # USD per kWh
                'medium': 0.13,
                'high': 0.20,
                'unit': 'USD/kWh'
            }
        }
        
        # Emission factors
        self.emission_factors = {
            'natural_gas': {
                'co2_kg_per_mmbtu': 53.06,  # kg CO2 per million BTU (EPA)
                'source': 'EPA'
            },
            'electricity': {
                # Grid emission factors by region (kg CO2 per kWh)
                # These vary significantly by state/region
                'us_average': 0.386,  # kg CO2 per kWh (EPA eGRID 2021)
                'northeast': 0.298,
                'midwest': 0.456,
                'south': 0.407,
                'west': 0.309,
                'source': 'EPA eGRID 2021'
            }
        }
        
        print("\nNatural Gas Prices:")
        for scenario, price in self.fuel_prices['natural_gas'].items():
            if scenario != 'unit':
                print(f"  {scenario:10s}: ${price:.2f}/MMBtu")
        
        print("\nElectricity Prices:")
        for scenario, price in self.fuel_prices['electricity'].items():
            if scenario != 'unit':
                print(f"  {scenario:10s}: ${price:.3f}/kWh")
        
        print("\nEmission Factors:")
        print(f"  Natural gas: {self.emission_factors['natural_gas']['co2_kg_per_mmbtu']:.2f} kg CO2/MMBtu")
        print(f"  Electricity (US avg): {self.emission_factors['electricity']['us_average']:.3f} kg CO2/kWh")
        
        return self
    
    def create_table5_retrofit_assumptions(self):
        """
        Generate Table 5: Retrofit and heat pump option assumptions
        """
        print("\n" + "=" * 80)
        print("Generating Table 5: Retrofit and HP Assumptions")
        print("=" * 80)
        
        # Part A: Retrofit measures
        retrofit_records = []
        for name, specs in self.retrofit_measures.items():
            retrofit_records.append({
                'measure': name,
                'description': specs['description'],
                'cost_per_sqft_USD': specs['cost_per_sqft'],
                'lifetime_years': specs['lifetime_years'],
                'intensity_reduction_pct': specs['intensity_reduction_pct'],
                'applicable_to': specs['applicable_to']
            })
        
        df_retrofits = pd.DataFrame(retrofit_records)
        
        # Part B: Heat pump options
        hp_records = []
        for name, specs in self.heat_pump_options.items():
            hp_records.append({
                'option': name,
                'description': specs['description'],
                'capital_cost_per_ton_USD': specs['capital_cost_per_ton'],
                'lifetime_years': specs['lifetime_years'],
                'cop_47F': specs['cop_at_47F'],
                'cop_17F': specs['cop_at_17F'],
                'cop_5F': specs['cop_at_5F'],
                'applicable_climate': specs['applicable_climate']
            })
        
        df_hp = pd.DataFrame(hp_records)
        
        # Part C: Fuel prices and emissions
        price_records = []
        
        for scenario in ['low', 'medium', 'high']:
            price_records.append({
                'scenario': scenario,
                'natural_gas_USD_per_MMBtu': self.fuel_prices['natural_gas'][scenario],
                'electricity_USD_per_kWh': self.fuel_prices['electricity'][scenario],
                'gas_co2_kg_per_MMBtu': self.emission_factors['natural_gas']['co2_kg_per_mmbtu'],
                'electricity_co2_kg_per_kWh': self.emission_factors['electricity']['us_average']
            })
        
        df_prices = pd.DataFrame(price_records)
        
        # Save tables
        retrofit_path = self.tables_dir / 'table5a_retrofit_measures.csv'
        df_retrofits.to_csv(retrofit_path, index=False)
        print(f"Saved: {retrofit_path}")
        
        hp_path = self.tables_dir / 'table5b_heat_pump_options.csv'
        df_hp.to_csv(hp_path, index=False)
        print(f"Saved: {hp_path}")
        
        prices_path = self.tables_dir / 'table5c_fuel_prices_emissions.csv'
        df_prices.to_csv(prices_path, index=False)
        print(f"Saved: {prices_path}")
        
        # Create combined table as text
        table_txt_path = self.tables_dir / 'table5_combined_assumptions.txt'
        with open(table_txt_path, 'w') as f:
            f.write("Table 5. Assumed cost and technical performance of retrofit measures and\n")
            f.write("heat pump options used in the optimization scenarios\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("Part A: Envelope Retrofit Measures\n")
            f.write("-" * 100 + "\n")
            f.write(df_retrofits.to_string(index=False))
            
            f.write("\n\n\nPart B: Heat Pump Technology Options\n")
            f.write("-" * 100 + "\n")
            f.write(df_hp.to_string(index=False))
            
            f.write("\n\n\nPart C: Fuel Price and Emission Factor Scenarios\n")
            f.write("-" * 100 + "\n")
            f.write(df_prices.to_string(index=False))
            
            f.write("\n\n\nNotes:\n")
            f.write("-" * 100 + "\n")
            f.write("- Costs are representative values based on DOE, NREL, and industry sources\n")
            f.write("- Intensity reduction percentages are approximate and vary by home characteristics\n")
            f.write("- COP (Coefficient of Performance) values are at specified outdoor temperatures\n")
            f.write("- Emission factors from EPA (natural gas) and eGRID 2021 (electricity)\n")
            f.write("- All costs in 2020 USD\n")
        
        print(f"Saved combined table: {table_txt_path}")
        
        self.df_retrofits = df_retrofits
        self.df_hp = df_hp
        self.df_prices = df_prices
        
        return self
    
    def calculate_scenario_costs_emissions(self, example_home):
        """
        Calculate costs and emissions for an example home scenario
        
        Parameters
        ----------
        example_home : dict
            Home characteristics (sqft, HDD, baseline_intensity, etc.)
        
        Returns
        -------
        DataFrame with scenario results
        """
        print("\n" + "=" * 80)
        print("Example Scenario Calculations")
        print("=" * 80)
        
        # Extract home characteristics
        sqft = example_home.get('sqft', 2000)
        hdd = example_home.get('hdd', 5000)
        baseline_intensity = example_home.get('baseline_intensity', 0.020)  # BTU/sqft/HDD
        
        print(f"\nExample Home:")
        print(f"  Floor area: {sqft:.0f} sqft")
        print(f"  HDD: {hdd:.0f}")
        print(f"  Baseline thermal intensity: {baseline_intensity:.6f} BTU/sqft/HDD")
        
        # Baseline annual heating energy (BTU)
        baseline_heating_btu = baseline_intensity * sqft * hdd
        print(f"  Baseline annual heating: {baseline_heating_btu/1e6:.1f} million BTU")
        
        # Scenario results
        results = []
        
        # Scenario 1: Baseline (gas, no retrofits)
        gas_price = self.fuel_prices['natural_gas']['medium']
        gas_emission = self.emission_factors['natural_gas']['co2_kg_per_mmbtu']
        
        baseline_cost = baseline_heating_btu / 1e6 * gas_price
        baseline_emissions = baseline_heating_btu / 1e6 * gas_emission
        
        results.append({
            'scenario': 'Baseline (gas, no retrofit)',
            'retrofit': 'none',
            'heat_pump': 'none',
            'annual_heating_btu': baseline_heating_btu,
            'annual_cost_USD': baseline_cost,
            'annual_co2_kg': baseline_emissions,
            'capital_cost_USD': 0,
        })
        
        print(f"\nBaseline scenario:")
        print(f"  Annual cost: ${baseline_cost:.2f}")
        print(f"  Annual CO2: {baseline_emissions:.1f} kg")
        
        # Scenario 2: Attic insulation + gas
        attic_retrofit = self.retrofit_measures['attic_insulation']
        reduced_intensity_attic = baseline_intensity * (1 - attic_retrofit['intensity_reduction_pct']/100)
        heating_btu_attic = reduced_intensity_attic * sqft * hdd
        cost_attic = heating_btu_attic / 1e6 * gas_price
        emissions_attic = heating_btu_attic / 1e6 * gas_emission
        capital_attic = attic_retrofit['cost_per_sqft'] * sqft
        
        results.append({
            'scenario': 'Attic insulation + gas',
            'retrofit': 'attic_insulation',
            'heat_pump': 'none',
            'annual_heating_btu': heating_btu_attic,
            'annual_cost_USD': cost_attic,
            'annual_co2_kg': emissions_attic,
            'capital_cost_USD': capital_attic,
        })
        
        # Scenario 3: Standard ASHP (no retrofit)
        hp_option = self.heat_pump_options['standard_ashp']
        # Approximate average COP
        avg_cop = 2.5
        # Convert BTU to kWh for electricity
        heating_kwh_hp = baseline_heating_btu / 3412.14 / avg_cop
        elec_price = self.fuel_prices['electricity']['medium']
        cost_hp = heating_kwh_hp * elec_price
        elec_emission = self.emission_factors['electricity']['us_average']
        emissions_hp = heating_kwh_hp * elec_emission
        # Estimate system size (1 ton per 600 sqft as rough rule)
        tons_needed = sqft / 600
        capital_hp = hp_option['capital_cost_per_ton'] * tons_needed
        
        results.append({
            'scenario': 'Standard ASHP (no retrofit)',
            'retrofit': 'none',
            'heat_pump': 'standard_ashp',
            'annual_heating_btu': baseline_heating_btu,
            'annual_cost_USD': cost_hp,
            'annual_co2_kg': emissions_hp,
            'capital_cost_USD': capital_hp,
        })
        
        # Scenario 4: Attic insulation + Standard ASHP
        heating_kwh_hp_retrofit = heating_btu_attic / 3412.14 / avg_cop
        cost_hp_retrofit = heating_kwh_hp_retrofit * elec_price
        emissions_hp_retrofit = heating_kwh_hp_retrofit * elec_emission
        capital_combined = capital_attic + capital_hp
        
        results.append({
            'scenario': 'Attic insulation + Standard ASHP',
            'retrofit': 'attic_insulation',
            'heat_pump': 'standard_ashp',
            'annual_heating_btu': heating_btu_attic,
            'annual_cost_USD': cost_hp_retrofit,
            'annual_co2_kg': emissions_hp_retrofit,
            'capital_cost_USD': capital_combined,
        })
        
        # Create DataFrame
        df_scenarios = pd.DataFrame(results)
        
        # Add annualized capital cost (assuming 5% discount rate, 20-year horizon)
        discount_rate = 0.05
        years = 20
        crf = discount_rate * (1 + discount_rate)**years / ((1 + discount_rate)**years - 1)
        df_scenarios['annualized_capital_USD'] = df_scenarios['capital_cost_USD'] * crf
        df_scenarios['total_annualized_cost_USD'] = (df_scenarios['annual_cost_USD'] + 
                                                      df_scenarios['annualized_capital_USD'])
        
        print("\nExample Scenario Results:")
        print(df_scenarios[['scenario', 'annual_cost_USD', 'annual_co2_kg', 
                           'capital_cost_USD', 'total_annualized_cost_USD']].to_string(index=False))
        
        # Save example
        example_path = self.tables_dir / 'example_scenario_calculations.csv'
        df_scenarios.to_csv(example_path, index=False)
        print(f"\nSaved example calculations: {example_path}")
        
        return df_scenarios
    
    def create_scenario_summary(self):
        """
        Create summary documentation for retrofit scenarios
        """
        print("\n" + "=" * 80)
        print("Creating Scenario Summary")
        print("=" * 80)
        
        summary_path = self.tables_dir / 'retrofit_scenario_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("RECS 2020 Heat Pump Retrofit Analysis - Scenario Framework Summary\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("Retrofit Measures Available:\n")
            f.write("-" * 100 + "\n")
            for name, specs in self.retrofit_measures.items():
                f.write(f"{name:25s}: {specs['description']}\n")
                f.write(f"  → Cost: ${specs['cost_per_sqft']:.2f}/sqft, ")
                f.write(f"Savings: {specs['intensity_reduction_pct']:.0f}%, ")
                f.write(f"Lifetime: {specs['lifetime_years']} years\n\n")
            
            f.write("\nHeat Pump Options:\n")
            f.write("-" * 100 + "\n")
            for name, specs in self.heat_pump_options.items():
                f.write(f"{name:20s}: {specs['description']}\n")
                f.write(f"  → Cost: ${specs['capital_cost_per_ton']:.0f}/ton, ")
                f.write(f"COP@17°F: {specs['cop_at_17F']:.1f}, ")
                f.write(f"Lifetime: {specs['lifetime_years']} years\n\n")
            
            f.write("\nScenario Calculation Method:\n")
            f.write("-" * 100 + "\n")
            f.write("1. Start with baseline thermal intensity (from XGBoost model predictions)\n")
            f.write("2. Apply retrofit intensity reduction percentage\n")
            f.write("3. Calculate annual heating energy: E = I_adjusted × Area × HDD\n")
            f.write("4. For gas heating: Cost = E × gas_price, Emissions = E × gas_emission_factor\n")
            f.write("5. For heat pump: kWh = E / (3412.14 × COP), Cost = kWh × elec_price\n")
            f.write("6. Add annualized capital costs using CRF method\n")
            f.write("7. Total annualized cost = OpEx + CapEx_annualized\n")
            
            f.write("\n\nKey Assumptions:\n")
            f.write("-" * 100 + "\n")
            f.write("- Discount rate: 5%\n")
            f.write("- Analysis period: 20 years\n")
            f.write("- Heat pump sizing: ~1 ton per 600 sqft (rough rule of thumb)\n")
            f.write("- Backup heating: electric resistance for temperatures below HP cutoff\n")
            f.write("- Grid emission factors: EPA eGRID 2021 (subject to decarbonization scenarios)\n")
            
            f.write("\n\nNext Steps:\n")
            f.write("-" * 100 + "\n")
            f.write("1. Use these definitions in NSGA-II optimization (06_nsga2_optimization.py)\n")
            f.write("2. Generate Pareto fronts of cost vs emissions for different archetypes\n")
            f.write("3. Identify tipping points where HP retrofits become viable\n")
        
        print(f"Saved: {summary_path}")
        
        return self


def main():
    """
    Main execution function
    """
    print("RECS 2020 Heat Pump Retrofit Analysis - Retrofit Scenarios")
    print("Author: Fafa (GitHub: Fateme9977)")
    print("=" * 80)
    
    # Initialize scenario generator
    scenarios = RetrofitScenarios(output_dir='../recs_output')
    
    scenarios.define_retrofit_measures() \
             .define_heat_pump_options() \
             .define_fuel_prices_and_emissions() \
             .create_table5_retrofit_assumptions()
    
    # Example calculation for a typical home
    example_home = {
        'sqft': 2000,
        'hdd': 5000,
        'baseline_intensity': 0.020,  # BTU/sqft/HDD
    }
    
    scenarios.calculate_scenario_costs_emissions(example_home) \
             .create_scenario_summary()
    
    print("\n" + "=" * 80)
    print("✓ Retrofit scenario definitions completed successfully!")
    print("\nOutputs:")
    print("  - Table 5a: recs_output/tables/table5a_retrofit_measures.csv")
    print("  - Table 5b: recs_output/tables/table5b_heat_pump_options.csv")
    print("  - Table 5c: recs_output/tables/table5c_fuel_prices_emissions.csv")
    print("  - Combined: recs_output/tables/table5_combined_assumptions.txt")
    print("\nNext step: Run 06_nsga2_optimization.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
