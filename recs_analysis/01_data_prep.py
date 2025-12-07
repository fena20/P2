"""
Step 1 - Data Preparation for RECS 2020 Heat Pump Retrofit Analysis

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology

This script:
1. Loads RECS 2020 public microdata
2. Filters for gas-heated homes
3. Constructs thermal intensity metric
4. Engineers features for ML modeling
5. Defines envelope efficiency classes
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RECS2020DataPrep:
    """
    Data preparation pipeline for RECS 2020 microdata analysis
    """
    
    def __init__(self, data_dir='../data', output_dir='../recs_output'):
        """
        Initialize data preparation pipeline
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing RECS 2020 files
        output_dir : str
            Path to output directory for processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df_raw = None
        self.df_clean = None
        
    def load_microdata(self, filename_pattern='recs2020_public'):
        """
        Load RECS 2020 public microdata CSV
        
        Parameters
        ----------
        filename_pattern : str
            Pattern to match RECS microdata file
        """
        print("=" * 80)
        print("STEP 1: Loading RECS 2020 Microdata")
        print("=" * 80)
        
        # Find the microdata file
        csv_files = list(self.data_dir.glob(f"{filename_pattern}*.csv"))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No RECS microdata file found matching '{filename_pattern}*.csv' in {self.data_dir}\n"
                f"Please download RECS 2020 microdata from:\n"
                f"https://github.com/Fateme9977/DataR/tree/main/data\n"
                f"and place it in the data/ directory."
            )
        
        # Use the first match
        microdata_file = csv_files[0]
        print(f"Loading: {microdata_file}")
        
        self.df_raw = pd.read_csv(microdata_file, low_memory=False)
        print(f"Loaded {len(self.df_raw):,} households")
        print(f"Variables: {len(self.df_raw.columns)}")
        
        return self
    
    def filter_gas_heated_homes(self):
        """
        Filter for homes with natural gas as main heating fuel
        and required data availability
        """
        print("\n" + "=" * 80)
        print("STEP 2: Filtering for Gas-Heated Homes")
        print("=" * 80)
        
        df = self.df_raw.copy()
        initial_count = len(df)
        
        # Key variables needed (adjust based on actual RECS 2020 codebook)
        required_vars = {
            'FUELHEAT': 'Main heating fuel',
            'HDD65': 'Heating degree days',
            'TOTSQFT_EN': 'Total square footage',
            'NWEIGHT': 'Final sample weight',
        }
        
        # Check which variables exist
        missing_vars = [var for var in required_vars.keys() if var not in df.columns]
        if missing_vars:
            print(f"\nWARNING: The following expected variables are missing: {missing_vars}")
            print("This script will need to be adjusted based on the actual RECS 2020 codebook.")
            print("Please refer to: 'RECS 2020 Codebook for Public File - v7.xlsx'")
        
        # Filter for natural gas heating (FUELHEAT code varies by RECS version)
        # Typically: 5 = Natural gas from underground pipes
        if 'FUELHEAT' in df.columns:
            gas_codes = [5]  # Adjust based on actual codebook
            df = df[df['FUELHEAT'].isin(gas_codes)]
            print(f"Homes with natural gas heating: {len(df):,} ({len(df)/initial_count*100:.1f}%)")
        else:
            print("WARNING: FUELHEAT variable not found. Skipping fuel filter.")
        
        # Filter for non-missing heated area
        if 'TOTSQFT_EN' in df.columns:
            df = df[df['TOTSQFT_EN'].notna() & (df['TOTSQFT_EN'] > 0)]
            print(f"After removing missing/zero floor area: {len(df):,}")
        
        # Filter for non-missing HDD
        if 'HDD65' in df.columns:
            df = df[df['HDD65'].notna() & (df['HDD65'] > 0)]
            print(f"After removing missing/zero HDD: {len(df):,}")
        
        # Filter for non-missing weight
        if 'NWEIGHT' in df.columns:
            df = df[df['NWEIGHT'].notna() & (df['NWEIGHT'] > 0)]
            print(f"After removing missing/zero weight: {len(df):,}")
        
        print(f"\nFinal filtered sample: {len(df):,} households ({len(df)/initial_count*100:.1f}% of original)")
        
        self.df_clean = df
        return self
    
    def construct_thermal_intensity(self):
        """
        Construct heating thermal intensity metric:
        I = E_heat / (A_heated × HDD65)
        
        Where:
        - E_heat: Annual heating energy (BTU or kWh)
        - A_heated: Heated floor area (sqft)
        - HDD65: Heating degree days base 65°F
        """
        print("\n" + "=" * 80)
        print("STEP 3: Constructing Thermal Intensity Metric")
        print("=" * 80)
        
        df = self.df_clean.copy()
        
        # Heating energy variables in RECS (adjust based on actual codebook)
        # Common options: BTUNG (natural gas BTU), DOLLNG (natural gas expenditure)
        # May need to construct from total consumption
        
        heating_energy_vars = ['BTUNG', 'BTUHEAT', 'TOTALBTU']  # Check actual names
        heating_var_found = None
        
        for var in heating_energy_vars:
            if var in df.columns:
                heating_var_found = var
                break
        
        if heating_var_found:
            print(f"Using heating energy variable: {heating_var_found}")
            df['E_heat_BTU'] = df[heating_var_found]
        else:
            print("WARNING: No direct heating energy variable found.")
            print("Attempting to estimate from natural gas consumption...")
            
            # Estimate: assume most natural gas goes to heating in gas-heated homes
            # This is a simplification - see RECS CE methodology for proper allocation
            if 'BTUNG' in df.columns:
                df['E_heat_BTU'] = df['BTUNG'] * 0.8  # Assume 80% for heating (rough estimate)
                print("Estimated heating energy as 80% of natural gas consumption")
            else:
                # Create dummy values for demonstration
                print("WARNING: Creating placeholder heating energy values for demonstration")
                df['E_heat_BTU'] = 50e6  # 50 million BTU as placeholder
        
        # Construct thermal intensity
        if 'TOTSQFT_EN' in df.columns and 'HDD65' in df.columns:
            df['Thermal_Intensity_I'] = df['E_heat_BTU'] / (df['TOTSQFT_EN'] * df['HDD65'])
            
            # Remove infinite/invalid values
            df = df[np.isfinite(df['Thermal_Intensity_I'])]
            df = df[df['Thermal_Intensity_I'] > 0]
            
            print(f"\nThermal Intensity Statistics (BTU/sqft/HDD):")
            print(df['Thermal_Intensity_I'].describe())
            
            # Flag outliers (e.g., > 99th percentile or < 1st percentile)
            p99 = df['Thermal_Intensity_I'].quantile(0.99)
            p01 = df['Thermal_Intensity_I'].quantile(0.01)
            outliers = (df['Thermal_Intensity_I'] > p99) | (df['Thermal_Intensity_I'] < p01)
            print(f"\nOutliers (< 1st or > 99th percentile): {outliers.sum()} ({outliers.sum()/len(df)*100:.1f}%)")
            
            # Optionally remove extreme outliers
            # df = df[~outliers]
        else:
            print("ERROR: Cannot construct thermal intensity - missing required variables")
        
        self.df_clean = df
        return self
    
    def engineer_features(self):
        """
        Engineer features for ML modeling based on RECS variables
        """
        print("\n" + "=" * 80)
        print("STEP 4: Feature Engineering")
        print("=" * 80)
        
        df = self.df_clean.copy()
        
        # Building characteristics (adjust variable names based on codebook)
        feature_mapping = {
            # Year built
            'YEARMADERANGE': 'year_built_category',
            
            # Housing type
            'TYPEHUQ': 'housing_type',
            
            # Climate
            'HDD65': 'hdd65',
            'CDD65': 'cdd65',
            
            # Geography
            'REGIONC': 'census_region',
            'DIVISION': 'census_division',
            
            # Building envelope
            'DRAFTY': 'drafty',
            'WINDOWS': 'window_type',
            'ADQINSUL': 'insulation_adequacy',
            
            # Equipment
            'EQUIPM': 'main_heating_equipment',
            'EQUIPAGE': 'heating_equipment_age',
            
            # Size
            'TOTSQFT_EN': 'heated_sqft',
            'TOTROOMS': 'total_rooms',
            'BEDROOMS': 'bedrooms',
        }
        
        # Map available features
        for old_name, new_name in feature_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
                print(f"✓ Mapped {old_name} → {new_name}")
            else:
                print(f"✗ Variable not found: {old_name}")
        
        # Create derived features
        
        # Age of home (if year built available)
        if 'YEARMADERANGE' in df.columns:
            # YEARMADERANGE is typically categorical
            # Create approximate building age
            current_year = 2020
            year_ranges = {
                1: 1950,  # Before 1950
                2: 1955,  # 1950-1959
                3: 1965,  # 1960-1969
                4: 1975,  # 1970-1979
                5: 1985,  # 1980-1989
                6: 1995,  # 1990-1999
                7: 2005,  # 2000-2009
                8: 2015,  # 2010-2015
                9: 2018,  # 2016-2020
            }
            df['building_age'] = df['YEARMADERANGE'].map(year_ranges)
            df['building_age'] = current_year - df['building_age']
        
        # Climate severity categories
        if 'HDD65' in df.columns:
            df['climate_zone'] = pd.cut(df['HDD65'], 
                                       bins=[0, 2000, 4000, 6000, 10000],
                                       labels=['mild', 'moderate', 'cold', 'very_cold'])
        
        # Floor area categories
        if 'TOTSQFT_EN' in df.columns:
            df['size_category'] = pd.cut(df['TOTSQFT_EN'],
                                        bins=[0, 1000, 1500, 2000, 2500, 10000],
                                        labels=['very_small', 'small', 'medium', 'large', 'very_large'])
        
        print(f"\nTotal features available: {len(df.columns)}")
        
        self.df_clean = df
        return self
    
    def define_envelope_classes(self):
        """
        Define envelope efficiency classes (poor, medium, good)
        based on building characteristics
        """
        print("\n" + "=" * 80)
        print("STEP 5: Defining Envelope Efficiency Classes")
        print("=" * 80)
        
        df = self.df_clean.copy()
        
        # Simple classification based on available variables
        # More sophisticated methods can use multiple indicators
        
        envelope_score = 0
        
        # Factor 1: Draftiness (if available)
        if 'DRAFTY' in df.columns:
            # DRAFTY typically: 1 = Never drafty, 2 = Sometimes, 3 = Most of the time
            df['drafty_score'] = df['DRAFTY'].map({1: 2, 2: 1, 3: 0})
            df['drafty_score'] = df['drafty_score'].fillna(1)  # Default to medium
            envelope_score += df['drafty_score']
            print("✓ Using DRAFTY variable")
        
        # Factor 2: Age of home (older = worse envelope typically)
        if 'building_age' in df.columns:
            df['age_score'] = pd.cut(df['building_age'],
                                    bins=[-1, 15, 30, 200],
                                    labels=[2, 1, 0]).astype(float)
            df['age_score'] = df['age_score'].fillna(1)
            envelope_score += df['age_score']
            print("✓ Using building age")
        
        # Factor 3: Insulation adequacy (if available)
        if 'ADQINSUL' in df.columns:
            # ADQINSUL typically: 1 = Well insulated, 2 = Adequate, 3 = Poorly insulated
            df['insulation_score'] = df['ADQINSUL'].map({1: 2, 2: 1, 3: 0})
            df['insulation_score'] = df['insulation_score'].fillna(1)
            envelope_score += df['insulation_score']
            print("✓ Using insulation adequacy")
        
        # Normalize and classify
        if isinstance(envelope_score, pd.Series):
            max_score = envelope_score.max()
            df['envelope_score_normalized'] = envelope_score / max_score
            
            # Classify into three classes
            df['envelope_class'] = pd.cut(df['envelope_score_normalized'],
                                         bins=[-0.01, 0.33, 0.67, 1.01],
                                         labels=['poor', 'medium', 'good'])
        else:
            # Default to medium if no variables available
            print("WARNING: No envelope variables available. All homes classified as 'medium'.")
            df['envelope_class'] = 'medium'
        
        # Distribution of envelope classes
        print("\nEnvelope Class Distribution:")
        if 'envelope_class' in df.columns:
            class_dist = df['envelope_class'].value_counts(normalize=True).sort_index()
            for cls, pct in class_dist.items():
                print(f"  {cls:10s}: {pct*100:5.1f}%")
        
        self.df_clean = df
        return self
    
    def save_prepared_data(self, filename='recs2020_gas_heated_prepared.csv'):
        """
        Save prepared dataset
        """
        print("\n" + "=" * 80)
        print("STEP 6: Saving Prepared Data")
        print("=" * 80)
        
        output_path = self.output_dir / filename
        self.df_clean.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        print(f"Records: {len(self.df_clean):,}")
        print(f"Variables: {len(self.df_clean.columns)}")
        
        # Save data dictionary
        dict_path = self.output_dir / 'data_dictionary.txt'
        with open(dict_path, 'w') as f:
            f.write("RECS 2020 Heat Pump Retrofit Analysis - Data Dictionary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total records: {len(self.df_clean):,}\n")
            f.write(f"Total variables: {len(self.df_clean.columns)}\n\n")
            f.write("Key Variables:\n")
            f.write("-" * 80 + "\n")
            
            key_vars = ['Thermal_Intensity_I', 'heated_sqft', 'hdd65', 
                       'envelope_class', 'climate_zone', 'building_age']
            
            for var in key_vars:
                if var in self.df_clean.columns:
                    f.write(f"\n{var}:\n")
                    f.write(f"  Type: {self.df_clean[var].dtype}\n")
                    if pd.api.types.is_numeric_dtype(self.df_clean[var]):
                        f.write(f"  Mean: {self.df_clean[var].mean():.2f}\n")
                        f.write(f"  Std: {self.df_clean[var].std():.2f}\n")
                        f.write(f"  Min: {self.df_clean[var].min():.2f}\n")
                        f.write(f"  Max: {self.df_clean[var].max():.2f}\n")
                    else:
                        f.write(f"  Unique values: {self.df_clean[var].nunique()}\n")
        
        print(f"Data dictionary saved to: {dict_path}")
        
        return output_path
    
    def get_summary_statistics(self):
        """
        Generate summary statistics for reporting
        """
        print("\n" + "=" * 80)
        print("DATA PREPARATION SUMMARY")
        print("=" * 80)
        
        df = self.df_clean
        
        print(f"\nSample Size: {len(df):,} households")
        
        if 'NWEIGHT' in df.columns:
            weighted_population = df['NWEIGHT'].sum()
            print(f"Weighted Population: {weighted_population/1e6:.1f} million households")
        
        print("\nKey Metrics:")
        
        if 'Thermal_Intensity_I' in df.columns:
            print(f"  Thermal Intensity (BTU/sqft/HDD):")
            print(f"    Mean:   {df['Thermal_Intensity_I'].mean():.4f}")
            print(f"    Median: {df['Thermal_Intensity_I'].median():.4f}")
            print(f"    Std:    {df['Thermal_Intensity_I'].std():.4f}")
        
        if 'heated_sqft' in df.columns:
            print(f"  Heated Floor Area (sqft):")
            print(f"    Mean:   {df['heated_sqft'].mean():.0f}")
            print(f"    Median: {df['heated_sqft'].median():.0f}")
        
        if 'hdd65' in df.columns:
            print(f"  Heating Degree Days:")
            print(f"    Mean:   {df['hdd65'].mean():.0f}")
            print(f"    Median: {df['hdd65'].median():.0f}")
        
        print("\n" + "=" * 80)


def main():
    """
    Main execution function
    """
    print("RECS 2020 Heat Pump Retrofit Analysis - Data Preparation")
    print("Author: Fafa (GitHub: Fateme9977)")
    print("=" * 80)
    
    # Initialize pipeline
    prep = RECS2020DataPrep(data_dir='../data', output_dir='../recs_output')
    
    try:
        # Execute pipeline
        prep.load_microdata() \
            .filter_gas_heated_homes() \
            .construct_thermal_intensity() \
            .engineer_features() \
            .define_envelope_classes() \
            .save_prepared_data()
        
        # Print summary
        prep.get_summary_statistics()
        
        print("\n✓ Data preparation completed successfully!")
        print("\nNext step: Run 02_descriptive_validation.py")
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nPlease ensure RECS 2020 microdata is downloaded and placed in the data/ directory.")
        print("See README for data sources.")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
