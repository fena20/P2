"""
Step 1: Data Preparation
========================

Goal: Build a clean analytical dataset of gas-heated homes and define thermal intensity metric.

This script:
1. Loads RECS 2020 microdata from data/
2. Filters to gas-heated dwellings
3. Constructs heating energy variables
4. Calculates thermal intensity: I = E_heat / (A_heated × HDD65)
5. Performs feature engineering
6. Defines envelope classes (poor, medium, good)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# File paths (adjust filenames as needed)
MICRODATA_FILE = DATA_DIR / "recs2020_public_v3.csv"  # Adjust to actual filename
CODEBOOK_FILE = DATA_DIR / "RECS 2020 Codebook for Public File - v7.xlsx"

# Key RECS variables (from codebook)
# Adjust these based on actual RECS 2020 variable names
VARIABLES = {
    'fuel_heat': 'FUELHEAT',  # Main heating fuel
    'heated_area': 'TOTSQFT_EN',  # Heated floor area
    'hdd65': 'HDD65',  # Heating degree days base 65°F
    'year_built': 'YEARMADE',  # Year built
    'housing_type': 'TYPEHUQ',  # Housing type
    'drafty': 'DRAFTY',  # Draftiness indicator
    'equipment': 'EQUIPM',  # Heating equipment type
    'equip_age': 'EQUIPAGE',  # Equipment age
    'region': 'REGIONC',  # Census region
    'division': 'DIVISION',  # Census division
    'heating_btu': 'BTUEL',  # Heating energy (BTU) - adjust as needed
    'weight': 'NWEIGHT',  # Sample weight
}


def load_microdata(filepath):
    """
    Load RECS 2020 microdata CSV.
    
    Parameters
    ----------
    filepath : Path
        Path to RECS 2020 public microdata CSV
        
    Returns
    -------
    pd.DataFrame
        Raw microdata
    """
    print(f"Loading microdata from {filepath}...")
    if not filepath.exists():
        raise FileNotFoundError(
            f"Microdata file not found: {filepath}\n"
            f"Please download from: https://github.com/Fateme9977/DataR/tree/main/data"
        )
    
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} records with {len(df.columns)} variables")
    return df


def filter_gas_heated(df, fuel_var='FUELHEAT'):
    """
    Filter to natural gas-heated dwellings.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw microdata
    fuel_var : str
        Variable name for main heating fuel
        
    Returns
    -------
    pd.DataFrame
        Filtered to gas-heated homes
    """
    print(f"\nFiltering to gas-heated dwellings...")
    print(f"Fuel distribution before filtering:")
    print(df[fuel_var].value_counts())
    
    # Natural gas code (typically 1, but check codebook)
    GAS_CODE = 1  # Adjust based on RECS 2020 codebook
    
    df_gas = df[df[fuel_var] == GAS_CODE].copy()
    print(f"\nFiltered to {len(df_gas):,} gas-heated dwellings")
    return df_gas


def validate_required_fields(df, required_vars):
    """
    Check for missing required variables and drop rows with missing critical fields.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    required_vars : dict
        Dictionary mapping variable names to their RECS codes
        
    Returns
    -------
    pd.DataFrame
        Dataframe with valid rows only
    """
    print("\nValidating required fields...")
    missing_vars = []
    for name, code in required_vars.items():
        if code not in df.columns:
            missing_vars.append(f"{name} ({code})")
    
    if missing_vars:
        print(f"WARNING: Missing variables: {', '.join(missing_vars)}")
        print("Please check RECS 2020 codebook for correct variable names")
    
    # Critical fields that cannot be missing
    critical = [required_vars['heated_area'], required_vars['hdd65']]
    critical = [v for v in critical if v in df.columns]
    
    if critical:
        initial_len = len(df)
        df = df.dropna(subset=critical)
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"Dropped {dropped:,} rows with missing critical fields")
    
    return df


def calculate_thermal_intensity(df, energy_var, area_var, hdd_var):
    """
    Calculate thermal intensity: I = E_heat / (A_heated × HDD65)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with heating energy, area, and HDD
    energy_var : str
        Variable name for annual heating energy (BTU or kWh)
    area_var : str
        Variable name for heated floor area
    hdd_var : str
        Variable name for HDD65
        
    Returns
    -------
    pd.Series
        Thermal intensity values
    """
    print("\nCalculating thermal intensity...")
    
    # Ensure positive values
    energy = df[energy_var].clip(lower=1)
    area = df[area_var].clip(lower=1)
    hdd = df[hdd_var].clip(lower=1)
    
    # I = E_heat / (A_heated × HDD65)
    intensity = energy / (area * hdd)
    
    # Remove extreme outliers (top/bottom 0.1%)
    q_low = intensity.quantile(0.001)
    q_high = intensity.quantile(0.999)
    intensity = intensity.clip(lower=q_low, upper=q_high)
    
    print(f"Thermal intensity range: {intensity.min():.2f} to {intensity.max():.2f}")
    print(f"Mean: {intensity.mean():.2f}, Median: {intensity.median():.2f}")
    
    return intensity


def create_envelope_classes(df, year_var, drafty_var, housing_var):
    """
    Define envelope efficiency classes: poor, medium, good
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with building characteristics
    year_var : str
        Variable name for year built
    drafty_var : str
        Variable name for draftiness indicator
    housing_var : str
        Variable name for housing type
        
    Returns
    -------
    pd.Series
        Envelope class labels ('poor', 'medium', 'good')
    """
    print("\nCreating envelope efficiency classes...")
    
    # Create copy for manipulation
    df_work = df.copy()
    
    # Age of building (2020 - year built)
    if year_var in df.columns:
        df_work['age'] = 2020 - df_work[year_var]
    else:
        df_work['age'] = np.nan
    
    # Scoring system (simplified - refine based on literature)
    score = 0
    
    # Age component (older = worse)
    if 'age' in df_work.columns:
        age_score = pd.cut(df_work['age'], 
                          bins=[0, 20, 50, 200],
                          labels=[2, 1, 0])
        score += age_score.astype(float).fillna(1)
    
    # Draftiness component
    if drafty_var in df_work.columns:
        # Assuming 1 = drafty, 0 = not drafty (check codebook)
        drafty_score = df_work[drafty_var].fillna(0)
        score += drafty_score * 1.5
    
    # Classify
    envelope_class = pd.cut(score,
                           bins=[-np.inf, 1, 2.5, np.inf],
                           labels=['good', 'medium', 'poor'])
    
    envelope_class = envelope_class.astype(str)
    envelope_class = envelope_class.replace('nan', 'medium')  # Default
    
    print("Envelope class distribution:")
    print(envelope_class.value_counts())
    
    return envelope_class


def engineer_features(df, variables):
    """
    Create additional features for modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Base dataframe
    variables : dict
        Dictionary of variable mappings
        
    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """
    print("\nEngineering features...")
    
    df_feat = df.copy()
    
    # Age of building
    if variables['year_built'] in df.columns:
        df_feat['building_age'] = 2020 - df_feat[variables['year_built']]
    
    # Log transformations for skewed variables
    if variables['heated_area'] in df.columns:
        df_feat['log_area'] = np.log1p(df_feat[variables['heated_area']])
    
    if variables['hdd65'] in df.columns:
        df_feat['log_hdd'] = np.log1p(df_feat[variables['hdd65']])
    
    # Climate zone categories (based on HDD)
    if variables['hdd65'] in df.columns:
        df_feat['climate_zone'] = pd.cut(df_feat[variables['hdd65']],
                                         bins=[0, 2000, 4000, 6000, np.inf],
                                         labels=['mild', 'moderate', 'cold', 'very_cold'])
    
    print(f"Created {len(df_feat.columns) - len(df.columns)} new features")
    
    return df_feat


def main():
    """Main data preparation pipeline."""
    print("=" * 70)
    print("RECS 2020 Heat Pump Retrofit Project - Data Preparation")
    print("=" * 70)
    
    # Load microdata
    df = load_microdata(MICRODATA_FILE)
    
    # Filter to gas-heated
    df_gas = filter_gas_heated(df, VARIABLES['fuel_heat'])
    
    # Validate required fields
    df_gas = validate_required_fields(df_gas, VARIABLES)
    
    # Calculate thermal intensity
    if all(v in df_gas.columns for v in [VARIABLES['heating_btu'], 
                                          VARIABLES['heated_area'],
                                          VARIABLES['hdd65']]):
        df_gas['thermal_intensity'] = calculate_thermal_intensity(
            df_gas,
            VARIABLES['heating_btu'],
            VARIABLES['heated_area'],
            VARIABLES['hdd65']
        )
    else:
        print("\nWARNING: Cannot calculate thermal intensity - missing variables")
        print("Please check variable names in VARIABLES dictionary")
        df_gas['thermal_intensity'] = np.nan
    
    # Create envelope classes
    df_gas['envelope_class'] = create_envelope_classes(
        df_gas,
        VARIABLES['year_built'],
        VARIABLES['drafty'],
        VARIABLES['housing_type']
    )
    
    # Engineer features
    df_gas = engineer_features(df_gas, VARIABLES)
    
    # Save cleaned dataset
    output_file = OUTPUT_DIR / "recs2020_gas_heated_cleaned.csv"
    df_gas.to_csv(output_file, index=False)
    print(f"\n✓ Saved cleaned dataset to {output_file}")
    print(f"  Final dataset: {len(df_gas):,} rows × {len(df_gas.columns)} columns")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    if 'thermal_intensity' in df_gas.columns and df_gas['thermal_intensity'].notna().any():
        print(f"\nThermal Intensity (I):")
        print(df_gas['thermal_intensity'].describe())
    
    if 'envelope_class' in df_gas.columns:
        print(f"\nEnvelope Class Distribution:")
        print(df_gas['envelope_class'].value_counts())
    
    if VARIABLES['division'] in df_gas.columns:
        print(f"\nDivision Distribution:")
        print(df_gas[VARIABLES['division']].value_counts().sort_index())
    
    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
