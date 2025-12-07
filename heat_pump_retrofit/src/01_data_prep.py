"""
01_data_prep.py
================
RECS 2020 Data Preparation for Heat Pump Retrofit Analysis

This module loads and cleans the RECS 2020 public-use microdata,
filters for gas-heated homes, constructs the thermal intensity metric,
and creates envelope efficiency classes.

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging

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

# Constants for analysis
BTU_PER_THERM = 100000  # 1 therm = 100,000 BTU
BTU_PER_KWH = 3412      # 1 kWh = 3,412 BTU
BTU_PER_GAL_PROPANE = 91500
BTU_PER_GAL_FUEL_OIL = 138500

# RECS 2020 fuel codes for natural gas heating
GAS_FUEL_CODES = [1]  # FUELHEAT=1 is natural gas


def find_microdata_file(data_dir: Path) -> Path:
    """Find the RECS 2020 microdata CSV file in the data directory."""
    patterns = ["recs2020_public*.csv", "RECS2020*.csv"]
    
    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            # Return the most recent version
            return sorted(matches)[-1]
    
    raise FileNotFoundError(
        f"Could not find RECS 2020 microdata CSV in {data_dir}. "
        "Expected filename like 'recs2020_public_v7.csv'"
    )


def load_raw_microdata(filepath: Path) -> pd.DataFrame:
    """
    Load the raw RECS 2020 microdata.
    
    Parameters
    ----------
    filepath : Path
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw microdata with all columns
    """
    logger.info(f"Loading microdata from {filepath}")
    
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df):,} households with {len(df.columns)} variables")
    
    return df


def select_key_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select key variables needed for analysis.
    
    Categories:
    - Identification and weights
    - Geographic/climate
    - Building characteristics
    - Heating system
    - Energy consumption
    - Energy expenditure
    """
    
    # Define variable groups
    id_weight_vars = ['DOEID', 'NWEIGHT']
    
    # Add replicate weights for RSE calculation
    replicate_weights = [f'NWEIGHT{i}' for i in range(1, 61)]
    
    geographic_vars = [
        'REGIONC',      # Census region (4 categories)
        'DIVISION',     # Census division (9 categories + 10 for territories)
        'UATYP10',      # Urban/rural type
        'CLIMATE_REGION_PUB',  # BA climate zone
        'HDD65',        # Heating degree days (base 65°F)
        'CDD65',        # Cooling degree days (base 65°F)
        'HDD30YR',      # 30-year normal HDD
        'CDD30YR',      # 30-year normal CDD
    ]
    
    building_vars = [
        'TYPEHUQ',      # Housing unit type
        'YEARMADERANGE', # Year built range
        'TOTSQFT_EN',   # Total square footage (conditioned)
        'TOTHSQFT',     # Heated square footage
        'TOTCSQFT',     # Cooled square footage
        'STORIES',      # Number of stories
        'NCOMBATH',     # Number of full bathrooms
        'NHAFBATH',     # Number of half bathrooms
        'BEDROOMS',     # Number of bedrooms
        'TOTROOMS',     # Total rooms
        'GARESSION',    # Garage/carport attached
        'PRKGPLC1',     # Garage attachment type
        'WINDOWS',      # Number of windows
        'WINFRAME',     # Window frame material
        'TYPEGLASS',    # Window glass type
        'ADQINSUL',     # Adequate insulation
        'DRAFTY',       # How drafty
        'HIGHCEIL',     # High ceilings
        'NUMFLRS',      # Number of floors
        'CELLAR',       # Basement type
        'CRAWL',        # Crawl space
        'CONCRETE',     # Slab foundation
        'BASEFIN',      # Basement finished
        'ATTIC',        # Attic type
        'ATTICFIN',     # Attic finished
        'WALLTYPE',     # Exterior wall material
        'ROOFTYPE',     # Roof material
    ]
    
    heating_vars = [
        'FUELHEAT',     # Main heating fuel
        'EQUIPM',       # Main heating equipment type
        'EQUIPMUSE',    # Main equipment used (yes/no)
        'EQUIPAGE',     # Main equipment age
        'FUELAUX',      # Secondary heating fuel (if any)
        'NOHEATBROKE',  # Heating broken in last year
        'NOHEATBULK',   # No heat - couldn't afford bulk fuel
        'NOHEATDAYS',   # Days without heat
        'THERMESSION',  # Programmable thermostat
        'HEESSION',     # Home energy management system
        'SMARTTHERM',   # Smart thermostat
        'HEATCNTL',     # Heating control type
        'TEMPHOME',     # Temperature when home
        'TEMPNITE',     # Temperature at night
        'TEMPGONE',     # Temperature when away
    ]
    
    cooling_vars = [
        'COOLTYPE',     # Cooling equipment type
        'FUELCOOL',     # Cooling fuel (electricity assumed)
        'ACEQUIPM_PUB', # AC equipment type (public)
        'ACEQUIPAGE',   # AC equipment age
        'CENACHP',      # Central AC is heat pump
    ]
    
    # Energy consumption variables (annual, in original units)
    consumption_vars = [
        'BTUEL',        # Electricity consumption (BTU)
        'BTUNG',        # Natural gas consumption (BTU)
        'BTULP',        # Propane/LPG consumption (BTU)
        'BTUFO',        # Fuel oil consumption (BTU)
        'BTUWOOD',      # Wood consumption (BTU)
        'TOTALBTU',     # Total consumption (BTU)
        'KWH',          # Electricity (kWh)
        'CUFEETNG',     # Natural gas (cubic feet)
        'GALLONLP',     # Propane (gallons)
        'GALLONFO',     # Fuel oil (gallons)
        'CORDS',        # Wood (cords)
        'PELESSION',    # Wood pellets (tons)
    ]
    
    # End-use consumption (where available)
    enduse_vars = [
        'BTUSPH',       # Space heating BTU
        'DOLLARSPH',    # Space heating expenditure
        'BTUCOL',       # Space cooling BTU
        'BTUWTH',       # Water heating BTU
        'BTUOTH',       # Other end uses BTU
    ]
    
    # Expenditure variables
    expenditure_vars = [
        'DOLLAREL',     # Electricity expenditure ($)
        'DOLLARNG',     # Natural gas expenditure ($)
        'DOLLARLP',     # Propane expenditure ($)
        'DOLLARFO',     # Fuel oil expenditure ($)
        'DOLLARWOOD',   # Wood expenditure ($)
        'TOTALDOL',     # Total expenditure ($)
    ]
    
    # Income and demographic
    demographic_vars = [
        'NHSLDMEM',     # Number of household members
        'HHAGE',        # Age of householder
        'HHSEX',        # Sex of householder
        'EDUCATION',    # Education level
        'EMPLOESSION',  # Employment status
        'MONESSION',    # Income
        'ELPAY',        # Who pays electric bill
        'NGPAY',        # Who pays gas bill
        'PAYHELP',      # Received energy assistance
        'SCALEE',       # LIHEAP electricity
        'SCALEG',       # LIHEAP gas
        'SCALEEB',      # LIHEAP other
    ]
    
    # Combine all variable lists
    all_vars = (
        id_weight_vars + 
        geographic_vars + 
        building_vars + 
        heating_vars + 
        cooling_vars + 
        consumption_vars + 
        enduse_vars + 
        expenditure_vars + 
        demographic_vars +
        replicate_weights
    )
    
    # Filter to only variables that exist in the dataframe
    available_vars = [v for v in all_vars if v in df.columns]
    missing_vars = [v for v in all_vars if v not in df.columns]
    
    if missing_vars:
        logger.warning(f"Missing {len(missing_vars)} variables: {missing_vars[:10]}...")
    
    logger.info(f"Selected {len(available_vars)} variables for analysis")
    
    return df[available_vars].copy()


def filter_gas_heated_homes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to include only gas-heated homes suitable for retrofit analysis.
    
    Criteria:
    - Main heating fuel is natural gas (FUELHEAT = 1)
    - Has valid heated floor area
    - Has valid HDD65
    - Has non-zero heating energy consumption
    """
    logger.info("Filtering for gas-heated homes...")
    
    initial_count = len(df)
    
    # Filter for natural gas as main heating fuel
    # FUELHEAT: 1=Natural gas, 2=Propane, 3=Fuel oil, 4=Electricity, 5=Wood, 21=Other
    if 'FUELHEAT' in df.columns:
        mask_gas = df['FUELHEAT'].isin(GAS_FUEL_CODES)
        df = df[mask_gas].copy()
        logger.info(f"After gas fuel filter: {len(df):,} households ({len(df)/initial_count*100:.1f}%)")
    else:
        logger.warning("FUELHEAT column not found - skipping fuel filter")
    
    # Filter for valid heated floor area
    if 'TOTHSQFT' in df.columns:
        mask_sqft = (df['TOTHSQFT'] > 0) & (df['TOTHSQFT'].notna())
        df = df[mask_sqft].copy()
        logger.info(f"After heated sqft filter: {len(df):,} households")
    elif 'TOTSQFT_EN' in df.columns:
        mask_sqft = (df['TOTSQFT_EN'] > 0) & (df['TOTSQFT_EN'].notna())
        df = df[mask_sqft].copy()
        logger.info(f"After total sqft filter: {len(df):,} households")
    
    # Filter for valid HDD65
    if 'HDD65' in df.columns:
        mask_hdd = (df['HDD65'] > 0) & (df['HDD65'].notna())
        df = df[mask_hdd].copy()
        logger.info(f"After HDD filter: {len(df):,} households")
    
    # Filter for non-zero heating consumption
    if 'BTUSPH' in df.columns:
        mask_btu = (df['BTUSPH'] > 0) & (df['BTUSPH'].notna())
        df = df[mask_btu].copy()
        logger.info(f"After heating consumption filter: {len(df):,} households")
    elif 'BTUNG' in df.columns:
        # Use gas consumption as proxy
        mask_btu = (df['BTUNG'] > 0) & (df['BTUNG'].notna())
        df = df[mask_btu].copy()
        logger.info(f"After gas consumption filter: {len(df):,} households")
    
    final_count = len(df)
    logger.info(f"Final sample: {final_count:,} gas-heated households ({final_count/initial_count*100:.1f}% of total)")
    
    return df


def compute_heating_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute heating energy consumption (BTU) from available variables.
    
    If BTUSPH (space heating BTU) is available, use directly.
    Otherwise, estimate from gas consumption.
    """
    logger.info("Computing heating energy...")
    
    if 'BTUSPH' in df.columns and df['BTUSPH'].notna().any():
        df['E_heat_btu'] = df['BTUSPH'].fillna(0)
        logger.info("Using BTUSPH (space heating BTU) directly")
    elif 'BTUNG' in df.columns:
        # For gas-heated homes, approximate space heating as fraction of gas use
        # Typical residential gas use: ~60-80% for space heating in cold climates
        # Using a simplified approach: assume 70% of gas consumption is for heating
        df['E_heat_btu'] = df['BTUNG'].fillna(0) * 0.70
        logger.warning("BTUSPH not available - estimating from BTUNG (70% assumption)")
    else:
        logger.error("Cannot compute heating energy - neither BTUSPH nor BTUNG available")
        df['E_heat_btu'] = np.nan
    
    # Convert to other units for convenience
    df['E_heat_therm'] = df['E_heat_btu'] / BTU_PER_THERM
    df['E_heat_kwh_equiv'] = df['E_heat_btu'] / BTU_PER_KWH
    
    return df


def compute_thermal_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute thermal intensity metric: I = E_heat / (A_heated × HDD65)
    
    Units: BTU / (sqft × degree-day)
    
    This metric normalizes heating consumption by:
    - Heated floor area (building size)
    - Heating degree days (climate severity)
    """
    logger.info("Computing thermal intensity...")
    
    # Use heated square footage if available, otherwise total conditioned
    if 'TOTHSQFT' in df.columns:
        area_col = 'TOTHSQFT'
    else:
        area_col = 'TOTSQFT_EN'
    
    # Compute thermal intensity
    df['A_heated'] = df[area_col]
    
    # Avoid division by zero
    valid_mask = (df['A_heated'] > 0) & (df['HDD65'] > 0)
    
    df['Thermal_Intensity_I'] = np.where(
        valid_mask,
        df['E_heat_btu'] / (df['A_heated'] * df['HDD65']),
        np.nan
    )
    
    # Log statistics
    intensity_stats = df['Thermal_Intensity_I'].describe()
    logger.info(f"Thermal intensity (BTU/sqft/HDD):")
    logger.info(f"  Mean: {intensity_stats['mean']:.2f}")
    logger.info(f"  Median: {intensity_stats['50%']:.2f}")
    logger.info(f"  Std: {intensity_stats['std']:.2f}")
    
    return df


def create_envelope_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create envelope efficiency classes based on building characteristics.
    
    Classification is based on:
    - Year built (YEARMADERANGE)
    - Draftiness (DRAFTY)
    - Insulation adequacy (ADQINSUL)
    - Window quality (TYPEGLASS)
    
    Classes: 'poor', 'medium', 'good'
    """
    logger.info("Creating envelope efficiency classes...")
    
    # Initialize envelope score (higher = better)
    df['envelope_score'] = 0
    
    # Year built scoring
    # YEARMADERANGE: 1=Before 1950, 2=1950-1959, ..., 8=2010-2015, 9=2016-2020
    if 'YEARMADERANGE' in df.columns:
        year_score = df['YEARMADERANGE'].map({
            1: 0,   # Before 1950 - poor
            2: 1,   # 1950-1959
            3: 1,   # 1960-1969
            4: 2,   # 1970-1979
            5: 2,   # 1980-1989
            6: 3,   # 1990-1999
            7: 3,   # 2000-2009
            8: 4,   # 2010-2015
            9: 4,   # 2016-2020
        }).fillna(2)
        df['envelope_score'] += year_score
    
    # Draftiness scoring
    # DRAFTY: 1=Very drafty, 2=Somewhat drafty, 3=Not drafty at all
    if 'DRAFTY' in df.columns:
        draft_score = df['DRAFTY'].map({
            1: 0,   # Very drafty
            2: 2,   # Somewhat drafty  
            3: 4,   # Not drafty
        }).fillna(2)
        df['envelope_score'] += draft_score
    
    # Insulation adequacy scoring
    # ADQINSUL: 1=Well insulated, 2=Adequately insulated, 3=Poorly insulated, 4=Not insulated
    if 'ADQINSUL' in df.columns:
        insul_score = df['ADQINSUL'].map({
            1: 4,   # Well insulated
            2: 3,   # Adequate
            3: 1,   # Poorly insulated
            4: 0,   # Not insulated
        }).fillna(2)
        df['envelope_score'] += insul_score
    
    # Window glass type scoring
    # TYPEGLASS: 1=Single-pane, 2=Double-pane, 3=Triple-pane
    if 'TYPEGLASS' in df.columns:
        glass_score = df['TYPEGLASS'].map({
            1: 0,   # Single pane
            2: 2,   # Double pane
            3: 4,   # Triple pane
        }).fillna(1)
        df['envelope_score'] += glass_score
    
    # Classify into envelope classes based on total score
    # Score range: 0-16 (4 variables × 0-4 each)
    df['envelope_class'] = pd.cut(
        df['envelope_score'],
        bins=[-1, 5, 10, 20],
        labels=['poor', 'medium', 'good']
    )
    
    # Log distribution
    class_dist = df['envelope_class'].value_counts(normalize=True) * 100
    logger.info("Envelope class distribution:")
    for cls, pct in class_dist.items():
        logger.info(f"  {cls}: {pct:.1f}%")
    
    return df


def create_climate_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create climate zone categories based on HDD65.
    
    Categories:
    - Cold: HDD65 >= 6000
    - Mixed: 3000 <= HDD65 < 6000
    - Mild: HDD65 < 3000
    """
    logger.info("Creating climate zone categories...")
    
    df['climate_zone'] = pd.cut(
        df['HDD65'],
        bins=[0, 3000, 6000, float('inf')],
        labels=['mild', 'mixed', 'cold']
    )
    
    # Also create HDD bands for more granular analysis
    df['hdd_band'] = pd.cut(
        df['HDD65'],
        bins=[0, 2000, 4000, 6000, 8000, float('inf')],
        labels=['<2000', '2000-4000', '4000-6000', '6000-8000', '>8000']
    )
    
    # Log distribution
    zone_dist = df['climate_zone'].value_counts(normalize=True) * 100
    logger.info("Climate zone distribution:")
    for zone, pct in zone_dist.items():
        logger.info(f"  {zone}: {pct:.1f}%")
    
    return df


def create_year_built_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create readable year built categories.
    """
    if 'YEARMADERANGE' in df.columns:
        df['year_built_cat'] = df['YEARMADERANGE'].map({
            1: 'Before 1950',
            2: '1950-1959',
            3: '1960-1969',
            4: '1970-1979',
            5: '1980-1989',
            6: '1990-1999',
            7: '2000-2009',
            8: '2010-2015',
            9: '2016-2020',
        })
    return df


def create_housing_type_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create readable housing type categories.
    """
    if 'TYPEHUQ' in df.columns:
        df['housing_type'] = df['TYPEHUQ'].map({
            1: 'Mobile home',
            2: 'Single-family detached',
            3: 'Single-family attached',
            4: 'Apartment (2-4 units)',
            5: 'Apartment (5+ units)',
        })
    return df


def create_division_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create readable census division labels.
    """
    if 'DIVISION' in df.columns:
        # Check if DIVISION is already string (newer RECS format)
        if df['DIVISION'].dtype == 'object':
            df['division_name'] = df['DIVISION']
        else:
            df['division_name'] = df['DIVISION'].map({
                1: 'New England',
                2: 'Middle Atlantic',
                3: 'East North Central',
                4: 'West North Central',
                5: 'South Atlantic',
                6: 'East South Central',
                7: 'West South Central',
                8: 'Mountain',
                9: 'Pacific',
                10: 'US Territories',
            })
    return df


def create_region_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create readable census region labels.
    """
    if 'REGIONC' in df.columns:
        # Check if REGIONC is already string (newer RECS format)
        if df['REGIONC'].dtype == 'object':
            df['region_name'] = df['REGIONC']
        else:
            df['region_name'] = df['REGIONC'].map({
                1: 'Northeast',
                2: 'Midwest',
                3: 'South',
                4: 'West',
            })
    return df


def remove_outliers(df: pd.DataFrame, column: str = 'Thermal_Intensity_I', 
                    method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    column : str
        Column to use for outlier detection
    method : str
        'iqr' for interquartile range, 'zscore' for z-score
    threshold : float
        Threshold for outlier detection (IQR multiplier or z-score)
    """
    logger.info(f"Removing outliers using {method} method...")
    
    initial_count = len(df)
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        mask = (df[column] >= lower) & (df[column] <= upper)
        df = df[mask].copy()
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        mask = np.abs((df[column] - mean) / std) <= threshold
        df = df[mask].copy()
    
    removed = initial_count - len(df)
    logger.info(f"Removed {removed:,} outliers ({removed/initial_count*100:.2f}%)")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features for modeling.
    """
    logger.info("Engineering features...")
    
    # Building age (approximate)
    year_mid = {
        1: 1940, 2: 1955, 3: 1965, 4: 1975,
        5: 1985, 6: 1995, 7: 2005, 8: 2013, 9: 2018
    }
    if 'YEARMADERANGE' in df.columns:
        df['building_age'] = 2020 - df['YEARMADERANGE'].map(year_mid)
    
    # Equipment age
    if 'EQUIPAGE' in df.columns:
        # EQUIPAGE categories need mapping
        equip_age_mid = {
            1: 1, 2: 4, 3: 8, 4: 13, 5: 18, 42: 25
        }
        df['heating_equip_age'] = df['EQUIPAGE'].map(equip_age_mid)
    
    # Log-transformed floor area
    if 'A_heated' in df.columns:
        df['log_sqft'] = np.log1p(df['A_heated'])
    
    # Interaction features
    if 'HDD65' in df.columns and 'A_heated' in df.columns:
        df['hdd_sqft_interaction'] = df['HDD65'] * df['A_heated'] / 1000000
    
    # Heating intensity without climate normalization
    if 'E_heat_btu' in df.columns and 'A_heated' in df.columns:
        df['heating_per_sqft'] = df['E_heat_btu'] / df['A_heated']
    
    return df


def prepare_analysis_dataset(save_intermediate: bool = True) -> pd.DataFrame:
    """
    Main function to prepare the analysis dataset.
    
    Steps:
    1. Load raw microdata
    2. Select key variables
    3. Filter for gas-heated homes
    4. Compute heating energy
    5. Compute thermal intensity
    6. Create envelope classes
    7. Create climate zones
    8. Engineer features
    9. Remove outliers
    
    Returns
    -------
    pd.DataFrame
        Cleaned, processed dataset ready for analysis
    """
    logger.info("=" * 60)
    logger.info("RECS 2020 Data Preparation Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load data
    microdata_path = find_microdata_file(DATA_DIR)
    df = load_raw_microdata(microdata_path)
    
    # Step 2: Select variables
    df = select_key_variables(df)
    
    if save_intermediate:
        out_path = OUTPUT_DIR / "01_selected_variables.csv"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved selected variables to {out_path}")
    
    # Step 3: Filter for gas-heated homes
    df = filter_gas_heated_homes(df)
    
    # Step 4: Compute heating energy
    df = compute_heating_energy(df)
    
    # Step 5: Compute thermal intensity
    df = compute_thermal_intensity(df)
    
    # Step 6: Create categorical variables
    df = create_envelope_classes(df)
    df = create_climate_zones(df)
    df = create_year_built_categories(df)
    df = create_housing_type_categories(df)
    df = create_division_labels(df)
    df = create_region_labels(df)
    
    # Step 7: Engineer features
    df = engineer_features(df)
    
    # Step 8: Remove outliers (optional - can be skipped for validation)
    df_clean = remove_outliers(df.copy(), 'Thermal_Intensity_I', method='iqr', threshold=3.0)
    
    # Save final dataset
    if save_intermediate:
        # Full dataset (with outliers)
        full_path = OUTPUT_DIR / "02_gas_heated_full.csv"
        df.to_csv(full_path, index=False)
        logger.info(f"Saved full dataset to {full_path}")
        
        # Clean dataset (outliers removed)
        clean_path = OUTPUT_DIR / "03_gas_heated_clean.csv"
        df_clean.to_csv(clean_path, index=False)
        logger.info(f"Saved clean dataset to {clean_path}")
    
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info(f"Final sample size: {len(df_clean):,} households")
    logger.info("=" * 60)
    
    return df_clean


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the prepared dataset.
    """
    summary = {
        'n_households': len(df),
        'n_weighted': df['NWEIGHT'].sum() if 'NWEIGHT' in df.columns else None,
        'thermal_intensity_mean': df['Thermal_Intensity_I'].mean(),
        'thermal_intensity_median': df['Thermal_Intensity_I'].median(),
        'thermal_intensity_std': df['Thermal_Intensity_I'].std(),
        'hdd_mean': df['HDD65'].mean(),
        'sqft_mean': df['A_heated'].mean(),
        'envelope_distribution': df['envelope_class'].value_counts().to_dict(),
        'climate_zone_distribution': df['climate_zone'].value_counts().to_dict(),
        'division_distribution': df['division_name'].value_counts().to_dict() if 'division_name' in df.columns else None,
    }
    return summary


if __name__ == "__main__":
    # Run the data preparation pipeline
    df = prepare_analysis_dataset(save_intermediate=True)
    
    # Print summary
    summary = summarize_dataset(df)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
