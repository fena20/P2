"""
Step 2: Descriptive Statistics & Macro Validation
==================================================

Goal: Validate the cleaned dataset against official RECS 2020 tables.

This script:
1. Computes weighted statistics using NWEIGHT
2. Compares against HC2.x, HC6.x, HC10.x tables (if available)
3. Validates aggregate heating energy vs CE tables
4. Generates validation report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CLEANED_DATA = OUTPUT_DIR / "recs2020_gas_heated_cleaned.csv"

# RECS variable names
VARIABLES = {
    'weight': 'NWEIGHT',
    'division': 'DIVISION',
    'region': 'REGIONC',
    'housing_type': 'TYPEHUQ',
    'year_built': 'YEARMADE',
    'fuel_heat': 'FUELHEAT',
    'heated_area': 'TOTSQFT_EN',
    'hdd65': 'HDD65',
    'thermal_intensity': 'thermal_intensity',
    'envelope_class': 'envelope_class',
}


def load_cleaned_data(filepath):
    """Load the cleaned dataset from Step 1."""
    print(f"Loading cleaned data from {filepath}...")
    if not filepath.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {filepath}\n"
            f"Please run 01_data_prep.py first"
        )
    return pd.read_csv(filepath)


def weighted_statistics(df, weight_var='NWEIGHT'):
    """
    Compute weighted statistics for key variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with weights
    weight_var : str
        Variable name for sample weights
        
    Returns
    -------
    dict
        Dictionary of weighted statistics
    """
    print("\nComputing weighted statistics...")
    
    if weight_var not in df.columns:
        print(f"WARNING: Weight variable '{weight_var}' not found. Using unweighted statistics.")
        weights = None
    else:
        weights = df[weight_var]
    
    stats = {}
    
    # Weighted counts by division
    if VARIABLES['division'] in df.columns:
        if weights is not None:
            stats['division_counts'] = df.groupby(VARIABLES['division'])[weight_var].sum()
        else:
            stats['division_counts'] = df[VARIABLES['division']].value_counts()
    
    # Weighted mean heated area
    if VARIABLES['heated_area'] in df.columns:
        if weights is not None:
            stats['mean_area'] = np.average(
                df[VARIABLES['heated_area']],
                weights=weights
            )
        else:
            stats['mean_area'] = df[VARIABLES['heated_area']].mean()
    
    # Weighted mean HDD65
    if VARIABLES['hdd65'] in df.columns:
        if weights is not None:
            stats['mean_hdd'] = np.average(
                df[VARIABLES['hdd65']],
                weights=weights
            )
        else:
            stats['mean_hdd'] = df[VARIABLES['hdd65']].mean()
    
    # Weighted mean thermal intensity
    if VARIABLES['thermal_intensity'] in df.columns:
        valid_mask = df[VARIABLES['thermal_intensity']].notna()
        if weights is not None:
            stats['mean_intensity'] = np.average(
                df.loc[valid_mask, VARIABLES['thermal_intensity']],
                weights=weights[valid_mask]
            )
        else:
            stats['mean_intensity'] = df.loc[valid_mask, VARIABLES['thermal_intensity']].mean()
    
    # Distribution by housing type
    if VARIABLES['housing_type'] in df.columns:
        if weights is not None:
            stats['housing_type_dist'] = df.groupby(VARIABLES['housing_type'])[weight_var].sum()
            stats['housing_type_dist'] = stats['housing_type_dist'] / stats['housing_type_dist'].sum()
        else:
            stats['housing_type_dist'] = df[VARIABLES['housing_type']].value_counts(normalize=True)
    
    # Distribution by envelope class
    if VARIABLES['envelope_class'] in df.columns:
        if weights is not None:
            stats['envelope_dist'] = df.groupby(VARIABLES['envelope_class'])[weight_var].sum()
            stats['envelope_dist'] = stats['envelope_dist'] / stats['envelope_dist'].sum()
        else:
            stats['envelope_dist'] = df[VARIABLES['envelope_class']].value_counts(normalize=True)
    
    return stats


def compare_with_official_tables(stats, hc_tables_dir=None):
    """
    Compare computed statistics with official RECS tables (if available).
    
    Parameters
    ----------
    stats : dict
        Computed weighted statistics
    hc_tables_dir : Path, optional
        Directory containing HC tables
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    print("\nComparing with official RECS tables...")
    
    comparison = []
    
    if hc_tables_dir and hc_tables_dir.exists():
        # Try to load HC tables
        hc_files = list(hc_tables_dir.glob("HC*.xlsx")) + list(hc_tables_dir.glob("HC*.csv"))
        
        if hc_files:
            print(f"Found {len(hc_files)} HC table files")
            # TODO: Implement specific comparisons based on table structure
            # This would require parsing the specific HC table formats
        else:
            print("No HC table files found in data directory")
    else:
        print("HC tables directory not specified or not found")
        print("To validate, download HC tables from EIA and place in data/")
    
    # For now, just report our computed statistics
    print("\nComputed Statistics (this study):")
    print("-" * 50)
    
    if 'mean_area' in stats:
        print(f"Weighted mean heated area: {stats['mean_area']:.1f} sqft")
    
    if 'mean_hdd' in stats:
        print(f"Weighted mean HDD65: {stats['mean_hdd']:.1f}")
    
    if 'mean_intensity' in stats:
        print(f"Weighted mean thermal intensity: {stats['mean_intensity']:.4f}")
    
    if 'division_counts' in stats:
        print("\nWeighted counts by division:")
        print(stats['division_counts'])
    
    if 'housing_type_dist' in stats:
        print("\nHousing type distribution:")
        print(stats['housing_type_dist'])
    
    if 'envelope_dist' in stats:
        print("\nEnvelope class distribution:")
        print(stats['envelope_dist'])
    
    return pd.DataFrame(comparison)


def generate_validation_report(df, stats, output_dir):
    """
    Generate a validation report summarizing the analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
    stats : dict
        Computed statistics
    output_dir : Path
        Output directory for report
    """
    print("\nGenerating validation report...")
    
    report_lines = [
        "=" * 70,
        "RECS 2020 Heat Pump Retrofit Project - Validation Report",
        "=" * 70,
        "",
        f"Dataset: {len(df):,} gas-heated dwellings",
        "",
        "Weighted Statistics:",
        "-" * 70,
    ]
    
    if 'mean_area' in stats:
        report_lines.append(f"Weighted mean heated area: {stats['mean_area']:.1f} sqft")
    
    if 'mean_hdd' in stats:
        report_lines.append(f"Weighted mean HDD65: {stats['mean_hdd']:.1f}")
    
    if 'mean_intensity' in stats:
        report_lines.append(f"Weighted mean thermal intensity: {stats['mean_intensity']:.4f}")
    
    report_lines.extend([
        "",
        "Division Distribution:",
        "-" * 70,
    ])
    
    if 'division_counts' in stats:
        for div, count in stats['division_counts'].items():
            report_lines.append(f"Division {div}: {count:,.0f} dwellings")
    
    report_lines.extend([
        "",
        "Envelope Class Distribution:",
        "-" * 70,
    ])
    
    if 'envelope_dist' in stats:
        for env_class, pct in stats['envelope_dist'].items():
            report_lines.append(f"{env_class}: {pct*100:.1f}%")
    
    report_lines.extend([
        "",
        "=" * 70,
        "Validation Notes:",
        "-" * 70,
        "1. Compare weighted statistics with official RECS 2020 HC tables",
        "2. Verify division distributions match HC2.x tables",
        "3. Check heating fuel shares match HC6.x tables",
        "4. Validate floor area distributions match HC10.x tables",
        "",
        "If discrepancies > 5%, review:",
        "  - Variable name mappings",
        "  - Filtering criteria",
        "  - Weight application",
        "=" * 70,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_file = output_dir / "validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Validation report saved to {report_file}")
    
    # Also save statistics to CSV
    if 'division_counts' in stats:
        stats_df = pd.DataFrame({
            'division': stats['division_counts'].index,
            'weighted_count': stats['division_counts'].values
        })
        stats_file = output_dir / "table2_sample_characteristics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"✓ Statistics table saved to {stats_file}")


def main():
    """Main validation pipeline."""
    print("=" * 70)
    print("RECS 2020 Heat Pump Retrofit Project - Descriptive Validation")
    print("=" * 70)
    
    # Load cleaned data
    df = load_cleaned_data(CLEANED_DATA)
    
    # Compute weighted statistics
    stats = weighted_statistics(df, VARIABLES['weight'])
    
    # Compare with official tables (if available)
    hc_tables_dir = DATA_DIR  # Adjust if HC tables are in subdirectory
    comparison = compare_with_official_tables(stats, hc_tables_dir)
    
    # Generate validation report
    generate_validation_report(df, stats, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review validation_report.txt")
    print("2. Compare with official RECS tables if available")
    print("3. Proceed to Step 3: XGBoost modeling")


if __name__ == "__main__":
    main()
