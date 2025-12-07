"""
02_descriptive_validation.py
=============================
Descriptive Statistics and Macro-Level Validation

This module:
1. Computes weighted descriptive statistics using NWEIGHT
2. Validates against official RECS 2020 tables (HC2.x, HC6.x, HC10.x)
3. Generates tables and figures for the thesis/paper

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats

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

# Create directories
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_processed_data(use_clean: bool = True) -> pd.DataFrame:
    """Load the processed dataset from Step 1."""
    if use_clean:
        filepath = OUTPUT_DIR / "03_gas_heated_clean.csv"
    else:
        filepath = OUTPUT_DIR / "02_gas_heated_full.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. "
            "Run 01_data_prep.py first."
        )
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} households from {filepath}")
    return df


def weighted_mean(df: pd.DataFrame, var: str, weight: str = 'NWEIGHT') -> float:
    """Compute weighted mean of a variable."""
    mask = df[var].notna() & df[weight].notna()
    return np.average(df.loc[mask, var], weights=df.loc[mask, weight])


def weighted_std(df: pd.DataFrame, var: str, weight: str = 'NWEIGHT') -> float:
    """Compute weighted standard deviation."""
    mask = df[var].notna() & df[weight].notna()
    values = df.loc[mask, var]
    weights = df.loc[mask, weight]
    
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean)**2, weights=weights)
    return np.sqrt(variance)


def weighted_quantile(df: pd.DataFrame, var: str, q: float, 
                      weight: str = 'NWEIGHT') -> float:
    """Compute weighted quantile."""
    mask = df[var].notna() & df[weight].notna()
    values = df.loc[mask, var].values
    weights = df.loc[mask, weight].values
    
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    
    cumsum = np.cumsum(weights)
    cutoff = q * cumsum[-1]
    return values[cumsum >= cutoff][0]


def weighted_proportion(df: pd.DataFrame, var: str, category,
                        weight: str = 'NWEIGHT') -> float:
    """Compute weighted proportion of a category."""
    mask = df[var].notna() & df[weight].notna()
    total_weight = df.loc[mask, weight].sum()
    cat_weight = df.loc[mask & (df[var] == category), weight].sum()
    return cat_weight / total_weight


def compute_weighted_stats(df: pd.DataFrame, 
                           numeric_vars: list,
                           categorical_vars: list,
                           groupby: str = None) -> dict:
    """
    Compute comprehensive weighted statistics.
    """
    results = {}
    
    if groupby:
        groups = df[groupby].unique()
        for group in groups:
            group_df = df[df[groupby] == group]
            results[group] = _compute_stats_for_group(
                group_df, numeric_vars, categorical_vars
            )
    else:
        results['overall'] = _compute_stats_for_group(
            df, numeric_vars, categorical_vars
        )
    
    return results


def _compute_stats_for_group(df: pd.DataFrame, numeric_vars: list, 
                             categorical_vars: list) -> dict:
    """Compute stats for a single group."""
    stats = {
        'n_sample': len(df),
        'n_weighted': df['NWEIGHT'].sum() if 'NWEIGHT' in df.columns else len(df),
    }
    
    # Numeric variables
    for var in numeric_vars:
        if var in df.columns:
            stats[f'{var}_mean'] = weighted_mean(df, var)
            stats[f'{var}_std'] = weighted_std(df, var)
            stats[f'{var}_median'] = weighted_quantile(df, var, 0.5)
            stats[f'{var}_q25'] = weighted_quantile(df, var, 0.25)
            stats[f'{var}_q75'] = weighted_quantile(df, var, 0.75)
    
    # Categorical variables
    for var in categorical_vars:
        if var in df.columns:
            for cat in df[var].dropna().unique():
                stats[f'{var}_{cat}_pct'] = weighted_proportion(df, var, cat) * 100
    
    return stats


def generate_table1_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 1: Overview of variables used in analysis.
    """
    logger.info("Generating Table 1: Variable definitions")
    
    variables = [
        ('DOEID', 'ID', '-', 'Unique household identifier', 'Identification'),
        ('NWEIGHT', 'w', '-', 'Sample weight for national estimates', 'Weight'),
        ('HDD65', 'HDD₆₅', '°F-days', 'Heating degree days (base 65°F)', 'Climate'),
        ('CDD65', 'CDD₆₅', '°F-days', 'Cooling degree days (base 65°F)', 'Climate'),
        ('TOTSQFT_EN', 'A', 'sqft', 'Total conditioned floor area', 'Building'),
        ('TOTHSQFT', 'A_heat', 'sqft', 'Heated floor area', 'Building'),
        ('YEARMADERANGE', '-', 'category', 'Year built range (1-9)', 'Building'),
        ('TYPEHUQ', '-', 'category', 'Housing type (1-5)', 'Building'),
        ('DRAFTY', '-', 'ordinal', 'Draftiness level (1-3)', 'Envelope'),
        ('ADQINSUL', '-', 'ordinal', 'Insulation adequacy (1-4)', 'Envelope'),
        ('TYPEGLASS', '-', 'category', 'Window glass type (1-3)', 'Envelope'),
        ('FUELHEAT', '-', 'category', 'Main heating fuel', 'Heating'),
        ('EQUIPM', '-', 'category', 'Heating equipment type', 'Heating'),
        ('EQUIPAGE', '-', 'category', 'Heating equipment age', 'Heating'),
        ('BTUSPH', 'E_heat', 'BTU', 'Annual space heating energy', 'Consumption'),
        ('BTUNG', 'E_gas', 'BTU', 'Annual natural gas consumption', 'Consumption'),
        ('DOLLARSPH', '$_heat', 'USD', 'Annual heating expenditure', 'Expenditure'),
        ('REGIONC', '-', 'category', 'Census region (1-4)', 'Geography'),
        ('DIVISION', '-', 'category', 'Census division (1-10)', 'Geography'),
        ('Thermal_Intensity_I', 'I', 'BTU/sqft/HDD', 'Heating thermal intensity', 'Derived'),
        ('envelope_class', '-', 'category', 'Envelope efficiency class', 'Derived'),
        ('climate_zone', '-', 'category', 'Climate zone (mild/mixed/cold)', 'Derived'),
    ]
    
    table1 = pd.DataFrame(
        variables,
        columns=['Variable', 'Symbol', 'Unit', 'Description', 'Category']
    )
    
    # Save
    table1.to_csv(TABLES_DIR / "table1_variable_definitions.csv", index=False)
    table1.to_latex(TABLES_DIR / "table1_variable_definitions.tex", index=False,
                    caption="Definition, units, and categories of main variables used in the analysis.",
                    label="tab:variables")
    
    return table1


def generate_table2_sample_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 2: Weighted sample characteristics by division and envelope class.
    """
    logger.info("Generating Table 2: Sample characteristics")
    
    results = []
    
    # By census division
    if 'division_name' in df.columns:
        for division in df['division_name'].dropna().unique():
            div_df = df[df['division_name'] == division]
            
            for env_class in ['poor', 'medium', 'good']:
                subset = div_df[div_df['envelope_class'] == env_class]
                
                if len(subset) > 0:
                    row = {
                        'Division': division,
                        'Envelope Class': env_class,
                        'N (sample)': len(subset),
                        'N (weighted, millions)': subset['NWEIGHT'].sum() / 1e6,
                        'Mean HDD65': weighted_mean(subset, 'HDD65'),
                        'Mean Sqft': weighted_mean(subset, 'A_heated'),
                        'Mean Thermal Intensity': weighted_mean(subset, 'Thermal_Intensity_I'),
                    }
                    results.append(row)
    
    table2 = pd.DataFrame(results)
    
    # Format numbers
    if len(table2) > 0:
        table2['N (weighted, millions)'] = table2['N (weighted, millions)'].round(2)
        table2['Mean HDD65'] = table2['Mean HDD65'].round(0).astype(int)
        table2['Mean Sqft'] = table2['Mean Sqft'].round(0).astype(int)
        table2['Mean Thermal Intensity'] = table2['Mean Thermal Intensity'].round(2)
    
    # Save
    table2.to_csv(TABLES_DIR / "table2_sample_characteristics.csv", index=False)
    
    return table2


def generate_figure2_climate_envelope(df: pd.DataFrame):
    """
    Generate Figure 2: Climate and envelope overview.
    (a) HDD65 distribution by division
    (b) Envelope class shares
    """
    logger.info("Generating Figure 2: Climate and envelope overview")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) HDD65 by division
    ax1 = axes[0]
    if 'division_name' in df.columns:
        division_order = df.groupby('division_name')['HDD65'].mean().sort_values(ascending=False).index
        
        # Weighted boxplot approximation
        sns.boxplot(
            data=df,
            x='division_name',
            y='HDD65',
            order=division_order,
            ax=ax1,
            color='steelblue'
        )
        ax1.set_xlabel('Census Division', fontsize=12)
        ax1.set_ylabel('Heating Degree Days (HDD65)', fontsize=12)
        ax1.set_title('(a) HDD65 Distribution by Census Division', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
    
    # (b) Envelope class shares
    ax2 = axes[1]
    if 'envelope_class' in df.columns:
        # Weighted proportions
        env_shares = df.groupby('envelope_class')['NWEIGHT'].sum()
        env_shares = env_shares / env_shares.sum() * 100
        
        colors = {'poor': '#d62728', 'medium': '#ff7f0e', 'good': '#2ca02c'}
        bars = ax2.bar(
            env_shares.index, 
            env_shares.values,
            color=[colors.get(x, 'gray') for x in env_shares.index]
        )
        
        # Add percentage labels
        for bar, pct in zip(bars, env_shares.values):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=12
            )
        
        ax2.set_xlabel('Envelope Efficiency Class', fontsize=12)
        ax2.set_ylabel('Share of Gas-Heated Homes (%)', fontsize=12)
        ax2.set_title('(b) Envelope Class Distribution', fontsize=14)
        ax2.set_ylim(0, 60)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_climate_envelope.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure2_climate_envelope.pdf", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Figure 2 to {FIGURES_DIR}")


def generate_figure3_thermal_intensity_distribution(df: pd.DataFrame):
    """
    Generate Figure 3: Distribution of thermal intensity by envelope and climate.
    """
    logger.info("Generating Figure 3: Thermal intensity distribution")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) By envelope class
    ax1 = axes[0]
    sns.boxplot(
        data=df,
        x='envelope_class',
        y='Thermal_Intensity_I',
        order=['poor', 'medium', 'good'],
        palette={'poor': '#d62728', 'medium': '#ff7f0e', 'good': '#2ca02c'},
        ax=ax1
    )
    ax1.set_xlabel('Envelope Efficiency Class', fontsize=12)
    ax1.set_ylabel('Thermal Intensity (BTU/sqft/HDD)', fontsize=12)
    ax1.set_title('(a) By Envelope Class', fontsize=14)
    
    # (b) By climate zone
    ax2 = axes[1]
    if 'climate_zone' in df.columns:
        sns.boxplot(
            data=df,
            x='climate_zone',
            y='Thermal_Intensity_I',
            order=['mild', 'mixed', 'cold'],
            palette='coolwarm',
            ax=ax2
        )
        ax2.set_xlabel('Climate Zone', fontsize=12)
        ax2.set_ylabel('Thermal Intensity (BTU/sqft/HDD)', fontsize=12)
        ax2.set_title('(b) By Climate Zone', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure3_thermal_intensity_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure3_thermal_intensity_distribution.pdf", bbox_inches='tight')
    plt.close()


def validate_against_hc_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate microdata aggregates against official RECS HC tables.
    
    This function compares weighted estimates from the microdata
    with values from HC2.x, HC6.x, HC10.x tables (if available).
    """
    logger.info("Validating against official RECS tables...")
    
    validation_results = []
    
    # Total housing units (should match HC2.1)
    total_units = df['NWEIGHT'].sum() / 1e6
    validation_results.append({
        'Metric': 'Total gas-heated units (millions)',
        'Microdata': round(total_units, 2),
        'Official': 'HC6.1',  # Reference table
        'Notes': 'Compare with natural gas heating row'
    })
    
    # Mean heated floor area
    mean_sqft = weighted_mean(df, 'A_heated')
    validation_results.append({
        'Metric': 'Mean heated sqft',
        'Microdata': round(mean_sqft, 0),
        'Official': 'HC10.1',
        'Notes': 'Compare with overall mean'
    })
    
    # Mean HDD65
    mean_hdd = weighted_mean(df, 'HDD65')
    validation_results.append({
        'Metric': 'Mean HDD65',
        'Microdata': round(mean_hdd, 0),
        'Official': '-',
        'Notes': 'For reference'
    })
    
    # Housing type distribution
    if 'housing_type' in df.columns:
        for htype in df['housing_type'].dropna().unique():
            pct = weighted_proportion(df, 'housing_type', htype) * 100
            validation_results.append({
                'Metric': f'Housing type: {htype} (%)',
                'Microdata': round(pct, 1),
                'Official': 'HC2.1',
                'Notes': 'Compare with housing type rows'
            })
    
    # Year built distribution
    if 'year_built_cat' in df.columns:
        for yb in ['Before 1950', '1950-1959', '1960-1969', '1970-1979', 
                   '1980-1989', '1990-1999', '2000-2009', '2010-2015', '2016-2020']:
            if yb in df['year_built_cat'].values:
                pct = weighted_proportion(df, 'year_built_cat', yb) * 100
                validation_results.append({
                    'Metric': f'Year built: {yb} (%)',
                    'Microdata': round(pct, 1),
                    'Official': 'HC2.1',
                    'Notes': 'Compare with year built rows'
                })
    
    # Division distribution
    if 'division_name' in df.columns:
        for div in df['division_name'].dropna().unique():
            pct = weighted_proportion(df, 'division_name', div) * 100
            validation_results.append({
                'Metric': f'Division: {div} (%)',
                'Microdata': round(pct, 1),
                'Official': 'HC6.1',
                'Notes': 'Compare with division columns'
            })
    
    validation_df = pd.DataFrame(validation_results)
    
    # Save
    validation_df.to_csv(TABLES_DIR / "validation_against_official.csv", index=False)
    
    return validation_df


def generate_figure4_validation(df: pd.DataFrame, validation_df: pd.DataFrame = None):
    """
    Generate Figure 4: Visual comparison with official RECS aggregates.
    """
    logger.info("Generating Figure 4: Validation comparison")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) By division - mean heating energy
    ax1 = axes[0]
    if 'division_name' in df.columns:
        div_stats = df.groupby('division_name').apply(
            lambda x: pd.Series({
                'mean_btu': weighted_mean(x, 'E_heat_btu') if 'E_heat_btu' in x.columns else 0,
                'n_weighted': x['NWEIGHT'].sum()
            })
        ).reset_index()
        
        div_stats = div_stats.sort_values('mean_btu', ascending=False)
        
        bars = ax1.barh(div_stats['division_name'], div_stats['mean_btu'] / 1e6)
        ax1.set_xlabel('Mean Heating Energy (Million BTU)', fontsize=12)
        ax1.set_ylabel('Census Division', fontsize=12)
        ax1.set_title('(a) Mean Heating Energy by Division', fontsize=14)
    
    # (b) Housing type distribution
    ax2 = axes[1]
    if 'housing_type' in df.columns:
        htype_shares = df.groupby('housing_type')['NWEIGHT'].sum()
        htype_shares = htype_shares / htype_shares.sum() * 100
        htype_shares = htype_shares.sort_values(ascending=True)
        
        bars = ax2.barh(htype_shares.index, htype_shares.values, color='steelblue')
        
        for bar, pct in zip(bars, htype_shares.values):
            ax2.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%',
                ha='left', va='center', fontsize=10
            )
        
        ax2.set_xlabel('Share of Gas-Heated Homes (%)', fontsize=12)
        ax2.set_ylabel('Housing Type', fontsize=12)
        ax2.set_title('(b) Housing Type Distribution', fontsize=14)
        ax2.set_xlim(0, 80)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_validation.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure4_validation.pdf", bbox_inches='tight')
    plt.close()


def compute_replicate_weight_rse(df: pd.DataFrame, var: str) -> float:
    """
    Compute Relative Standard Error using replicate weights.
    
    RSE = (SE / Mean) × 100
    
    Where SE is computed using the successive difference replication (SDR) method.
    """
    # Find replicate weight columns
    rep_weights = [col for col in df.columns if col.startswith('NWEIGHT') and col != 'NWEIGHT']
    
    if len(rep_weights) == 0:
        logger.warning("No replicate weights found for RSE calculation")
        return np.nan
    
    # Compute main estimate
    main_mean = weighted_mean(df, var, 'NWEIGHT')
    
    # Compute replicate estimates
    rep_means = []
    for rep_col in rep_weights:
        rep_mean = weighted_mean(df, var, rep_col)
        rep_means.append(rep_mean)
    
    rep_means = np.array(rep_means)
    
    # Fay's BRR variance
    fay_constant = 0.5  # Typical for RECS
    variance = (1 / (len(rep_weights) * (1 - fay_constant)**2)) * np.sum((rep_means - main_mean)**2)
    se = np.sqrt(variance)
    
    rse = (se / main_mean) * 100 if main_mean != 0 else np.nan
    
    return rse


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics with RSEs.
    """
    logger.info("Generating summary statistics with RSEs...")
    
    numeric_vars = ['HDD65', 'A_heated', 'E_heat_btu', 'Thermal_Intensity_I']
    
    results = []
    for var in numeric_vars:
        if var in df.columns:
            row = {
                'Variable': var,
                'N': df[var].notna().sum(),
                'Mean': weighted_mean(df, var),
                'Std': weighted_std(df, var),
                'Min': df[var].min(),
                'Q25': weighted_quantile(df, var, 0.25),
                'Median': weighted_quantile(df, var, 0.5),
                'Q75': weighted_quantile(df, var, 0.75),
                'Max': df[var].max(),
                'RSE (%)': compute_replicate_weight_rse(df, var),
            }
            results.append(row)
    
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(TABLES_DIR / "summary_statistics.csv", index=False)
    
    return summary_df


def run_descriptive_validation():
    """
    Main function to run all descriptive statistics and validation.
    """
    logger.info("=" * 60)
    logger.info("Descriptive Statistics and Validation")
    logger.info("=" * 60)
    
    # Load data
    df = load_processed_data(use_clean=True)
    
    # Generate Table 1: Variable definitions
    table1 = generate_table1_variables(df)
    logger.info(f"Table 1: {len(table1)} variables documented")
    
    # Generate Table 2: Sample characteristics
    table2 = generate_table2_sample_characteristics(df)
    logger.info(f"Table 2: {len(table2)} rows (division × envelope combinations)")
    
    # Generate summary statistics
    summary = generate_summary_statistics(df)
    logger.info(f"Summary statistics generated for {len(summary)} variables")
    
    # Validation against official tables
    validation = validate_against_hc_tables(df)
    logger.info(f"Validation table: {len(validation)} metrics compared")
    
    # Generate figures
    generate_figure2_climate_envelope(df)
    generate_figure3_thermal_intensity_distribution(df)
    generate_figure4_validation(df, validation)
    
    logger.info("=" * 60)
    logger.info("Descriptive statistics and validation complete!")
    logger.info("=" * 60)
    
    return {
        'table1': table1,
        'table2': table2,
        'summary': summary,
        'validation': validation,
        'data': df
    }


if __name__ == "__main__":
    results = run_descriptive_validation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(results['validation'].to_string())
