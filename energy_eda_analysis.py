"""
Energy Consumption Data - Exploratory Data Analysis
For Applied Energy Manuscript Submission

This script performs comprehensive EDA including:
1. Data loading with physical sensor mapping
2. Statistical profiling (Table 1)
3. Temporal analysis (Figures 1 & 2)
4. Physics-based correlation analysis (Figure 3)
5. Outlier and data quality checks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# STEP 1: DATA LOADING & PHYSICAL MAPPING
# ============================================================================

def load_and_map_data(filepath='energydata_complete.csv'):
    """
    Load energy consumption data and create physical location mapping.
    
    Based on the original study (Candanedo et al., 2017), the sensors are located at:
    - T1/RH1: Kitchen
    - T2/RH2: Living Room
    - T3/RH3: Laundry Room
    - T4/RH4: Office Room
    - T5/RH5: Bathroom
    - T6/RH6: Outside North
    - T7/RH7: Ironing Room
    - T8/RH8: Teenager Room
    - T9/RH9: Parents Room
    - T_out/RH_out: Weather Station (Outside)
    """
    
    # Download data if not present
    import urllib.request
    import os
    
    if not os.path.exists(filepath):
        print(f"Downloading dataset from GitHub...")
        url = "https://raw.githubusercontent.com/Fateme9977/P2/main/energydata_complete.csv"
        urllib.request.urlretrieve(url, filepath)
        print(f"Dataset downloaded successfully.")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Physical location mapping dictionary
    sensor_mapping = {
        'T1': 'Kitchen',
        'T2': 'Living_Room',
        'T3': 'Laundry_Room',
        'T4': 'Office_Room',
        'T5': 'Bathroom',
        'T6': 'Outside_North',
        'T7': 'Ironing_Room',
        'T8': 'Teenager_Room',
        'T9': 'Parents_Room',
        'RH1': 'Kitchen',
        'RH2': 'Living_Room',
        'RH3': 'Laundry_Room',
        'RH4': 'Office_Room',
        'RH5': 'Bathroom',
        'RH6': 'Outside_North',
        'RH7': 'Ironing_Room',
        'RH8': 'Teenager_Room',
        'RH9': 'Parents_Room',
        'T_out': 'Weather_Station',
        'RH_out': 'Weather_Station',
        'Press_mm_hg': 'Weather_Station',
        'Windspeed': 'Weather_Station',
        'Visibility': 'Weather_Station',
        'Tdewpoint': 'Weather_Station',
        'rv1': 'Random_Variable_1',
        'rv2': 'Random_Variable_2',
        'Appliances': 'Total_Energy_Consumption',
        'lights': 'Lighting_Energy'
    }
    
    # Create descriptive column names for display
    display_names = {}
    for col in df.columns:
        if col in sensor_mapping:
            if col.startswith('T') and col != 'T_out' and col != 'Tdewpoint':
                display_names[col] = f"T_{sensor_mapping[col]}"
            elif col.startswith('RH') and col != 'RH_out':
                display_names[col] = f"RH_{sensor_mapping[col]}"
            else:
                display_names[col] = col.replace('_', ' ').title()
        else:
            display_names[col] = col.replace('_', ' ').title()
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total time span: {(df.index.max() - df.index.min()).days} days")
    
    return df, sensor_mapping, display_names

# ============================================================================
# STEP 2: STATISTICAL PROFILING (TABLE 1)
# ============================================================================

def generate_statistical_profile(df, display_names):
    """
    Generate comprehensive descriptive statistics including skewness and kurtosis.
    This creates Table 1 for the manuscript.
    """
    print("\n" + "="*80)
    print("STEP 2: STATISTICAL PROFILING (TABLE 1)")
    print("="*80)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate statistics
    stats_dict = {
        'Variable': [],
        'Mean': [],
        'Median': [],
        'Std_Dev': [],
        'Min': [],
        'Max': [],
        'Skewness': [],
        'Kurtosis': []
    }
    
    for col in numeric_cols:
        stats_dict['Variable'].append(display_names.get(col, col))
        stats_dict['Mean'].append(df[col].mean())
        stats_dict['Median'].append(df[col].median())
        stats_dict['Std_Dev'].append(df[col].std())
        stats_dict['Min'].append(df[col].min())
        stats_dict['Max'].append(df[col].max())
        stats_dict['Skewness'].append(stats.skew(df[col].dropna()))
        stats_dict['Kurtosis'].append(stats.kurtosis(df[col].dropna()))
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Round to appropriate decimal places
    for col in ['Mean', 'Median', 'Std_Dev', 'Min', 'Max', 'Skewness', 'Kurtosis']:
        stats_df[col] = stats_df[col].round(3)
    
    # Display key statistics
    print("\nDescriptive Statistics (Table 1):")
    print(stats_df.to_string(index=False))
    
    # Save to CSV
    stats_df.to_csv('table1_statistical_profile.csv', index=False)
    print("\nTable 1 saved to 'table1_statistical_profile.csv'")
    
    # Interpretation of skewness
    print("\n--- Skewness Interpretation ---")
    print("Skewness > 1: Strong positive skew (right-tailed)")
    print("Skewness 0.5-1: Moderate positive skew")
    print("Skewness -0.5 to 0.5: Approximately symmetric")
    print("Skewness < -0.5: Negative skew (left-tailed)")
    
    # Highlight non-Gaussian distributions
    non_gaussian = stats_df[abs(stats_df['Skewness']) > 1]
    if len(non_gaussian) > 0:
        print(f"\nVariables with strong non-Gaussian distribution (|Skewness| > 1):")
        for _, row in non_gaussian.iterrows():
            print(f"  - {row['Variable']}: Skewness = {row['Skewness']:.3f}")
    
    return stats_df

# ============================================================================
# STEP 3: TEMPORAL ANALYSIS (FIGURES 1 & 2)
# ============================================================================

def temporal_analysis(df, display_names):
    """
    Create temporal visualizations:
    - Figure 1: Time-series plot (sample week)
    - Figure 2: Average daily profile (hourly patterns)
    """
    print("\n" + "="*80)
    print("STEP 3: TEMPORAL ANALYSIS (FIGURES 1 & 2)")
    print("="*80)
    
    # Figure 1: Time-series plot for a sample week
    # Find first week of February (or any representative week)
    feb_data = df[df.index.month == 2]
    if len(feb_data) > 0:
        sample_start = feb_data.index[0]
        sample_end = sample_start + pd.Timedelta(days=7)
        sample_week = df.loc[sample_start:sample_end]
    else:
        # Use first week of data if February not available
        sample_start = df.index[0]
        sample_end = sample_start + pd.Timedelta(days=7)
        sample_week = df.loc[sample_start:sample_end]
    
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot Appliances (Energy Consumption)
    ax1.plot(sample_week.index, sample_week['Appliances'], 
             color='#2E86AB', linewidth=1.5, label='Appliances Energy (Wh)')
    ax1.set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    ax1.set_title('Figure 1: Time-Series Analysis - Sample Week\nAppliances Energy Consumption', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot Outdoor Temperature
    ax2.plot(sample_week.index, sample_week['T_out'], 
             color='#A23B72', linewidth=1.5, label='Outdoor Temperature (°C)')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax2.set_title('Outdoor Temperature', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('figure1_time_series.png', dpi=300, bbox_inches='tight')
    print("Figure 1 saved: 'figure1_time_series.png'")
    plt.close()
    
    # Figure 2: Average Daily Profile (Hourly Patterns)
    df['hour'] = df.index.hour
    hourly_profile = df.groupby('hour')['Appliances'].mean()
    
    fig2, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hourly_profile.index, hourly_profile.values, 
            marker='o', linewidth=2.5, markersize=8, 
            color='#F18F01', label='Average Energy Consumption')
    ax.fill_between(hourly_profile.index, hourly_profile.values, 
                    alpha=0.3, color='#F18F01')
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Average Daily Profile - Hourly Energy Consumption Patterns', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    
    # Highlight peak hours
    peak_hour = hourly_profile.idxmax()
    ax.axvline(x=peak_hour, color='red', linestyle='--', linewidth=2, 
               label=f'Peak Hour: {peak_hour}:00')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figure2_hourly_profile.png', dpi=300, bbox_inches='tight')
    print("Figure 2 saved: 'figure2_hourly_profile.png'")
    plt.close()
    
    # Print insights
    print(f"\n--- Temporal Analysis Insights ---")
    print(f"Peak energy consumption hour: {peak_hour}:00 ({hourly_profile.max():.2f} Wh)")
    print(f"Minimum energy consumption hour: {hourly_profile.idxmin()}:00 ({hourly_profile.min():.2f} Wh)")
    print(f"Peak-to-minimum ratio: {hourly_profile.max()/hourly_profile.min():.2f}x")
    
    return hourly_profile

# ============================================================================
# STEP 4: PHYSICS-BASED CORRELATION ANALYSIS (FIGURE 3)
# ============================================================================

def correlation_analysis(df, display_names):
    """
    Generate correlation heatmap focusing on indoor vs outdoor temperature relationships.
    This helps assess building thermal insulation.
    """
    print("\n" + "="*80)
    print("STEP 4: PHYSICS-BASED CORRELATION ANALYSIS (FIGURE 3)")
    print("="*80)
    
    # Select temperature columns
    temp_cols = [col for col in df.columns if col.startswith('T') and col != 'Tdewpoint']
    temp_cols.append('Appliances')  # Include target variable
    
    # Calculate correlation matrix
    corr_matrix = df[temp_cols].corr()
    
    # Create heatmap
    fig3, ax = plt.subplots(figsize=(14, 12))
    
    # Create custom labels
    labels = [display_names.get(col, col) for col in temp_cols]
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=labels, yticklabels=labels,
                ax=ax)
    
    ax.set_title('Figure 3: Correlation Heatmap - Temperature Variables and Energy Consumption\n' +
                 'Focus: Indoor-Outdoor Temperature Relationships (Thermal Insulation Assessment)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('figure3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Figure 3 saved: 'figure3_correlation_heatmap.png'")
    plt.close()
    
    # Analyze indoor-outdoor temperature correlations
    indoor_temps = [f'T{i}' for i in range(1, 10)]
    outdoor_temp = 'T_out'
    
    print("\n--- Thermal Insulation Analysis ---")
    print("Correlation between Indoor Temperatures and Outdoor Temperature (T_out):")
    print("-" * 70)
    
    insulation_scores = []
    for temp_col in indoor_temps:
        if temp_col in df.columns:
            corr_value = df[temp_col].corr(df[outdoor_temp])
            insulation_scores.append({
                'Location': display_names.get(temp_col, temp_col),
                'Correlation': corr_value
            })
            insulation_level = "Low insulation" if abs(corr_value) > 0.7 else "Moderate insulation" if abs(corr_value) > 0.4 else "High insulation"
            print(f"{display_names.get(temp_col, temp_col):25s}: {corr_value:6.3f} ({insulation_level})")
    
    avg_corr = np.mean([abs(s['Correlation']) for s in insulation_scores])
    print(f"\nAverage absolute correlation: {avg_corr:.3f}")
    
    if avg_corr > 0.6:
        print("→ Building Assessment: LOW thermal insulation (strong indoor-outdoor coupling)")
    elif avg_corr > 0.3:
        print("→ Building Assessment: MODERATE thermal insulation")
    else:
        print("→ Building Assessment: HIGH thermal insulation (weak indoor-outdoor coupling)")
    
    return corr_matrix

# ============================================================================
# STEP 5: OUTLIER & QUALITY CHECK
# ============================================================================

def data_quality_check(df, display_names):
    """
    Perform comprehensive data quality checks:
    - Missing values
    - Unrealistic readings (e.g., RH > 100% or < 0%)
    - Outlier detection for target variable
    """
    print("\n" + "="*80)
    print("STEP 5: OUTLIER & DATA QUALITY CHECK")
    print("="*80)
    
    # Check missing values
    print("\n--- Missing Values Check ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Variable': [display_names.get(col, col) for col in df.columns],
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("✓ No missing values detected.")
    
    # Check unrealistic humidity values
    print("\n--- Humidity Range Check ---")
    rh_cols = [col for col in df.columns if col.startswith('RH')]
    unrealistic_rh = []
    
    for col in rh_cols:
        out_of_range = df[(df[col] < 0) | (df[col] > 100)]
        if len(out_of_range) > 0:
            unrealistic_rh.append({
                'Variable': display_names.get(col, col),
                'Out_of_Range_Count': len(out_of_range),
                'Percentage': (len(out_of_range) / len(df)) * 100
            })
    
    if unrealistic_rh:
        print(pd.DataFrame(unrealistic_rh).to_string(index=False))
    else:
        print("✓ All humidity values within valid range (0-100%).")
    
    # Check temperature ranges (should be reasonable for indoor/outdoor)
    print("\n--- Temperature Range Check ---")
    temp_cols = [col for col in df.columns if col.startswith('T')]
    temp_ranges = []
    
    for col in temp_cols:
        temp_ranges.append({
            'Variable': display_names.get(col, col),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Mean': df[col].mean()
        })
    
    temp_range_df = pd.DataFrame(temp_ranges)
    print(temp_range_df.to_string(index=False))
    
    # Outlier detection for Appliances (target variable)
    print("\n--- Outlier Detection: Appliances Energy Consumption ---")
    
    Q1 = df['Appliances'].quantile(0.25)
    Q3 = df['Appliances'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['Appliances'] < lower_bound) | (df['Appliances'] > upper_bound)]
    outlier_pct = (len(outliers) / len(df)) * 100
    
    print(f"IQR Method:")
    print(f"  Q1: {Q1:.2f} Wh")
    print(f"  Q3: {Q3:.2f} Wh")
    print(f"  IQR: {IQR:.2f} Wh")
    print(f"  Lower bound: {lower_bound:.2f} Wh")
    print(f"  Upper bound: {upper_bound:.2f} Wh")
    print(f"  Outliers detected: {len(outliers)} ({outlier_pct:.2f}%)")
    
    # Boxplot visualization
    fig5, ax = plt.subplots(figsize=(10, 6))
    box_plot = ax.boxplot(df['Appliances'], vert=True, patch_artist=True,
                          boxprops=dict(facecolor='#FF6B6B', alpha=0.7),
                          medianprops=dict(color='black', linewidth=2),
                          whiskerprops=dict(color='black', linewidth=1.5),
                          capprops=dict(color='black', linewidth=1.5))
    
    ax.set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    ax.set_title('Boxplot: Appliances Energy Consumption Distribution\n' +
                 f'Outliers: {len(outliers)} observations ({outlier_pct:.1f}%)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Median: {df["Appliances"].median():.1f} Wh\n'
    stats_text += f'Mean: {df["Appliances"].mean():.1f} Wh\n'
    stats_text += f'Std Dev: {df["Appliances"].std():.1f} Wh'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figure5_outlier_boxplot.png', dpi=300, bbox_inches='tight')
    print("\nBoxplot saved: 'figure5_outlier_boxplot.png'")
    plt.close()
    
    return {
        'missing': missing_df,
        'unrealistic_rh': unrealistic_rh,
        'outliers': outliers,
        'outlier_stats': {
            'count': len(outliers),
            'percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for comprehensive EDA analysis.
    """
    print("="*80)
    print("ENERGY CONSUMPTION DATA - EXPLORATORY DATA ANALYSIS")
    print("For Applied Energy Manuscript Submission")
    print("="*80)
    
    # Step 1: Load and map data
    df, sensor_mapping, display_names = load_and_map_data()
    
    # Step 2: Statistical profiling
    stats_df = generate_statistical_profile(df, display_names)
    
    # Step 3: Temporal analysis
    hourly_profile = temporal_analysis(df, display_names)
    
    # Step 4: Correlation analysis
    corr_matrix = correlation_analysis(df, display_names)
    
    # Step 5: Data quality check
    quality_results = data_quality_check(df, display_names)
    
    print("\n" + "="*80)
    print("EDA ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print("  - table1_statistical_profile.csv")
    print("  - figure1_time_series.png")
    print("  - figure2_hourly_profile.png")
    print("  - figure3_correlation_heatmap.png")
    print("  - figure5_outlier_boxplot.png")
    
    return df, stats_df, hourly_profile, corr_matrix, quality_results

if __name__ == "__main__":
    df, stats_df, hourly_profile, corr_matrix, quality_results = main()
