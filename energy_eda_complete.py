"""
Energy Consumption Data - Complete Exploratory Data Analysis
For Applied Energy Manuscript Submission

This script performs comprehensive EDA including:
1. Data loading with physical sensor mapping
2. Statistical profiling (Table 1)
3. Temporal analysis (Figures 1 & 2)
4. Physics-based correlation analysis (Figure 3)
5. Feature importance analysis (Figure 4)
6. Outlier and data quality checks (Figure 5)
7. Weather impact analysis (Figure 6)
8. Weekly patterns analysis (Figure 7)
9. Distribution analysis (Figure 8)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'dark': '#3B1F2B',
    'light': '#E8E8E8'
}

# ============================================================================
# STEP 1: DATA LOADING & PHYSICAL MAPPING
# ============================================================================

def load_and_map_data(filepath='energydata_complete.csv'):
    """
    Load energy consumption data and create physical location mapping.
    """
    import urllib.request
    import os
    
    if not os.path.exists(filepath):
        print(f"Downloading dataset from GitHub...")
        url = "https://raw.githubusercontent.com/Fateme9977/P2/main/energydata_complete.csv"
        urllib.request.urlretrieve(url, filepath)
        print(f"Dataset downloaded successfully.")
    
    print("Loading data...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Physical location mapping
    sensor_mapping = {
        'T1': 'Kitchen', 'T2': 'Living_Room', 'T3': 'Laundry_Room',
        'T4': 'Office_Room', 'T5': 'Bathroom', 'T6': 'Outside_North',
        'T7': 'Ironing_Room', 'T8': 'Teenager_Room', 'T9': 'Parents_Room',
        'RH1': 'Kitchen', 'RH2': 'Living_Room', 'RH3': 'Laundry_Room',
        'RH4': 'Office_Room', 'RH5': 'Bathroom', 'RH6': 'Outside_North',
        'RH7': 'Ironing_Room', 'RH8': 'Teenager_Room', 'RH9': 'Parents_Room',
        'T_out': 'Weather_Station', 'RH_out': 'Weather_Station',
        'Press_mm_hg': 'Weather_Station', 'Windspeed': 'Weather_Station',
        'Visibility': 'Weather_Station', 'Tdewpoint': 'Weather_Station',
        'rv1': 'Random_Variable_1', 'rv2': 'Random_Variable_2',
        'Appliances': 'Total_Energy_Consumption', 'lights': 'Lighting_Energy'
    }
    
    display_names = {}
    for col in df.columns:
        if col in sensor_mapping:
            if col.startswith('T') and col not in ['T_out', 'Tdewpoint']:
                display_names[col] = f"T_{sensor_mapping[col]}"
            elif col.startswith('RH') and col != 'RH_out':
                display_names[col] = f"RH_{sensor_mapping[col]}"
            else:
                display_names[col] = col.replace('_', ' ').title()
        else:
            display_names[col] = col.replace('_', ' ').title()
    
    # Add time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print(f"✓ Data loaded successfully. Shape: {df.shape}")
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
    print(f"✓ Total time span: {(df.index.max() - df.index.min()).days} days")
    
    return df, sensor_mapping, display_names

# ============================================================================
# STEP 2: STATISTICAL PROFILING (TABLE 1)
# ============================================================================

def generate_statistical_profile(df, display_names):
    """Generate comprehensive descriptive statistics."""
    print("\n" + "="*80)
    print("STEP 2: STATISTICAL PROFILING (TABLE 1)")
    print("="*80)
    
    # Exclude time features
    exclude_cols = ['hour', 'day_of_week', 'month', 'is_weekend']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    stats_dict = {
        'Variable': [], 'Count': [], 'Mean': [], 'Median': [],
        'Std_Dev': [], 'Min': [], 'Max': [], 'Skewness': [], 'Kurtosis': []
    }
    
    for col in numeric_cols:
        stats_dict['Variable'].append(display_names.get(col, col))
        stats_dict['Count'].append(int(df[col].count()))
        stats_dict['Mean'].append(df[col].mean())
        stats_dict['Median'].append(df[col].median())
        stats_dict['Std_Dev'].append(df[col].std())
        stats_dict['Min'].append(df[col].min())
        stats_dict['Max'].append(df[col].max())
        stats_dict['Skewness'].append(stats.skew(df[col].dropna()))
        stats_dict['Kurtosis'].append(stats.kurtosis(df[col].dropna()))
    
    stats_df = pd.DataFrame(stats_dict)
    for col in ['Mean', 'Median', 'Std_Dev', 'Min', 'Max', 'Skewness', 'Kurtosis']:
        stats_df[col] = stats_df[col].round(3)
    
    print("\nDescriptive Statistics (Table 1):")
    print(stats_df.to_string(index=False))
    
    stats_df.to_csv('table1_statistical_profile.csv', index=False)
    print("\n✓ Table 1 saved to 'table1_statistical_profile.csv'")
    
    # Non-Gaussian distributions
    print("\n--- Non-Gaussian Distributions (|Skewness| > 1) ---")
    non_gaussian = stats_df[abs(stats_df['Skewness']) > 1]
    for _, row in non_gaussian.iterrows():
        print(f"  • {row['Variable']}: Skewness = {row['Skewness']:.3f}, Kurtosis = {row['Kurtosis']:.3f}")
    
    return stats_df

# ============================================================================
# STEP 3: TEMPORAL ANALYSIS (FIGURES 1 & 2)
# ============================================================================

def temporal_analysis(df, display_names):
    """Create temporal visualizations."""
    print("\n" + "="*80)
    print("STEP 3: TEMPORAL ANALYSIS (FIGURES 1 & 2)")
    print("="*80)
    
    # Figure 1: Time-series (sample week)
    feb_data = df[df.index.month == 2]
    if len(feb_data) > 0:
        sample_start = feb_data.index[0]
        sample_end = sample_start + pd.Timedelta(days=7)
        sample_week = df.loc[sample_start:sample_end]
    else:
        sample_start = df.index[0]
        sample_end = sample_start + pd.Timedelta(days=7)
        sample_week = df.loc[sample_start:sample_end]
    
    fig1, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Energy consumption
    axes[0].plot(sample_week.index, sample_week['Appliances'], 
                 color=COLORS['primary'], linewidth=1.5, label='Appliances Energy (Wh)')
    axes[0].fill_between(sample_week.index, sample_week['Appliances'], alpha=0.3, color=COLORS['primary'])
    axes[0].set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
    axes[0].set_title('Figure 1: Time-Series Analysis - Sample Week', fontsize=16, fontweight='bold', pad=15)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Outdoor temperature
    axes[1].plot(sample_week.index, sample_week['T_out'], 
                 color=COLORS['secondary'], linewidth=1.5, label='Outdoor Temperature (°C)')
    axes[1].fill_between(sample_week.index, sample_week['T_out'], alpha=0.3, color=COLORS['secondary'])
    axes[1].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Indoor temperature (Kitchen)
    axes[2].plot(sample_week.index, sample_week['T1'], 
                 color=COLORS['accent'], linewidth=1.5, label='Kitchen Temperature (°C)')
    axes[2].fill_between(sample_week.index, sample_week['T1'], alpha=0.3, color=COLORS['accent'])
    axes[2].set_xlabel('Date', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig('figure1_time_series.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 1 saved: 'figure1_time_series.png'")
    plt.close()
    
    # Figure 2: Hourly Profile
    hourly_profile = df.groupby('hour')['Appliances'].agg(['mean', 'std', 'median'])
    
    fig2, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(hourly_profile.index, hourly_profile['mean'], 
            marker='o', linewidth=3, markersize=10, color=COLORS['accent'], label='Mean')
    ax.fill_between(hourly_profile.index, 
                    hourly_profile['mean'] - hourly_profile['std'],
                    hourly_profile['mean'] + hourly_profile['std'],
                    alpha=0.2, color=COLORS['accent'], label='±1 Std Dev')
    ax.plot(hourly_profile.index, hourly_profile['median'], 
            marker='s', linewidth=2, markersize=6, color=COLORS['primary'], 
            linestyle='--', label='Median')
    
    peak_hour = hourly_profile['mean'].idxmax()
    min_hour = hourly_profile['mean'].idxmin()
    ax.axvline(x=peak_hour, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate(f'Peak: {peak_hour}:00\n({hourly_profile["mean"].max():.1f} Wh)',
                xy=(peak_hour, hourly_profile['mean'].max()),
                xytext=(peak_hour+1.5, hourly_profile['mean'].max()+20),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energy Consumption (Wh)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 2: Average Daily Energy Profile (24-Hour Pattern)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure2_hourly_profile.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 2 saved: 'figure2_hourly_profile.png'")
    plt.close()
    
    print(f"\n--- Temporal Insights ---")
    print(f"  • Peak hour: {peak_hour}:00 ({hourly_profile['mean'].max():.2f} Wh)")
    print(f"  • Minimum hour: {min_hour}:00 ({hourly_profile['mean'].min():.2f} Wh)")
    print(f"  • Peak-to-min ratio: {hourly_profile['mean'].max()/hourly_profile['mean'].min():.2f}x")
    
    return hourly_profile

# ============================================================================
# STEP 4: CORRELATION ANALYSIS (FIGURE 3)
# ============================================================================

def correlation_analysis(df, display_names):
    """Generate correlation heatmap."""
    print("\n" + "="*80)
    print("STEP 4: PHYSICS-BASED CORRELATION ANALYSIS (FIGURE 3)")
    print("="*80)
    
    # Select key variables
    key_cols = ['Appliances', 'lights', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 
                'T_out', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
    available_cols = [c for c in key_cols if c in df.columns]
    
    corr_matrix = df[available_cols].corr()
    
    fig3, ax = plt.subplots(figsize=(16, 14))
    
    labels = [display_names.get(col, col) for col in available_cols]
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Correlation"},
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 9})
    
    ax.set_title('Figure 3: Correlation Heatmap\nTemperature, Humidity & Energy Relationships',
                 fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('figure3_correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 3 saved: 'figure3_correlation_heatmap.png'")
    plt.close()
    
    # Thermal insulation analysis
    indoor_temps = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
    print("\n--- Thermal Insulation Analysis ---")
    print("Indoor-Outdoor Temperature Correlations:")
    print("-" * 60)
    
    correlations = []
    for temp_col in indoor_temps:
        if temp_col in df.columns:
            corr = df[temp_col].corr(df['T_out'])
            level = "Low insulation" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "High insulation"
            print(f"  {display_names.get(temp_col, temp_col):25s}: {corr:6.3f} ({level})")
            correlations.append(abs(corr))
    
    avg_corr = np.mean(correlations)
    print(f"\n  Average correlation: {avg_corr:.3f}")
    
    if avg_corr > 0.6:
        print("  → Building: LOW thermal insulation (strong indoor-outdoor coupling)")
    elif avg_corr > 0.3:
        print("  → Building: MODERATE thermal insulation")
    else:
        print("  → Building: HIGH thermal insulation")
    
    return corr_matrix

# ============================================================================
# STEP 5: FEATURE IMPORTANCE (FIGURE 4)
# ============================================================================

def feature_importance_analysis(df, display_names):
    """Analyze feature importance for energy prediction."""
    print("\n" + "="*80)
    print("STEP 5: FEATURE IMPORTANCE ANALYSIS (FIGURE 4)")
    print("="*80)
    
    # Calculate correlations with Appliances
    exclude_cols = ['Appliances', 'lights', 'rv1', 'rv2', 'hour', 'day_of_week', 'month', 'is_weekend']
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    correlations = []
    for col in feature_cols:
        corr = df[col].corr(df['Appliances'])
        correlations.append({
            'Feature': display_names.get(col, col),
            'Correlation': corr,
            'Abs_Correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=True)
    
    fig4, ax = plt.subplots(figsize=(12, 10))
    
    colors = [COLORS['primary'] if c > 0 else COLORS['secondary'] for c in corr_df['Correlation']]
    
    bars = ax.barh(range(len(corr_df)), corr_df['Correlation'], color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['Feature'], fontsize=10)
    ax.set_xlabel('Correlation with Energy Consumption', fontsize=13, fontweight='bold')
    ax.set_title('Figure 4: Feature Importance - Correlation with Appliances Energy',
                 fontsize=16, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, corr_df['Correlation'])):
        ax.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}',
                va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figure4_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 4 saved: 'figure4_feature_importance.png'")
    plt.close()
    
    # Top features
    print("\n--- Top 5 Features (by absolute correlation) ---")
    top5 = corr_df.nlargest(5, 'Abs_Correlation')
    for _, row in top5.iterrows():
        print(f"  • {row['Feature']}: {row['Correlation']:.3f}")
    
    return corr_df

# ============================================================================
# STEP 6: OUTLIER DETECTION (FIGURE 5)
# ============================================================================

def data_quality_check(df, display_names):
    """Perform data quality checks and outlier detection."""
    print("\n" + "="*80)
    print("STEP 6: DATA QUALITY & OUTLIER ANALYSIS (FIGURE 5)")
    print("="*80)
    
    # Missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values detected.")
    else:
        print(f"⚠ Missing values: {missing[missing > 0].to_dict()}")
    
    # Humidity range check
    print("\n--- Humidity Range Check ---")
    rh_cols = [col for col in df.columns if col.startswith('RH')]
    rh_issues = 0
    for col in rh_cols:
        out_of_range = df[(df[col] < 0) | (df[col] > 100)]
        if len(out_of_range) > 0:
            print(f"⚠ {col}: {len(out_of_range)} values out of range")
            rh_issues += len(out_of_range)
    if rh_issues == 0:
        print("✓ All humidity values within valid range (0-100%).")
    
    # Outlier detection
    print("\n--- Outlier Detection (Appliances) ---")
    Q1 = df['Appliances'].quantile(0.25)
    Q3 = df['Appliances'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['Appliances'] < lower_bound) | (df['Appliances'] > upper_bound)]
    outlier_pct = (len(outliers) / len(df)) * 100
    
    print(f"  Q1: {Q1:.2f} Wh | Q3: {Q3:.2f} Wh | IQR: {IQR:.2f} Wh")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] Wh")
    print(f"  Outliers: {len(outliers)} ({outlier_pct:.2f}%)")
    
    # Figure 5: Boxplot
    fig5, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Boxplot
    bp = axes[0].boxplot(df['Appliances'], vert=True, patch_artist=True,
                         boxprops=dict(facecolor=COLORS['accent'], alpha=0.7),
                         medianprops=dict(color='black', linewidth=2),
                         whiskerprops=dict(color='black', linewidth=1.5),
                         capprops=dict(color='black', linewidth=1.5),
                         flierprops=dict(marker='o', markerfacecolor=COLORS['secondary'], 
                                        markersize=4, alpha=0.5))
    
    axes[0].set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Boxplot: Appliances Energy\nOutliers: {len(outliers)} ({outlier_pct:.1f}%)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Distribution histogram
    axes[1].hist(df['Appliances'], bins=50, color=COLORS['primary'], alpha=0.7, 
                 edgecolor='black', density=True)
    axes[1].axvline(df['Appliances'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {df["Appliances"].mean():.1f}')
    axes[1].axvline(df['Appliances'].median(), color='green', linestyle='-', 
                    linewidth=2, label=f'Median: {df["Appliances"].median():.1f}')
    axes[1].set_xlabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Appliances Energy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 5: Outlier & Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure5_outlier_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 5 saved: 'figure5_outlier_boxplot.png'")
    plt.close()
    
    return {'outliers': len(outliers), 'outlier_pct': outlier_pct}

# ============================================================================
# STEP 7: WEATHER IMPACT ANALYSIS (FIGURE 6)
# ============================================================================

def weather_impact_analysis(df, display_names):
    """Analyze weather impact on energy consumption."""
    print("\n" + "="*80)
    print("STEP 7: WEATHER IMPACT ANALYSIS (FIGURE 6)")
    print("="*80)
    
    fig6, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Temperature vs Energy
    axes[0, 0].scatter(df['T_out'], df['Appliances'], alpha=0.1, c=COLORS['primary'], s=10)
    z = np.polyfit(df['T_out'], df['Appliances'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['T_out'].min(), df['T_out'].max(), 100)
    axes[0, 0].plot(x_line, p(x_line), color='red', linewidth=2, label=f'Trend (r={df["T_out"].corr(df["Appliances"]):.3f})')
    axes[0, 0].set_xlabel('Outdoor Temperature (°C)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Temperature vs Energy', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Humidity vs Energy
    axes[0, 1].scatter(df['RH_out'], df['Appliances'], alpha=0.1, c=COLORS['secondary'], s=10)
    z = np.polyfit(df['RH_out'], df['Appliances'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['RH_out'].min(), df['RH_out'].max(), 100)
    axes[0, 1].plot(x_line, p(x_line), color='red', linewidth=2, label=f'Trend (r={df["RH_out"].corr(df["Appliances"]):.3f})')
    axes[0, 1].set_xlabel('Outdoor Humidity (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Humidity vs Energy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Windspeed vs Energy
    axes[1, 0].scatter(df['Windspeed'], df['Appliances'], alpha=0.1, c=COLORS['accent'], s=10)
    z = np.polyfit(df['Windspeed'], df['Appliances'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Windspeed'].min(), df['Windspeed'].max(), 100)
    axes[1, 0].plot(x_line, p(x_line), color='red', linewidth=2, label=f'Trend (r={df["Windspeed"].corr(df["Appliances"]):.3f})')
    axes[1, 0].set_xlabel('Wind Speed (m/s)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Wind Speed vs Energy', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Visibility vs Energy
    axes[1, 1].scatter(df['Visibility'], df['Appliances'], alpha=0.1, c=COLORS['success'], s=10)
    z = np.polyfit(df['Visibility'], df['Appliances'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Visibility'].min(), df['Visibility'].max(), 100)
    axes[1, 1].plot(x_line, p(x_line), color='red', linewidth=2, label=f'Trend (r={df["Visibility"].corr(df["Appliances"]):.3f})')
    axes[1, 1].set_xlabel('Visibility (km)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Visibility vs Energy', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 6: Weather Impact on Energy Consumption', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure6_weather_impact.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 6 saved: 'figure6_weather_impact.png'")
    plt.close()
    
    # Print correlations
    print("\n--- Weather-Energy Correlations ---")
    weather_vars = ['T_out', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'Press_mm_hg']
    for var in weather_vars:
        if var in df.columns:
            corr = df[var].corr(df['Appliances'])
            print(f"  {var:15s}: {corr:7.3f}")

# ============================================================================
# STEP 8: WEEKLY PATTERNS (FIGURE 7)
# ============================================================================

def weekly_patterns_analysis(df, display_names):
    """Analyze weekly energy consumption patterns."""
    print("\n" + "="*80)
    print("STEP 8: WEEKLY PATTERNS ANALYSIS (FIGURE 7)")
    print("="*80)
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_profile = df.groupby('day_of_week')['Appliances'].agg(['mean', 'std', 'median'])
    
    fig7, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Daily average by day of week
    colors = [COLORS['primary'] if i < 5 else COLORS['accent'] for i in range(7)]
    bars = axes[0].bar(range(7), daily_profile['mean'], color=colors, alpha=0.8, 
                       edgecolor='black', yerr=daily_profile['std']/np.sqrt(len(df)/7), capsize=5)
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(days, rotation=45, ha='right')
    axes[0].set_xlabel('Day of Week', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    axes[0].set_title('Average Energy by Day of Week', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['primary'], label='Weekday'),
                       Patch(facecolor=COLORS['accent'], label='Weekend')]
    axes[0].legend(handles=legend_elements)
    
    # Heatmap: Hour vs Day of Week
    pivot_table = df.pivot_table(values='Appliances', index='hour', columns='day_of_week', aggfunc='mean')
    pivot_table.columns = days
    
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=False, ax=axes[1],
                cbar_kws={'label': 'Energy (Wh)'})
    axes[1].set_xlabel('Day of Week', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Hour of Day', fontsize=12, fontweight='bold')
    axes[1].set_title('Energy Consumption Heatmap\n(Hour × Day)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Figure 7: Weekly Energy Consumption Patterns', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure7_weekly_patterns.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 7 saved: 'figure7_weekly_patterns.png'")
    plt.close()
    
    # Weekday vs Weekend comparison
    weekday_avg = df[df['is_weekend'] == 0]['Appliances'].mean()
    weekend_avg = df[df['is_weekend'] == 1]['Appliances'].mean()
    
    print("\n--- Weekday vs Weekend ---")
    print(f"  Weekday average: {weekday_avg:.2f} Wh")
    print(f"  Weekend average: {weekend_avg:.2f} Wh")
    print(f"  Difference: {((weekend_avg - weekday_avg) / weekday_avg) * 100:.1f}%")

# ============================================================================
# STEP 9: ROOM TEMPERATURE COMPARISON (FIGURE 8)
# ============================================================================

def room_temperature_analysis(df, display_names):
    """Analyze temperature distribution across rooms."""
    print("\n" + "="*80)
    print("STEP 9: ROOM TEMPERATURE ANALYSIS (FIGURE 8)")
    print("="*80)
    
    temp_cols = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T_out']
    room_names = ['Kitchen', 'Living\nRoom', 'Laundry\nRoom', 'Office', 'Bathroom',
                  'Outside\nNorth', 'Ironing\nRoom', 'Teenager\nRoom', 'Parents\nRoom', 'Outdoor']
    
    fig8, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Boxplot comparison
    temp_data = [df[col].values for col in temp_cols]
    bp = axes[0].boxplot(temp_data, labels=room_names, patch_artist=True)
    
    colors_box = plt.cm.viridis(np.linspace(0, 1, len(temp_cols)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_xlabel('Location', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    axes[0].set_title('Temperature Distribution by Room', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Average temperature bar chart
    avg_temps = [df[col].mean() for col in temp_cols]
    std_temps = [df[col].std() for col in temp_cols]
    
    bars = axes[1].bar(range(len(temp_cols)), avg_temps, color=colors_box, alpha=0.8,
                       edgecolor='black', yerr=std_temps, capsize=5)
    axes[1].set_xticks(range(len(temp_cols)))
    axes[1].set_xticklabels(room_names, rotation=45, ha='right')
    axes[1].set_xlabel('Location', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Temperature (°C)', fontsize=12, fontweight='bold')
    axes[1].set_title('Average Temperature by Room', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, avg_temps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Figure 8: Room Temperature Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure8_room_temperatures.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 8 saved: 'figure8_room_temperatures.png'")
    plt.close()
    
    # Print summary
    print("\n--- Room Temperature Summary ---")
    for col, name in zip(temp_cols, room_names):
        name_clean = name.replace('\n', ' ')
        print(f"  {name_clean:15s}: Mean={df[col].mean():5.1f}°C, Std={df[col].std():4.1f}°C")

# ============================================================================
# STEP 10: MONTHLY TRENDS (FIGURE 9)
# ============================================================================

def monthly_trends_analysis(df, display_names):
    """Analyze monthly energy consumption trends."""
    print("\n" + "="*80)
    print("STEP 10: MONTHLY TRENDS ANALYSIS (FIGURE 9)")
    print("="*80)
    
    months = ['January', 'February', 'March', 'April', 'May']
    monthly_data = df.groupby('month').agg({
        'Appliances': ['mean', 'std', 'sum'],
        'T_out': 'mean',
        'lights': 'mean'
    })
    
    fig9, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Monthly energy consumption
    month_labels = [months[m-1] for m in monthly_data.index]
    axes[0].bar(range(len(monthly_data)), monthly_data['Appliances']['mean'],
                color=COLORS['primary'], alpha=0.8, edgecolor='black',
                yerr=monthly_data['Appliances']['std']/10, capsize=5, label='Appliances')
    axes[0].set_xticks(range(len(monthly_data)))
    axes[0].set_xticklabels(month_labels, rotation=45, ha='right')
    axes[0].set_xlabel('Month', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Energy Consumption (Wh)', fontsize=12, fontweight='bold')
    axes[0].set_title('Average Energy by Month', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Energy vs Temperature trend
    ax2 = axes[1]
    ax3 = ax2.twinx()
    
    line1, = ax2.plot(range(len(monthly_data)), monthly_data['Appliances']['mean'],
                      marker='o', linewidth=3, markersize=10, color=COLORS['primary'], label='Energy')
    line2, = ax3.plot(range(len(monthly_data)), monthly_data['T_out']['mean'],
                      marker='s', linewidth=3, markersize=10, color=COLORS['secondary'], label='Temperature')
    
    ax2.set_xticks(range(len(monthly_data)))
    ax2.set_xticklabels(month_labels, rotation=45, ha='right')
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Energy Consumption (Wh)', fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax3.set_ylabel('Outdoor Temperature (°C)', fontsize=12, fontweight='bold', color=COLORS['secondary'])
    axes[1].set_title('Energy vs Temperature Trend', fontsize=14, fontweight='bold')
    
    lines = [line1, line2]
    labels = ['Energy', 'Temperature']
    ax2.legend(lines, labels, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 9: Monthly Trends Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure9_monthly_trends.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure 9 saved: 'figure9_monthly_trends.png'")
    plt.close()
    
    # Print summary
    print("\n--- Monthly Summary ---")
    for i, m in enumerate(monthly_data.index):
        print(f"  {months[m-1]:10s}: Energy={monthly_data['Appliances']['mean'].iloc[i]:.1f} Wh, "
              f"Temp={monthly_data['T_out']['mean'].iloc[i]:.1f}°C")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("COMPLETE ENERGY CONSUMPTION DATA - EXPLORATORY DATA ANALYSIS")
    print("For Applied Energy Manuscript Submission")
    print("="*80)
    
    # Step 1: Load data
    df, sensor_mapping, display_names = load_and_map_data()
    
    # Step 2: Statistical profiling
    stats_df = generate_statistical_profile(df, display_names)
    
    # Step 3: Temporal analysis
    hourly_profile = temporal_analysis(df, display_names)
    
    # Step 4: Correlation analysis
    corr_matrix = correlation_analysis(df, display_names)
    
    # Step 5: Feature importance
    importance_df = feature_importance_analysis(df, display_names)
    
    # Step 6: Data quality & outliers
    quality_results = data_quality_check(df, display_names)
    
    # Step 7: Weather impact
    weather_impact_analysis(df, display_names)
    
    # Step 8: Weekly patterns
    weekly_patterns_analysis(df, display_names)
    
    # Step 9: Room temperatures
    room_temperature_analysis(df, display_names)
    
    # Step 10: Monthly trends
    monthly_trends_analysis(df, display_names)
    
    print("\n" + "="*80)
    print("✅ COMPLETE EDA ANALYSIS FINISHED")
    print("="*80)
    print("\nGenerated Files:")
    print("  📊 Tables:")
    print("     - table1_statistical_profile.csv")
    print("  📈 Figures:")
    print("     - figure1_time_series.png")
    print("     - figure2_hourly_profile.png")
    print("     - figure3_correlation_heatmap.png")
    print("     - figure4_feature_importance.png")
    print("     - figure5_outlier_boxplot.png")
    print("     - figure6_weather_impact.png")
    print("     - figure7_weekly_patterns.png")
    print("     - figure8_room_temperatures.png")
    print("     - figure9_monthly_trends.png")
    print("="*80)
    
    return df, stats_df, hourly_profile, corr_matrix

if __name__ == "__main__":
    df, stats_df, hourly_profile, corr_matrix = main()
