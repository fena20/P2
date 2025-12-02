# Energy Consumption Data - Exploratory Data Analysis

## Project Overview

This project performs comprehensive Exploratory Data Analysis (EDA) on residential energy consumption data for an Applied Energy manuscript submission. The analysis focuses on understanding energy usage patterns, thermal building performance, and occupant behavior.

## Dataset

- **Source:** UCI Machine Learning Repository / Candanedo et al., 2017
- **Records:** 19,735 observations
- **Time Span:** 137 days (January 11 - May 27, 2016)
- **Sampling:** 10-minute intervals
- **Features:** 28 variables including:
  - Indoor temperatures (9 rooms)
  - Indoor humidity (9 rooms)
  - Weather data (temperature, humidity, wind, visibility)
  - Energy consumption (appliances, lights)

## Quick Start

```bash
# Run complete EDA analysis
python3 energy_eda_complete.py
```

## Generated Outputs

### Tables
| File | Description |
|------|-------------|
| `table1_statistical_profile.csv` | Complete descriptive statistics |

### Figures
| File | Description |
|------|-------------|
| `figure1_time_series.png` | Sample week time-series analysis |
| `figure2_hourly_profile.png` | 24-hour average energy profile |
| `figure3_correlation_heatmap.png` | Temperature-Energy correlations |
| `figure4_feature_importance.png` | Feature importance analysis |
| `figure5_outlier_boxplot.png` | Outlier and distribution analysis |
| `figure6_weather_impact.png` | Weather impact on energy |
| `figure7_weekly_patterns.png` | Weekly consumption patterns |
| `figure8_room_temperatures.png` | Room temperature comparison |
| `figure9_monthly_trends.png` | Monthly trends analysis |

## Key Findings

### 1. Energy Consumption Patterns
- **Peak hour:** 18:00 (190.36 Wh)
- **Minimum hour:** 03:00 (48.24 Wh)
- **Peak-to-min ratio:** 3.95x

### 2. Thermal Insulation
- Average indoor-outdoor correlation: 0.696
- Assessment: **LOW thermal insulation**

### 3. Data Quality
- Missing values: None ✓
- Outliers: 10.83% (legitimate high-consumption events)

### 4. Distribution Characteristics
- Appliances energy: Skewness = 3.386 (non-Gaussian)
- Right-skewed distribution with occasional high-energy spikes

## Project Structure

```
/workspace
├── README.md                      # This file
├── EDA_INSIGHTS_SUMMARY.md        # Detailed analysis summary
├── energy_eda_complete.py         # Complete EDA script
├── energy_eda_analysis.py         # Original EDA script
├── energydata_complete.csv        # Dataset
├── table1_statistical_profile.csv # Statistical summary
├── figure1_time_series.png        # Time-series visualization
├── figure2_hourly_profile.png     # Hourly patterns
├── figure3_correlation_heatmap.png # Correlation matrix
├── figure4_feature_importance.png  # Feature analysis
├── figure5_outlier_boxplot.png    # Outlier detection
├── figure6_weather_impact.png     # Weather analysis
├── figure7_weekly_patterns.png    # Weekly patterns
├── figure8_room_temperatures.png  # Room comparison
└── figure9_monthly_trends.png     # Monthly trends
```

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Citation

If you use this analysis, please cite:
- Candanedo, L.M., Feldheim, V., Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. Energy and Buildings, 140, 81-97.

## License

This project is for academic research purposes.
