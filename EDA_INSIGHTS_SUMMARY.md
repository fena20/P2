# Exploratory Data Analysis - Complete Insights Summary
## Energy Consumption Data for Applied Energy Manuscript

---

## Dataset Overview
| Parameter | Value |
|-----------|-------|
| **Total observations** | 19,735 records |
| **Time span** | 137 days (January 11 - May 27, 2016) |
| **Sampling frequency** | 10-minute intervals |
| **Variables** | 28 features (temperature, humidity, weather, energy) |

---

## 1. Statistical Profiling (Table 1)

### Key Findings:

**Non-Gaussian Distributions (High Skewness):**

| Variable | Skewness | Kurtosis | Interpretation |
|----------|----------|----------|----------------|
| Appliances | 3.386 | 13.664 | Strong right-skew, frequent low consumption with occasional spikes |
| Lights | 2.195 | 4.461 | Most observations show zero/minimal lighting |
| Bathroom Humidity (RH5) | 1.867 | 4.502 | High variability due to shower/bath activities |

**Energy Consumption Statistics:**
- Mean: 97.7 Wh
- Median: 60.0 Wh
- Std Dev: 102.5 Wh
- Range: 10 - 1,080 Wh

**Temperature Characteristics:**
- Indoor range: 14.9°C to 29.9°C
- Outdoor range: -5.0°C to 26.1°C
- Hottest room: Laundry Room (22.3°C average)
- Coldest indoor: Parents Room (19.5°C average)

---

## 2. Temporal Analysis (Figures 1 & 2)

### Figure 1: Time-Series Pattern
- Energy consumption shows clear **diurnal patterns**
- Multiple peaks throughout the day corresponding to occupant activities
- Outdoor temperature exhibits natural daily cycles

### Figure 2: Average Daily Profile

| Metric | Value |
|--------|-------|
| **Peak consumption hour** | 18:00 (6 PM) - 190.36 Wh |
| **Minimum consumption hour** | 03:00 (3 AM) - 48.24 Wh |
| **Peak-to-minimum ratio** | 3.95x |

**Behavioral Interpretation:**
- Evening peak (18:00) = dinner preparation and cooking
- Secondary peaks at breakfast (7-8 AM) and lunch (12-13 PM)
- Low consumption during early morning (2-5 AM) = minimal occupancy

---

## 3. Physics-Based Correlation Analysis (Figure 3)

### Thermal Insulation Assessment

| Room | Correlation with T_out | Insulation Level |
|------|------------------------|------------------|
| Kitchen | 0.683 | Moderate |
| Living Room | 0.792 | Low (poor) |
| Laundry Room | 0.699 | Moderate |
| Office Room | 0.663 | Moderate |
| Bathroom | 0.651 | Moderate |
| Outside North | 0.975 | Expected (outdoor) |
| Ironing Room | 0.631 | Moderate |
| Teenager Room | 0.503 | Moderate (best) |
| Parents Room | 0.668 | Moderate |

**Building Assessment:**
- Average correlation: **0.696**
- Result: **LOW thermal insulation** (strong indoor-outdoor coupling)
- Living Room has worst insulation (large windows likely)
- Teenager Room has best insulation among indoor spaces

---

## 4. Feature Importance Analysis (Figure 4)

### Top Features Correlated with Energy Consumption:

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | Outdoor Humidity (RH_out) | -0.152 |
| 2 | Living Room Temp (T2) | +0.120 |
| 3 | Outside North Temp (T6) | +0.118 |
| 4 | Outdoor Temp (T_out) | +0.099 |
| 5 | Teenager Room Humidity (RH8) | -0.094 |

**Key Insight:** Weather conditions (humidity, temperature) have stronger correlation with energy than individual room sensors.

---

## 5. Data Quality & Outlier Analysis (Figure 5)

### Data Quality: ✓ Excellent

| Check | Status |
|-------|--------|
| Missing values | None detected ✓ |
| Humidity range (0-100%) | All valid ✓ |
| Temperature range | All reasonable ✓ |

### Outlier Detection (Appliances)

| Metric | Value |
|--------|-------|
| Q1 (25th percentile) | 50.00 Wh |
| Q3 (75th percentile) | 100.00 Wh |
| IQR | 50.00 Wh |
| Upper bound | 175.00 Wh |
| **Outliers detected** | 2,138 (10.83%) |
| Maximum observed | 1,080 Wh |

**Interpretation:** Outliers represent legitimate high-consumption events (cooking, washing, dishwasher) - NOT data errors.

---

## 6. Weather Impact Analysis (Figure 6)

### Weather-Energy Correlations:

| Weather Variable | Correlation | Impact |
|------------------|-------------|--------|
| Outdoor Humidity | -0.152 | Negative (higher humidity → lower energy) |
| Outdoor Temperature | +0.099 | Weak positive |
| Wind Speed | +0.087 | Weak positive |
| Dew Point | +0.015 | Negligible |
| Visibility | +0.000 | No correlation |
| Pressure | -0.035 | Negligible |

**Key Finding:** Weather has weak influence on appliance energy (HVAC excluded from dataset).

---

## 7. Weekly Patterns Analysis (Figure 7)

### Weekday vs Weekend Comparison:

| Period | Average Energy | Difference |
|--------|----------------|------------|
| Weekday | 96.59 Wh | Baseline |
| Weekend | 100.58 Wh | +4.1% |

**Interpretation:** Slightly higher weekend consumption due to increased home occupancy.

---

## 8. Room Temperature Analysis (Figure 8)

### Temperature Summary by Room:

| Room | Mean Temp | Std Dev |
|------|-----------|---------|
| Laundry Room | 22.3°C | 2.0°C |
| Teenager Room | 22.0°C | 2.0°C |
| Kitchen | 21.7°C | 1.6°C |
| Office Room | 20.9°C | 2.0°C |
| Living Room | 20.3°C | 2.2°C |
| Ironing Room | 20.3°C | 2.1°C |
| Bathroom | 19.6°C | 1.8°C |
| Parents Room | 19.5°C | 2.0°C |
| **Outdoor** | 7.4°C | 5.3°C |

---

## 9. Monthly Trends Analysis (Figure 9)

### Monthly Energy and Temperature:

| Month | Avg Energy | Avg Outdoor Temp |
|-------|------------|------------------|
| January | 97.0 Wh | 4.1°C |
| February | 100.9 Wh | 4.8°C |
| March | 97.0 Wh | 5.4°C |
| April | 98.9 Wh | 8.5°C |
| May | 94.2 Wh | 13.8°C |

**Trend:** Energy consumption remains relatively stable despite warming temperatures (appliances not HVAC-dependent).

---

## Key Scientific Insights for Applied Energy Manuscript

### 1. Energy Consumption Patterns Reflect Occupant Behavior
- Clear diurnal patterns with 4x variation between peak and minimum hours
- Evening peak (18:00) strongly associated with kitchen activities
- Predictable low consumption during early morning hours

### 2. Building Thermal Performance
- Low thermal insulation (average correlation 0.696)
- Living Room shows worst performance (0.792 correlation)
- Energy efficiency improvements should focus on insulation upgrades

### 3. Non-Gaussian Energy Distribution
- Strong right-skewness (3.386) requires special modeling approaches
- Log-transformation or robust regression recommended
- Mean significantly higher than median indicates spike-driven averages

### 4. Weekend vs Weekday Patterns
- 4.1% higher consumption on weekends
- Reflects increased home occupancy and activity

### 5. Data Quality for Modeling
- Clean dataset with no missing values
- 10.83% outliers represent real high-consumption events
- Suitable for machine learning and time-series forecasting

---

## Generated Files

### Tables:
- `table1_statistical_profile.csv` - Complete descriptive statistics

### Figures:
| Figure | Description |
|--------|-------------|
| `figure1_time_series.png` | Sample week time-series (Energy, Indoor/Outdoor Temp) |
| `figure2_hourly_profile.png` | 24-hour average energy profile |
| `figure3_correlation_heatmap.png` | Temperature-Energy correlation matrix |
| `figure4_feature_importance.png` | Feature correlation with energy consumption |
| `figure5_outlier_boxplot.png` | Boxplot and distribution analysis |
| `figure6_weather_impact.png` | Weather variables vs energy scatter plots |
| `figure7_weekly_patterns.png` | Weekly patterns and hour×day heatmap |
| `figure8_room_temperatures.png` | Room-by-room temperature comparison |
| `figure9_monthly_trends.png` | Monthly energy and temperature trends |

---

## Recommendations for Manuscript

1. **Emphasize behavioral patterns** revealed by hourly profiles
2. **Highlight thermal insulation findings** for actionable energy efficiency insights
3. **Address non-Gaussian distribution** in methodology section
4. **Clarify outlier treatment** - these are real events, not anomalies
5. **Connect sensor locations to energy patterns** - especially kitchen correlation

---

*Analysis completed with comprehensive EDA for Applied Energy manuscript submission*
