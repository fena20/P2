# Exploratory Data Analysis - Key Insights Summary

## Dataset Overview
- **Total observations:** 19,735 records
- **Time span:** 137 days (January 11, 2016 to May 27, 2016)
- **Sampling frequency:** 10-minute intervals
- **Variables:** 28 features including indoor/outdoor temperatures, humidity, weather data, and energy consumption

---

## 1. Statistical Profiling (Table 1)

### Key Findings:

**Non-Gaussian Distributions (High Skewness):**
- **Appliances (Energy Consumption):** Skewness = 3.386, Kurtosis = 13.664
  - Strongly right-skewed distribution indicating frequent low consumption with occasional high-energy spikes
  - Mean (97.7 Wh) significantly higher than median (60.0 Wh), confirming asymmetric distribution
  
- **Lights:** Skewness = 2.195
  - Most observations show zero or minimal lighting energy consumption
  
- **Bathroom Humidity (RH5):** Skewness = 1.867
  - Higher variability likely due to shower/bath activities

**Temperature Characteristics:**
- Indoor temperatures range from 14.9°C to 29.9°C (comfortable range)
- Outdoor temperature ranges from -5.0°C to 26.1°C (winter to spring transition)
- Kitchen (T1) shows highest average indoor temperature (21.7°C), consistent with cooking activities

---

## 2. Temporal Analysis

### Figure 1: Time-Series Pattern
The sample week visualization reveals:
- **Energy consumption** shows clear diurnal patterns with multiple peaks throughout the day
- **Outdoor temperature** exhibits natural daily cycles with nighttime cooling

### Figure 2: Average Daily Profile (Hourly Patterns)

**Critical Finding: Strong Evening Peak**
- **Peak consumption hour:** 18:00 (6 PM) with 190.36 Wh
- **Minimum consumption hour:** 3:00 (3 AM) with 48.24 Wh
- **Peak-to-minimum ratio:** 3.95x

**Behavioral Interpretation:**
- The evening peak (18:00) aligns with typical dinner preparation and cooking activities
- This corresponds to the **Kitchen temperature sensor (T1)** which shows the highest average indoor temperature
- Secondary peaks likely occur during breakfast (7-8 AM) and lunch (12-13 PM) periods
- Low consumption during early morning hours (2-5 AM) indicates minimal occupancy and appliance use

---

## 3. Physics-Based Correlation Analysis (Figure 3)

### Thermal Insulation Assessment

**Indoor-Outdoor Temperature Correlations:**
- **Average correlation:** 0.696 (moderate to high)
- **Building Assessment:** **LOW thermal insulation** (strong indoor-outdoor coupling)

**Room-Specific Analysis:**
- **Living Room (T2):** Highest correlation (0.792) - likely has large windows or poor insulation
- **Outside North (T6):** Near-perfect correlation (0.975) - expected for outdoor sensor
- **Teenager Room (T8):** Lowest correlation (0.503) - better insulated or less exposed to external conditions
- **Kitchen (T1):** Moderate correlation (0.683) - internal heat sources (appliances) partially decouple from outdoor conditions

**Energy-Temperature Relationships:**
- Strong positive correlation between energy consumption and indoor temperatures during heating periods
- Outdoor temperature shows moderate correlation with energy consumption, indicating HVAC system responsiveness to external conditions

---

## 4. Data Quality & Outlier Analysis

### Data Quality: ✓ Excellent
- **Missing values:** None detected
- **Humidity ranges:** All values within valid 0-100% range
- **Temperature ranges:** All values within physically reasonable bounds

### Outlier Detection (Appliances Energy Consumption)

**IQR Method Results:**
- **Outliers:** 2,138 observations (10.83% of dataset)
- **Upper bound:** 175 Wh (values above this are considered outliers)
- **Maximum observed:** 1,080 Wh (6.2x the median)

**Interpretation:**
- The high-energy spikes (>175 Wh) represent legitimate high-consumption events:
  - Cooking activities (oven, stove, microwave)
  - Washing machine cycles
  - Dishwasher operation
  - Space heating during cold periods
- These outliers are **not data errors** but represent real-world high-energy usage patterns
- The right-skewed distribution (Skewness = 3.386) confirms this pattern

---

## 5. Key Scientific Insights for Applied Energy Manuscript

### 1. **Energy Consumption Patterns Reflect Occupant Behavior**
   - Clear diurnal patterns with 4x variation between peak and minimum hours
   - Evening peak (18:00) strongly associated with kitchen activities
   - Low consumption during early morning indicates predictable occupancy patterns

### 2. **Building Thermal Performance**
   - Low thermal insulation (average correlation 0.696) suggests:
     - Significant heat transfer between indoor and outdoor environments
     - HVAC system must work harder to maintain comfort
     - Energy efficiency improvements could focus on insulation upgrades
   - Room-specific variations indicate heterogeneous building envelope performance

### 3. **Non-Gaussian Energy Distribution**
   - Strong right-skewness (3.386) indicates:
     - Most energy consumption is low-to-moderate
     - Occasional high-energy events drive average consumption upward
     - Predictive models should account for this distribution (e.g., log-transformation, robust regression)

### 4. **Sensor Network Effectiveness**
   - Comprehensive coverage across 9 indoor locations plus weather station
   - Temperature and humidity sensors provide complementary information
   - Kitchen sensor (T1) shows highest correlation with energy consumption patterns

### 5. **Data Quality for Modeling**
   - Clean dataset with no missing values or unrealistic readings
   - 10.83% outliers represent legitimate high-consumption events, not errors
   - Dataset suitable for advanced modeling approaches (machine learning, time-series forecasting)

---

## Recommendations for Manuscript

1. **Emphasize the behavioral patterns** revealed by hourly profiles - this connects energy consumption to human activity
2. **Highlight the thermal insulation findings** - this provides actionable insights for energy efficiency
3. **Address the non-Gaussian distribution** in methodology - explain transformation strategies or robust modeling approaches
4. **Discuss outlier treatment** - clarify that outliers are real events, not anomalies to be removed
5. **Connect sensor locations to energy patterns** - particularly the kitchen sensor correlation with evening peaks

---

*Analysis completed: Comprehensive EDA for Applied Energy manuscript submission*
*Generated files: table1_statistical_profile.csv, figure1_time_series.png, figure2_hourly_profile.png, figure3_correlation_heatmap.png, figure5_outlier_boxplot.png*
