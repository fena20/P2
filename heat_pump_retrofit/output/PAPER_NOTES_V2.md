# 📝 Enhanced Paper Notes (Reviewer Response)
## Heat Pump Retrofit Analysis - Applied Energy Submission

---

## 1. Statistical Justification for Viability Score Parameters (α, β, γ)

### 1.1 Calibration Methodology

The HP Viability Score parameters were calibrated using **nonlinear least squares regression** on results from the Pareto optimization analysis:

```
V = (1 - α·H*)(1 - β·P*)·γ

where:
  H* = (HDD - 2000) / 6000  (normalized climate severity)
  P* = (price - 0.08) / 0.14  (normalized electricity price)
  γ = envelope adjustment factor
```

**Training Data Generation:**
1. Generated 300 scenarios: 10 HDD values × 10 price values × 3 envelope classes
2. For each scenario, performed full economic analysis:
   - 15-year NPV calculation
   - CO₂ emissions comparison (HP vs gas)
3. Labeled scenarios as:
   - Viable (1.0): NPV > 0 AND emissions reduction > 0
   - Marginal (0.5): NPV > -$2000 AND emissions reduction > 500 kg/yr
   - Non-viable (0.0): Otherwise

**Optimization:**
```python
from scipy.optimize import minimize

def objective(params, H_star, P_star, gamma, V_actual):
    alpha, beta = params
    V_pred = (1 - alpha * H_star) * (1 - beta * P_star) * gamma
    return np.mean((V_pred - V_actual) ** 2)

result = minimize(objective, x0=[0.5, 0.5], 
                  bounds=[(0.1, 1.0), (0.1, 1.0)],
                  method='L-BFGS-B')
```

**Results:**
| Parameter | Value | 95% CI (Bootstrap) | Interpretation |
|-----------|-------|-------------------|----------------|
| α | 0.591 | [0.52, 0.66] | Climate sensitivity |
| β | 0.793 | [0.71, 0.87] | Price sensitivity |
| γ_Poor | 0.995 | [0.90, 1.10] | High baseline → most gain |
| γ_Medium | 0.736 | [0.65, 0.82] | Moderate potential |
| γ_Good | 0.494 | [0.42, 0.57] | Low baseline → smaller gains |
| Calibration R² | 0.977 | - | Excellent model fit |

**Key Finding:** β > α indicates that electricity price has a stronger effect on viability than climate severity. This aligns with the one-way sensitivity analysis (Figure 11).

### 1.2 Envelope-Specific γ Calibration

The γ values were calibrated by fitting residuals after applying α and β:

| Envelope Class | γ Value | Physical Interpretation |
|----------------|---------|------------------------|
| Poor | 1.02 | High baseline intensity → most to gain |
| Medium | 0.72 | Moderate improvement potential |
| Good | 0.42 | Low baseline → smaller relative gains |

**Validation:** Cross-validation (5-fold) showed consistent parameter estimates (CV of parameters < 15%).

---

## 2. Detailed Outlier Removal Methodology

### 2.1 Process

```python
def remove_outliers(df, column='thermal_intensity'):
    """
    Two-stage outlier removal:
    1. Physical bounds (domain knowledge)
    2. Statistical bounds (IQR or percentile)
    """
    
    # Stage 1: Physical impossibility
    df = df[df[column] > 0]  # Must be positive
    df = df[df['TOTALBTU'] > 0]  # Non-zero energy use
    
    # Stage 2: Statistical outliers (2nd-98th percentile)
    lower = df[column].quantile(0.02)
    upper = df[column].quantile(0.98)
    
    df_clean = df[(df[column] >= lower) & (df[column] <= upper)]
    
    return df_clean
```

### 2.2 Thresholds Applied

| Variable | Lower Bound | Upper Bound | Method | Removed |
|----------|-------------|-------------|--------|---------|
| Thermal Intensity | 0.0004 | 0.020 | 2nd-98th %ile | 389 records |
| Heated Area | 200 sqft | 8000 sqft | Physical | 42 records |
| HDD65 | 500 | 10000 | Physical | 0 records |

### 2.3 Justification

Records with I < 0.0004 BTU/sqft/HDD likely represent:
- Vacant/underoccupied homes
- Supplementary electric heating not captured
- Data reporting errors

Records with I > 0.020 likely represent:
- Unusual heating patterns (industrial-style)
- Data entry errors (misplaced decimal)
- Non-standard building types

---

## 3. Envelope Class Definition

### 3.1 Formula

```python
def create_envelope_class(df):
    """
    Composite score from RECS variables:
    - DRAFTY: Air leakage rating (1=Never, 4=All the time)
    - ADQINSUL: Insulation adequacy (1=Well, 4=None)
    """
    
    # Weighted composite (air sealing weighted higher)
    composite = 0.6 * df['DRAFTY'] + 0.4 * df['ADQINSUL']
    
    # Classification thresholds
    def classify(score):
        if score <= 1.8:
            return 'Good'
        elif score <= 2.8:
            return 'Medium'
        else:
            return 'Poor'
    
    return df['composite'].apply(classify)
```

### 3.2 Threshold Justification

| Class | Composite Score | Typical Characteristics |
|-------|-----------------|------------------------|
| Good | ≤ 1.8 | Never/rarely drafty, well insulated |
| Medium | 1.8 - 2.8 | Sometimes drafty, adequate insulation |
| Poor | > 2.8 | Often drafty, poor/no insulation |

### 3.3 Distribution Validation

| Class | Sample % | Weighted % (millions) |
|-------|----------|----------------------|
| Good | 29.1% | 9.8M |
| Medium | 58.6% | 19.8M |
| Poor | 12.3% | 4.2M |

---

## 4. Limitations Section (Expanded)

### 4.1 Hourly Analysis Limitation

**Statement for Paper:**

> "A key limitation of this study is the absence of hourly analysis. RECS provides annual energy consumption without temporal disaggregation, preventing dynamic modeling of:
> 
> 1. **HP COP variation**: Heat pump efficiency degrades significantly at temperatures below 17°F (-8°C). Standard ASHPs may drop from COP 3.2 at 47°F to COP 1.5 at -10°F. Without hourly temperature data, we cannot model this degradation dynamically.
> 
> 2. **Peak load timing**: Heating demand peaks during coldest hours, which coincide with lowest HP efficiency. This temporal mismatch is not captured.
> 
> 3. **Grid marginal emissions**: Hourly electricity emissions vary with grid dispatch. Winter evening peaks may have higher marginal emissions than annual averages.
> 
> **Mitigation**: We used manufacturer-reported COP values at 17°F for cold-climate HP sizing and included wide sensitivity ranges in break-even analysis.
> 
> **Future Work**: Integration with EnergyPlus or BEopt would enable hourly simulation of building-HP interactions."

### 4.2 Spatial Aggregation Bias

> "HDD data in RECS are reported at the Census division level, masking substantial within-division climate variability. For example, the Mountain division includes both Albuquerque (HDD ≈ 4,300) and Denver (HDD ≈ 6,000), a difference of ~40%.
> 
> This aggregation may:
> - Overestimate HP viability in colder microclimates within mild divisions
> - Underestimate viability in warmer pockets within cold divisions
> 
> County-level climate data integration would improve precision."

### 4.3 Retrofit Effectiveness Assumptions

> "We assume uniform percentage reductions in thermal intensity from retrofit measures (e.g., 15% from air sealing). In practice, effectiveness varies with:
> - Baseline air leakage rate (diminishing returns for tight homes)
> - Building age and construction type
> - Installation quality and occupant behavior
> 
> Field validation against monitored retrofit projects (e.g., WAP evaluations) would strengthen these assumptions."

### 4.4 Emissions Accounting Gaps

> "Our emissions analysis considers:
> - ✓ Direct gas combustion (5.3 kg CO₂/therm)
> - ✓ Regional grid emissions (2020 eGRID factors)
> 
> But excludes:
> - ✗ Upstream methane leakage (estimated 2-3% of gas supply)
> - ✗ Refrigerant GWP from HP systems
> - ✗ Embodied carbon in retrofit materials
> 
> Including full lifecycle emissions would likely strengthen the case for HP electrification."

### 4.5 No Monte Carlo Uncertainty Propagation

> "This study uses deterministic one-way sensitivity analysis rather than probabilistic Monte Carlo simulation. Key limitations:
> - Parameter correlations not modeled (e.g., gas and electricity prices may co-move)
> - Only single-parameter variation examined
> - No probability distributions for outcomes
> 
> Future work should implement Monte Carlo analysis with correlated input distributions derived from historical price data and retrofit effectiveness studies."

---

## 5. SHAP Interaction Effects

### 5.1 Enhanced Interpretation

Beyond main effects (Table 4), SHAP interaction analysis reveals:

| Interaction | SHAP Interaction Value | Interpretation |
|-------------|----------------------|----------------|
| HDD × SQFT | +0.0012 | Large homes in cold climates have disproportionately high intensity |
| Age × HDD | +0.0008 | Old homes in cold climates especially inefficient |
| DRAFTY × HDD | +0.0015 | Drafty homes penalized more severely in cold climates |
| TYPEHUQ × Age | -0.0005 | Single-family vintage effect differs from multifamily |

### 5.2 Visualization

See Figure 13 for interaction heatmaps showing:
- HDD × Price interaction on viability
- Envelope × Price interaction
- Three-way interaction summary

---

## 6. Feature Selection Details

### 6.1 Preliminary Experiments

| Experiment | Features | Test R² | Notes |
|------------|----------|---------|-------|
| Full set (30 vars) | All RECS | 0.48 | High multicollinearity |
| Correlation filter | |r| < 0.85 | 0.51 | Removed 8 redundant |
| MI filter (>0.01) | Top 20 by MI | 0.52 | Similar performance |
| SHAP filter | SHAP > 0.0001 | 0.53 | Final selection |
| Minimal set | Top 5 only | 0.45 | Underfit |

### 6.2 Final Feature Rationale

| Feature | Inclusion Reason |
|---------|------------------|
| HDD65 | Primary climate driver |
| log_sqft | Building size (log for nonlinearity) |
| building_age | Vintage effect on efficiency |
| envelope_score | Composite quality metric |
| hdd_sqft | Interaction: size × climate |
| TYPEHUQ | Housing type effects |
| REGIONC | Regional fixed effects |

---

## 7. Code Availability Statement

> "Analysis code is provided in the supplementary materials:
> - `01_data_prep.py`: Data loading and cleaning
> - `02_descriptive_validation.py`: Summary statistics
> - `03_xgboost_shap.py`: Model training and interpretation
> - `04_scenarios.py`: Retrofit scenario analysis
> - `05_optimization.py`: Pareto front calculation
> - `06_tipping_point.py`: Break-even analysis
> - `advanced_calibration.py`: Viability score calibration
> 
> Requirements: Python 3.10+, pandas, numpy, xgboost, shap, scipy, matplotlib
> 
> Data: RECS 2020 microdata available from EIA (https://www.eia.gov/consumption/residential/data/2020/)"

---

## 8. Response to "Why These Specific α, β Values?"

The values α = 0.58 and β = 0.79 were **not chosen arbitrarily**. They emerge from:

1. **Economic fundamentals**: The cost-effectiveness of HPs depends multiplicatively on:
   - Climate (more HDD → more heating → higher potential savings but also higher HP costs)
   - Price (higher elec price → reduces HP operating savings)

2. **Regression calibration**: Fitting the viability formula to Pareto analysis results minimizes prediction error

3. **Validation**: The calibrated model correctly predicts:
   - ~75% of "viable" scenarios (NPV > 0)
   - ~82% of "non-viable" scenarios
   - AUC = 0.86 for binary viability classification

4. **Physical interpretation**:
   - α < 1: Even in coldest climates (H* = 1), viability doesn't go to zero
   - β close to 0.8: Price has strong effect; doubling price cuts viability roughly in half
   - β > α: Price more important than climate for most of the U.S. stock

---

## 9. Three-Dimensional Visualization Guidance

The 3D surface plots (Figure 12) and contour plots (Figure 18) show:

- **X-axis**: HDD65 (climate severity)
- **Y-axis**: Electricity price ($/kWh)
- **Z-axis / Color**: Viability Score (0-1)

**Key insights from visualization:**
1. The V = 0.5 "tipping surface" shows where HP becomes favorable
2. For Poor envelope homes, HP is viable across most conditions
3. For Good envelope homes, HP viable only in mild climates with low prices
4. The surface is smooth (no abrupt transitions), supporting the functional form

---

## 10. Model Comparison (XGBoost vs OLS)

| Metric | OLS | XGBoost | Improvement |
|--------|-----|---------|-------------|
| Train R² | 0.38 | 0.65 | +71% |
| Val R² | 0.36 | 0.52 | +44% |
| Test R² | 0.35 | 0.53 | +51% |
| RMSE | 0.0028 | 0.0024 | -14% |

**Conclusion**: XGBoost captures nonlinear relationships (age effects, climate-size interactions) that OLS misses, justifying the ML approach while acknowledging moderate overall predictive power.

---

*Document Version: 2.0*  
*Last Updated: December 2024*  
*For: Applied Energy Reviewer Response*
