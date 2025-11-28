# Methodology Documentation

## Multi-Objective Optimization Framework for Building Energy Management

### 1. Multi-Source Data Harmonization & Physics-Invariant Feature Alignment

#### 1.1 Schema Standardization

**Objective**: Create a dataset-agnostic input vector for the surrogate model.

**Method**: Maps inconsistent variable identifiers across datasets to standardized nomenclature:
- Temperature: `T_out` (outdoor), `T_in` (indoor)
- Humidity: `RH_out`, `RH_in`
- Energy: `E_load` (or `Appliances`)

**Scientific Justification**: Enables the model to learn from multiple building typologies without manual feature mapping.

#### 1.2 Temporal Synchronization

**Objective**: Align dynamic response characteristics of different buildings and eliminate sampling rate bias.

**Method**: 
- Resample all time-series to unified frequency (Δt=1h)
- Use mean aggregation for continuous variables (Temperature, Humidity)
- Use sum aggregation for cumulative variables (Energy)

**Implementation**: `DataHarmonizer.synchronize_temporal()`

#### 1.3 Physics-Based Feature Synthesis: Delta-T Gradient

**Formula**: ΔT(t) = T_in(t) - T_out(t)

**Scientific Justification**: 
According to **Fourier's Law of Conduction** (Q ∝ ΔT), the temperature difference is the primary driving force for heat flux through the building envelope. By explicitly calculating ΔT, we provide the model with a "common thermodynamic language," allowing it to learn fundamental heat transfer laws irrespective of the specific building's location or baseline temperature.

**Mechanical Context**: This ensures the model learns physics, not just site-specific correlations.

**Implementation**: `DataHarmonizer.compute_delta_t()`

### 2. Thermodynamic Feature Engineering

#### 2.1 Enthalpy Calculation

**Formula**: 
```
h = cp_air * T + ω * (hfg + cp_vapor * T)
```

Where:
- `cp_air` = 1.006 kJ/(kg·K) - Specific heat of air
- `hfg` = 2501.0 kJ/kg - Latent heat of vaporization
- `ω` = Humidity ratio (kg water vapor / kg dry air)
- `cp_vapor` = 1.86 kJ/(kg·K) - Specific heat of water vapor

**Purpose**: Represents total energy content of air (sensible + latent), crucial for latent vs. sensible load analysis in HVAC systems.

**Implementation**: `ThermodynamicFeatureEngineer.calculate_enthalpy()`

#### 2.2 Thermal Inertia Lag

**Method**: Exponential Moving Average (EMA) on indoor temperature

**Formula**: 
```
T_inertia(t) = α * T_in(t) + (1 - α) * T_inertia(t-1)
```

Where α is the smoothing factor (default: 0.3).

**Purpose**: Represents the building envelope's heat storage capacity (mass). Thermal inertia causes temperature to lag behind setpoint changes, which is critical for HVAC control optimization.

**Implementation**: `ThermodynamicFeatureEngineer.compute_thermal_inertia()`

#### 2.3 Delta-T (Temperature Difference)

**Formula**: ΔT = T_out - T_in

**Purpose**: Represents the driving force of conductive heat transfer through the building envelope (Fourier's Law: Q ∝ ΔT).

**Implementation**: `ThermodynamicFeatureEngineer.compute_delta_t()`

### 3. Generalizable Digital Twin (Surrogate Modeling)

#### 3.1 Model Architecture

**Type**: Stacking Ensemble

**Base Models**:
1. XGBoost Regressor
2. LightGBM Regressor
3. CatBoost Regressor (optional)

**Meta-Model**: Ridge Regression

**Validation**: K-Fold Cross-Validation (default: 5 folds)

#### 3.2 Training Process

1. Train base models on training data
2. Generate out-of-fold predictions from base models
3. Train meta-model on base model predictions
4. Evaluate using cross-validation metrics (RMSE, MAE, R²)

#### 3.3 Role in Optimization

The Digital Twin serves as a fast, accurate proxy for building thermal physics, replacing computationally expensive simulation software (like EnergyPlus) in the optimization loop. This enables real-time or near-real-time optimization.

**Implementation**: `DigitalTwin` class

### 4. Multi-Objective Optimization Problem Formulation

#### 4.1 Decision Variables

- **T_set_heat**: Heating setpoint (lower bound) [18-24°C]
- **T_set_cool**: Cooling setpoint (upper bound) [20-26°C]

#### 4.2 Constraints

1. **Deadband Constraint**: T_cool - T_heat ≥ 2°C
   - Prevents short-cycling of HVAC system
   - Ensures minimum separation between heating and cooling setpoints

2. **Actuator Limits**: 18°C ≤ T_set ≤ 26°C
   - System operational range
   - Safety constraints preventing freezing or overheating risks

#### 4.3 Objective Functions

**J₁: Minimize Energy Consumption**
```
J₁ = Σ E_pred
```
Where E_pred is predicted energy consumption from the Digital Twin.

**J₂: Minimize Thermal Discomfort (ASHRAE 55 Compliance Proxy)**
```
J₂ = Σ(|T_in - T_optimal| + λ · |RH_in - 50%|)
```

Where:
- T_optimal = 22°C (optimal indoor temperature)
- λ = 0.5 (relative weight for humidity discomfort)
- 50% = Optimal relative humidity (ASHRAE Standard 55)

**Mechanical Context**: This ensures the solution considers both sensible heat (Temperature) and latent comfort (Humidity).

#### 4.4 Optimization Algorithm: NSGA-II

**Algorithm**: Non-dominated Sorting Genetic Algorithm II

**Parameters**:
- Population size: 50 (default)
- Generations: 100 (default)
- Crossover: SBX (Simulated Binary Crossover), prob=0.9, η=15
- Mutation: PM (Polynomial Mutation), η=20

**Output**: Pareto-optimal front of solutions

**Implementation**: `NSGA2Optimizer` and `HVACOptimizationProblem` classes

### 5. Multi-Criteria Decision Making (MCDM)

#### 5.1 TOPSIS Method

**Full Name**: Technique for Order of Preference by Similarity to Ideal Solution

**Steps**:
1. Normalize decision matrix (vector normalization)
2. Calculate ideal and negative-ideal solutions
3. Calculate weighted distances to ideal and negative-ideal
4. Calculate relative closeness: C = d⁻ / (d⁺ + d⁻)
5. Select solution with highest relative closeness

#### 5.2 Knee Point Identification

**Method**: Distance-based knee point detection

**Purpose**: Identifies solution with maximum curvature change on Pareto front, representing the best trade-off between objectives.

**Mechanical Interpretation**: The knee point represents the "Best Engineering Compromise" where significant energy is saved with negligible loss in occupant comfort.

**Implementation**: `ParetoFrontAnalyzer.find_knee_point()`

### 6. Generalization & Robustness Verification

#### 6.1 Scenario Testing

**Method**: Apply optimized control setpoints derived from Training Data (Datasets 1-3) to Test Data (Datasets 4-6).

**Objective**: Demonstrate that the optimized strategy reduces energy across different building types compared to fixed control strategies.

#### 6.2 Validation Metrics

- **Energy Reduction**: Percentage reduction compared to baseline
- **Comfort Maintenance**: Discomfort index comparison
- **Generalization Error**: Performance on unseen building types

## References

1. **Fourier's Law of Conduction**: Q = -k·A·(dT/dx)
2. **ASHRAE Standard 55**: Thermal Environmental Conditions for Human Occupancy
3. **NSGA-II**: Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
4. **TOPSIS**: Hwang, C.L., & Yoon, K. (1981). "Multiple Attribute Decision Making: Methods and Applications"
