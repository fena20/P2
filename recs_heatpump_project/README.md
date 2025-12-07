# Heat Pump Retrofit Project – RECS 2020 Workflow

## Project Overview

This project performs a comprehensive techno-economic feasibility and optimization study of heat pump retrofits in aging U.S. housing stock using RECS 2020 microdata. The analysis combines machine learning (XGBoost), multi-objective optimization (NSGA-II), and scenario analysis to identify conditions where heat pump retrofits become economically and environmentally preferable to natural gas heating.

### Core Objectives

1. **Predict Thermal Intensity**: Use XGBoost to predict building thermal intensity based on building characteristics, envelope quality, and climate.
2. **Interpret Model**: Apply SHAP analysis to understand key drivers of heating energy consumption.
3. **Optimize Retrofit Decisions**: Use NSGA-II to find Pareto-optimal solutions balancing cost and emissions.
4. **Identify Tipping Points**: Map conditions (electricity prices, climate zones, envelope classes) where heat pumps become preferable.

## Project Structure

```
recs_heatpump_project/
├── data/                          # Raw RECS 2020 microdata and codebooks
│   ├── recs2020_public_v3.csv     # Main microdata file
│   ├── RECS 2020 Codebook...xlsx  # Variable definitions
│   └── ...                        # Additional RECS files
├── src/                           # Main workflow scripts
│   ├── 01_data_prep.py           # Data loading, cleaning, feature engineering
│   ├── 02_descriptive_validation.py  # Weighted statistics and validation
│   ├── 03_xgboost_model.py       # XGBoost regression model
│   ├── 04_shap_analysis.py       # SHAP interpretation
│   ├── 05_retrofit_scenarios.py  # Retrofit and HP scenario definitions
│   ├── 06_nsga2_optimization.py  # Multi-objective optimization
│   └── 07_tipping_point_maps.py  # Tipping point analysis and visualization
├── notebooks/                     # Jupyter notebooks for exploration
├── output/                        # Generated outputs
│   ├── figures/                  # All generated plots
│   ├── tables/                   # Generated data tables (CSV)
│   └── models/                   # Saved ML models
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download RECS 2020 Data**:
   - Official source: [EIA RECS 2020 Data](https://www.eia.gov/consumption/residential/data/2020/)
   - Place the microdata CSV file (`recs2020_public_v3.csv`) in the `data/` directory
   - Place the codebook Excel file in the `data/` directory

## Workflow

The project follows a 7-step workflow, with each step implemented as a standalone Python script:

### Step 1: Data Preparation (`01_data_prep.py`)

- Loads RECS 2020 microdata
- Filters to natural gas-heated dwellings
- Constructs heating energy variables
- Calculates **thermal intensity**: `I = E_heat / (A_heated × HDD65)`
- Performs feature engineering (building age, log area, climate zones)
- Defines envelope classes (poor, medium, good)

**Outputs:**
- `output/recs2020_gas_heated_cleaned.csv`

### Step 2: Descriptive Statistics & Validation (`02_descriptive_validation.py`)

- Computes weighted descriptive statistics using `NWEIGHT`
- Validates against official RECS aggregate tables (HC2.x, HC6.x, HC10.x)
- Generates sample characteristics table

**Outputs:**
- `output/validation_report.txt`
- `output/tables/table2_sample_characteristics.csv`

### Step 3: XGBoost Model (`03_xgboost_model.py`)

- Trains XGBoost regressor to predict thermal intensity
- Uses stratified train/validation/test split (60/20/20)
- Evaluates performance (RMSE, MAE, R²) overall and by subgroups
- Saves trained model

**Outputs:**
- `output/models/xgboost_thermal_intensity.pkl`
- `output/tables/table3_xgboost_performance.csv`
- `output/feature_importance.csv`

### Step 4: SHAP Analysis (`04_shap_analysis.py`)

- Computes SHAP values for model interpretation
- Generates global feature importance plots
- Creates dependence plots for key features
- Identifies key drivers of thermal intensity

**Outputs:**
- `output/figures/figure6_shap_global_importance.png`
- `output/figures/figure6_shap_summary.png`
- `output/figures/figure7_shap_dependence.png`
- `output/tables/table4_shap_feature_importance.csv`

### Step 5: Retrofit & Heat Pump Scenarios (`05_retrofit_scenarios.py`)

- Defines retrofit measures (attic insulation, wall insulation, windows, etc.)
- Defines heat pump options (standard vs cold-climate)
- Calculates intensity adjustments based on retrofits
- Computes annualized capital costs
- Computes operational costs (gas vs electricity)
- Computes CO₂ emissions (regional grid + gas factors)
- Creates comprehensive scenario dataframe

**Outputs:**
- `output/tables/table5_retrofit_hp_assumptions.csv`
- `output/retrofit_scenarios.csv`

### Step 6: NSGA-II Optimization (`06_nsga2_optimization.py`)

- Sets up multi-objective optimization problem:
  - **Objectives**: Minimize annualized total cost, minimize annual CO₂ emissions
  - **Decision variables**: Retrofit combinations + heat pump choice
  - **Constraints**: Comfort (meet peak load), optional budget
- Runs NSGA-II algorithm for representative archetypes
- Extracts and visualizes Pareto fronts

**Outputs:**
- `output/tables/table6_nsga2_configuration.csv`
- `output/nsga2_results_summary.csv`
- `output/figures/figure8_pareto_*.png` (one per archetype)

**Note**: Requires `pymoo` library. Install with: `pip install pymoo`

### Step 7: Tipping Point Maps (`07_tipping_point_maps.py`)

- Builds scenario grid (electricity price × HDD bands × envelope classes)
- Identifies conditions where HP retrofits are preferable (cost-effective AND lower emissions)
- Generates visualizations:
  - Tipping point heatmaps
  - Cost vs emissions tradeoff plots
  - Electricity price sensitivity analysis

**Outputs:**
- `output/tipping_point_scenario_grid.csv`
- `output/tables/table7_tipping_points.csv`
- `output/tables/table8_tipping_point_summary.csv`
- `output/figures/figure9_tipping_point_heatmap.png`
- `output/figures/figure10_cost_emissions_tradeoff.png`
- `output/figures/figure11_electricity_price_sensitivity.png`

## Usage

### Running the Complete Workflow

Execute scripts in sequential order:

```bash
cd recs_heatpump_project

# Step 1: Data preparation
python src/01_data_prep.py

# Step 2: Descriptive validation
python src/02_descriptive_validation.py

# Step 3: Train XGBoost model
python src/03_xgboost_model.py

# Step 4: SHAP analysis
python src/04_shap_analysis.py

# Step 5: Define retrofit scenarios
python src/05_retrofit_scenarios.py

# Step 6: NSGA-II optimization
python src/06_nsga2_optimization.py

# Step 7: Tipping point analysis
python src/07_tipping_point_maps.py
```

### Running Individual Steps

Each script can be run independently, but **requires outputs from previous steps**. Check the script docstrings for specific input/output requirements.

## Key Concepts

### Thermal Intensity

Thermal intensity (`I`) is a normalized metric of heating energy consumption:

```
I = E_heat / (A_heated × HDD65)
```

Where:
- `E_heat`: Annual heating energy consumption (BTU)
- `A_heated`: Heated floor area (sqft)
- `HDD65`: Heating degree days base 65°F

This metric allows comparison across different climates and building sizes.

### Envelope Classes

Buildings are categorized into three envelope quality classes based on insulation, windows, and air sealing:

- **Poor**: Minimal insulation, single-pane windows, high air leakage
- **Medium**: Moderate insulation, double-pane windows, moderate air leakage
- **Good**: Well-insulated, efficient windows, low air leakage

### Multi-Objective Optimization

The NSGA-II optimization balances two competing objectives:

1. **Minimize Annualized Total Cost**: Capital costs (annualized) + operational costs
2. **Minimize Annual CO₂ Emissions**: From heating operations

Solutions on the Pareto front represent optimal tradeoffs between cost and emissions.

### Tipping Points

A "tipping point" occurs when a heat pump retrofit becomes:
- **Economically preferable**: Lower or equal total cost compared to gas baseline
- **Environmentally preferable**: Lower CO₂ emissions

Tipping points depend on:
- Electricity prices
- Climate (HDD)
- Building envelope quality
- Fuel prices
- Grid emission factors

## Data Sources

- **RECS 2020 Microdata**: [EIA Residential Energy Consumption Survey 2020](https://www.eia.gov/consumption/residential/data/2020/)
- **Official RECS Tables**: Used for validation (HC2.x, HC6.x, HC10.x)

## Output Files Summary

### Tables
- `table2_sample_characteristics.csv`: Weighted descriptive statistics
- `table3_xgboost_performance.csv`: Model performance metrics
- `table4_shap_feature_importance.csv`: SHAP-based feature importance
- `table5_retrofit_hp_assumptions.csv`: Retrofit and HP assumptions
- `table6_nsga2_configuration.csv`: Optimization configuration
- `table7_tipping_points.csv`: Tipping point conditions
- `table8_tipping_point_summary.csv`: Summary statistics

### Figures
- `figure6_shap_global_importance.png`: SHAP global importance
- `figure6_shap_summary.png`: SHAP summary plot
- `figure7_shap_dependence.png`: SHAP dependence plots
- `figure8_pareto_*.png`: Pareto fronts by archetype
- `figure9_tipping_point_heatmap.png`: Tipping point heatmap
- `figure10_cost_emissions_tradeoff.png`: Cost vs emissions tradeoff
- `figure11_electricity_price_sensitivity.png`: Electricity price sensitivity

### Models
- `xgboost_thermal_intensity.pkl`: Trained XGBoost model

## Configuration

Key parameters can be adjusted in each script:

- **Discount Rate**: `DISCOUNT_RATE = 0.03` (3%)
- **Analysis Period**: `ANALYSIS_PERIOD = 20` (years)
- **Fuel Prices**: Gas and electricity prices (in `05_retrofit_scenarios.py`)
- **Emission Factors**: Regional grid and gas emission factors
- **Retrofit Costs**: Unit costs and lifetimes (in `05_retrofit_scenarios.py`)
- **Heat Pump Performance**: COP values by climate (in `05_retrofit_scenarios.py`)

## Validation Strategy

1. **Internal Validation**: Test-set metrics (RMSE, MAE, R²) and cross-validation
2. **External Validation**: Comparison with official RECS aggregate tables
3. **Sensitivity Analysis**: Scenario grid analysis in Step 7

## Troubleshooting

### Common Issues

1. **Missing Data Files**: Ensure RECS microdata CSV is in `data/` directory
2. **Import Errors**: Install all dependencies from `requirements.txt`
3. **pymoo Not Found**: Install with `pip install pymoo` (required for Step 6)
4. **Memory Issues**: For large datasets, consider sampling or using chunked processing

### Getting Help

- Check script docstrings for detailed function descriptions
- Review output error messages for specific issues
- Ensure all previous steps have completed successfully

## Citation

If you use this project, please cite:

- **RECS 2020 Data**: U.S. Energy Information Administration, Residential Energy Consumption Survey (RECS) 2020
- **XGBoost**: Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
- **SHAP**: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
- **NSGA-II**: Deb et al. (2002), "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"

## License

[Specify your license here]

## Authors

[Your name/affiliation]

## Acknowledgments

- U.S. Energy Information Administration for RECS 2020 data
- Open-source community for excellent Python libraries
