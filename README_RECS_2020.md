# 🔥 Heat Pump Retrofit Project – RECS 2020 Complete Workflow

> **Author:** Fafa ([GitHub: Fateme9977](https://github.com/Fateme9977))  
> **Institution:** K. N. Toosi University of Technology – Mechanical Engineering / Energy Conversion  
> **Project:** Techno-Economic Feasibility and Optimization of Heat Pump Retrofits in Aging U.S. Housing Stock

---

## 📌 Project Overview

This repository contains a complete, production-ready pipeline for analyzing the techno-economic feasibility of heat pump retrofits in U.S. residential buildings using **RECS 2020 microdata**.

### Core Methodology

```
RECS 2020 Microdata 
    ↓
XGBoost Thermal Intensity Model 
    ↓
SHAP Interpretability Analysis 
    ↓
Retrofit & Heat Pump Scenario Modeling 
    ↓
NSGA-II Multi-Objective Optimization 
    ↓
Tipping Point Identification & Mapping
```

### Research Question

**When do heat pump retrofits become economically and environmentally preferable to natural gas heating?**

The answer depends on:
- Climate severity (heating degree days)
- Electricity vs. gas price ratio
- Building envelope quality
- Retrofit investment costs
- Grid emission factors

---

## 📂 Repository Structure

```
project_root/
├── data/                           # RECS 2020 microdata (download separately)
│   ├── recs2020_public_v*.csv     # Main microdata file
│   ├── RECS 2020 Codebook...xlsx  # Variable definitions
│   ├── micro.pdf                   # Microdata guide
│   ├── 2020 RECS_Methodology...pdf
│   └── ... (other RECS documentation)
│
├── recs_analysis/                  # Analysis scripts (run in order)
│   ├── 01_data_prep.py            # Data loading, filtering, feature engineering
│   ├── 02_descriptive_validation.py # Weighted statistics, validation
│   ├── 03_xgboost_model.py        # Thermal intensity prediction model
│   ├── 04_shap_analysis.py        # Model interpretation with SHAP
│   ├── 05_retrofit_scenarios.py   # Retrofit & HP cost/emissions calculations
│   ├── 06_nsga2_optimization.py   # Multi-objective optimization
│   └── 07_tipping_point_maps.py   # Tipping point analysis & visualization
│
├── recs_output/                    # Generated outputs
│   ├── tables/                     # All tables (CSV + formatted text)
│   │   ├── table2_sample_characteristics.csv
│   │   ├── table3_model_performance.csv
│   │   ├── table4_shap_feature_importance.csv
│   │   ├── table5_*_assumptions.csv
│   │   ├── table6_nsga2_configuration.csv
│   │   └── table7_tipping_point_summary.csv
│   ├── figures/                    # All figures (PNG, 300 DPI)
│   │   ├── figure2_climate_envelope_overview.png
│   │   ├── figure3_thermal_intensity_distribution.png
│   │   ├── figure5_predicted_vs_observed.png
│   │   ├── figure6_global_shap_importance.png
│   │   ├── figure7_shap_dependence.png
│   │   ├── figure8_pareto_fronts.png
│   │   ├── figure9_tipping_point_heatmap.png
│   │   └── figure10_us_divisions_viability.png
│   ├── models/                     # Trained models
│   │   └── xgboost_thermal_intensity.json
│   └── recs2020_gas_heated_prepared.csv  # Cleaned dataset
│
├── notebooks/                      # (Optional) Jupyter notebooks for exploration
│
├── requirements_recs.txt           # Python dependencies
└── README_RECS_2020.md            # This file
```

---

## 🌐 Data Sources

### Primary Source: EIA RECS 2020

All data come from the **U.S. Energy Information Administration (EIA)**:

- **RECS 2020 Public-Use Microdata**
- **Housing Characteristics (HC) Tables**
- **Consumption & Expenditures (CE) Tables**
- **Methodology Documentation**

🔗 Official source: https://www.eia.gov/consumption/residential/data/2020/

### Repository Mirror

For convenience, copies of RECS 2020 files are available in:

🔗 https://github.com/Fateme9977/DataR/tree/main/data

> ✅ **Important:** Always cite **EIA RECS 2020** as the original data provider in publications.

---

## 🚀 Quick Start

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/Fateme9977/your-repo-name.git
cd your-repo-name

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_recs.txt
```

### Step 2: Download RECS 2020 Data

Download the following files from the DataR repository or EIA website and place them in the `data/` directory:

**Required:**
- `recs2020_public_v*.csv` (main microdata)
- `RECS 2020 Codebook for Public File - v7.xlsx` (variable definitions)

**Recommended:**
- `micro.pdf` (microdata user guide)
- `2020 RECS_Methodology Report.pdf`
- Other RECS documentation PDFs

### Step 3: Run the Analysis Pipeline

Execute the scripts in order:

```bash
cd recs_analysis

# Step 1: Data preparation
python 01_data_prep.py

# Step 2: Descriptive statistics and validation
python 02_descriptive_validation.py

# Step 3: Train XGBoost model
python 03_xgboost_model.py

# Step 4: SHAP analysis
python 04_shap_analysis.py

# Step 5: Define retrofit scenarios
python 05_retrofit_scenarios.py

# Step 6: NSGA-II optimization
python 06_nsga2_optimization.py

# Step 7: Tipping point maps
python 07_tipping_point_maps.py
```

### Step 4: Review Outputs

All results are saved in `recs_output/`:
- **Tables:** `recs_output/tables/`
- **Figures:** `recs_output/figures/`
- **Models:** `recs_output/models/`

---

## 📊 Key Outputs

### Tables (for thesis/paper)

| Table | Description | Key Content |
|-------|-------------|-------------|
| **Table 1** | Variable definitions | (Create manually from codebook) |
| **Table 2** | Sample characteristics | Weighted stats by region & envelope class |
| **Table 3** | Model performance | RMSE, MAE, R² overall and by subgroups |
| **Table 4** | SHAP feature importance | Ranked list of key drivers |
| **Table 5** | Retrofit & HP assumptions | Costs, lifetimes, performance specs |
| **Table 6** | NSGA-II configuration | Algorithm parameters and scenarios |
| **Table 7** | Tipping point summary | Viability by division & envelope class |

### Figures (publication-ready, 300 DPI)

| Figure | Description |
|--------|-------------|
| **Figure 2** | Climate & envelope overview (HDD distribution + envelope shares) |
| **Figure 3** | Thermal intensity distribution by envelope & climate |
| **Figure 5** | XGBoost predicted vs observed (with R²) |
| **Figure 6** | SHAP global feature importance (bar chart or beeswarm) |
| **Figure 7** | SHAP dependence plots for top 3 features |
| **Figure 8** | Pareto fronts: cost vs. emissions (2 archetypes) |
| **Figure 9** | Tipping point heatmap (HDD × price × envelope) |
| **Figure 10** | U.S. division viability map/chart |

---

## 🧪 Methodology Details

### 1. Data Preparation (`01_data_prep.py`)

- **Input:** RECS 2020 microdata CSV
- **Filters:**
  - Main heating fuel = natural gas
  - Non-missing floor area, HDD, and weights
- **Thermal Intensity Definition:**
  ```
  I = E_heat / (A_heated × HDD65)
  ```
  Where:
  - `E_heat`: Annual heating energy (BTU)
  - `A_heated`: Heated floor area (sqft)
  - `HDD65`: Heating degree days (base 65°F)
- **Envelope Classification:**
  - Based on: draftiness, building age, insulation adequacy
  - Classes: poor, medium, good
- **Output:** Cleaned dataset with engineered features

### 2. Descriptive Statistics & Validation (`02_descriptive_validation.py`)

- Compute **weighted statistics** using `NWEIGHT`
- Generate **Table 2** (sample characteristics)
- Validate against official RECS HC tables
- Create **Figures 2-3** (overview visualizations)

### 3. XGBoost Modeling (`03_xgboost_model.py`)

- **Target:** Thermal intensity `I`
- **Features:** Building characteristics, climate, envelope quality
- **Model:** XGBoost regressor with sample weights
- **Evaluation:** Overall and by region/envelope class
- **Outputs:** 
  - Trained model (JSON)
  - **Table 3** (performance metrics)
  - **Figure 5** (predicted vs observed)

### 4. SHAP Analysis (`04_shap_analysis.py`)

- Compute SHAP values using TreeExplainer
- Identify most influential features
- Create dependence plots to reveal non-linear effects
- **Outputs:**
  - **Table 4** (feature importance ranking)
  - **Figures 6-7** (SHAP visualizations)

### 5. Retrofit Scenarios (`05_retrofit_scenarios.py`)

- Define retrofit measures:
  - None, attic insulation, wall insulation, windows, comprehensive
- Define heat pump options:
  - None (gas), standard ASHP, cold-climate HP, ground-source
- Specify costs, lifetimes, and performance (COP curves)
- Define fuel price scenarios and emission factors
- **Output:** **Table 5** (all assumptions)

### 6. NSGA-II Optimization (`06_nsga2_optimization.py`)

- **Decision Variables:**
  1. Retrofit option (categorical)
  2. Heat pump option (categorical)
- **Objectives:**
  1. Minimize total annualized cost (CapEx + OpEx)
  2. Minimize annual CO₂ emissions
- **Algorithm:** NSGA-II (multi-objective genetic algorithm)
- Run for multiple archetypes (climate × envelope combinations)
- **Outputs:**
  - **Table 6** (algorithm configuration)
  - **Figure 8** (Pareto fronts)
  - Pareto solution CSVs

### 7. Tipping Point Analysis (`07_tipping_point_maps.py`)

- Build scenario grid: HDD × electricity price × envelope class
- For each scenario, compare:
  - Baseline (gas, no retrofit)
  - HP retrofit (e.g., standard ASHP, no envelope retrofit)
- Identify conditions where HP is:
  - **Viable:** Cost-competitive AND lower emissions
  - **Emissions-only:** Lower emissions but higher cost
  - **Not viable:** Neither cost-competitive nor lower emissions
- **Outputs:**
  - **Table 7** (tipping points by division)
  - **Figure 9** (heatmap)
  - **Figure 10** (U.S. map)

---

## 🔬 Key Assumptions

### Economic Parameters

- **Discount rate:** 5%
- **Analysis period:** 20 years
- **Natural gas price (medium):** $12/MMBtu
- **Electricity price (medium):** $0.13/kWh
- **Retrofit costs:** $1.50–$12/sqft (varies by measure)
- **Heat pump costs:** $3,500–$8,000 per ton

### Technical Parameters

- **Heat pump sizing:** ~1 ton per 600 sqft (rule of thumb)
- **Heat pump COP:**
  - Standard ASHP: 3.5 @ 47°F, 2.0 @ 17°F
  - Cold-climate HP: 3.8 @ 47°F, 2.5 @ 17°F
- **Retrofit intensity reductions:**
  - Attic insulation: 15%
  - Wall insulation: 20%
  - Windows: 12%
  - Comprehensive: 45%

### Emission Factors

- **Natural gas:** 53.06 kg CO₂/MMBtu (EPA)
- **Electricity (U.S. avg):** 0.386 kg CO₂/kWh (EPA eGRID 2021)

> **Note:** All assumptions are documented in the output tables for transparency and can be adjusted as needed.

---

## 📈 Expected Results

### Key Findings

1. **Climate Dependence:**
   - Mild climates (HDD < 3,500): HP retrofits often viable at current prices
   - Cold climates (HDD > 6,000): Envelope improvements needed first

2. **Envelope Quality Matters:**
   - Poor envelopes have higher cost barriers but greater savings potential
   - Good envelopes enable HP viability even in colder climates

3. **Price Sensitivity:**
   - Electricity/gas price ratio is critical
   - Tipping point typically at $0.12–$0.16/kWh for moderate climates

4. **Emission Benefits:**
   - HP retrofits reduce CO₂ by 30–60% even with current grid mix
   - Benefits increase as grid decarbonizes

### Policy Implications

- **Targeted incentives** needed for cold climate + poor envelope homes
- **Envelope-first** strategy in harsh climates
- **Direct HP adoption** viable in mild climates
- **Rate design** (electricity pricing) affects viability
- **Regional heterogeneity** requires customized policies

---

## 🛠 Customization Guide

### Adjusting Parameters

All key parameters are defined at the top of each script and can be easily modified:

**In `05_retrofit_scenarios.py`:**
```python
# Modify retrofit costs
self.retrofit_measures = {
    'attic_insulation': {
        'cost_per_sqft': 1.50,  # ← Change this
        'intensity_reduction_pct': 15.0,  # ← Or this
        ...
    }
}

# Modify fuel prices
self.fuel_prices = {
    'natural_gas': {'medium': 12.0},  # ← Adjust scenarios
    'electricity': {'medium': 0.13},
}
```

**In `06_nsga2_optimization.py`:**
```python
# Modify NSGA-II settings
algorithm = NSGA2(
    pop_size=100,  # ← Population size
    ...
)
termination = get_termination("n_gen", 100)  # ← Generations
```

### Adding New Scenarios

To add new retrofit measures or heat pump types:

1. Edit `05_retrofit_scenarios.py`
2. Add entries to `self.retrofit_measures` or `self.heat_pump_options`
3. Rerun scripts 05, 06, 07

### Regional Analysis

To focus on specific census divisions:

1. In `01_data_prep.py`, filter by `DIVISION` variable
2. Adjust HDD ranges in `07_tipping_point_maps.py`

---

## 📚 Citation

### For the Project

```bibtex
@mastersthesis{fafa2024heatpump,
  author = {Fafa},
  title = {Techno-Economic Feasibility and Optimization of Heat Pump Retrofits in Aging U.S. Housing Stock},
  school = {K. N. Toosi University of Technology},
  year = {2024},
  note = {GitHub: \url{https://github.com/Fateme9977}}
}
```

### For RECS 2020 Data (Required)

```bibtex
@techreport{eia2023recs,
  author = {{U.S. Energy Information Administration}},
  title = {2020 Residential Energy Consumption Survey (RECS)},
  institution = {U.S. Department of Energy},
  year = {2023},
  url = {https://www.eia.gov/consumption/residential/data/2020/}
}
```

---

## 🤝 Contributing

This project is part of a master's thesis. Suggestions and improvements are welcome!

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes
4. Push and create a Pull Request

---

## 📞 Contact

**Author:** Fafa  
**GitHub:** [Fateme9977](https://github.com/Fateme9977)  
**Institution:** K. N. Toosi University of Technology  
**Department:** Mechanical Engineering / Energy Conversion  

---

## 📄 License

This project is open-source and available under the MIT License (or specify your preferred license).

**Data License:** RECS 2020 data is public domain (U.S. government work).

---

## 🎓 Acknowledgments

- **U.S. Energy Information Administration (EIA)** for providing RECS 2020 microdata
- **K. N. Toosi University of Technology** for academic support
- Open-source communities for:
  - XGBoost (machine learning)
  - SHAP (model interpretation)
  - pymoo (multi-objective optimization)
  - pandas, numpy, matplotlib, seaborn (data science stack)

---

## 🔄 Version History

- **v1.0** (2024-12): Initial complete pipeline release
  - All 7 analysis scripts
  - Full table and figure generation
  - Comprehensive documentation

---

## 🚨 Troubleshooting

### Common Issues

**1. "RECS microdata file not found"**
- Ensure `recs2020_public_v*.csv` is in the `data/` directory
- Check filename matches the pattern in `01_data_prep.py`

**2. "Missing column: FUELHEAT"**
- Variable names may differ between RECS versions
- Consult the RECS 2020 Codebook and update variable names in scripts

**3. "SHAP computation very slow"**
- Reduce sample size in `04_shap_analysis.py` (line: `max_samples = 1000`)
- SHAP computation scales with sample size × features

**4. "NSGA-II not converging"**
- Increase population size or generations in `06_nsga2_optimization.py`
- Check that objective functions return valid values

**5. "Figures look different than examples"**
- This is expected! Your results depend on:
  - RECS microdata version
  - Random seeds
  - Parameter choices

### Getting Help

1. Check script docstrings and comments
2. Review RECS 2020 documentation in `data/`
3. Open an issue on GitHub
4. Contact the author (see Contact section)

---

## 🎯 Next Steps After Completing Analysis

1. **Review all outputs** in `recs_output/`
2. **Validate results** against literature and official RECS tables
3. **Sensitivity analysis:** Vary key assumptions and rerun
4. **Write up results** in thesis/paper:
   - Introduction: Literature review + research questions
   - Methods: Describe workflow (use Figure 1 schematic)
   - Results: Present Tables 2-7 and Figures 2-10
   - Discussion: Interpret tipping points and policy implications
   - Conclusion: Summarize findings and future work
5. **Prepare supplementary materials:**
   - Code repository (GitHub)
   - Data availability statement (cite EIA RECS 2020)
   - Assumption tables for reproducibility

---

## 🌟 Project Highlights

✅ **Complete end-to-end pipeline**  
✅ **Publication-ready tables and figures**  
✅ **Reproducible and well-documented**  
✅ **Flexible and customizable**  
✅ **Uses best-practice methods** (XGBoost, SHAP, NSGA-II)  
✅ **Validated against official RECS statistics**  
✅ **Policy-relevant tipping point analysis**  

---

**🎉 Thank you for using this analysis pipeline! Good luck with your thesis! 🎉**

---

*Last updated: December 2024*
