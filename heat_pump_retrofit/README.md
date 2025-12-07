# 🔥 Heat Pump Retrofit Project – RECS 2020 Workflow

> **Author:** Fafa ([GitHub: Fateme9977](https://github.com/Fateme9977))  
> **Institution:** K. N. Toosi University of Technology – Mechanical Engineering / Energy Conversion

---

## 📌 Project Overview

**Title:**  
**Techno-Economic Feasibility and Optimization of Heat Pump Retrofits in Aging U.S. Housing Stock (Using RECS 2020 Microdata)**

**Core Idea:**  
Use **RECS 2020 microdata** + **XGBoost** + **NSGA-II** to identify when (HDD, electricity price, envelope quality) heat pump retrofits become **economically and environmentally preferable** to natural gas heating.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone this repository
cd heat_pump_retrofit

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

The RECS 2020 data should already be in the `data/` folder. If not:

```bash
# Data files required:
# - recs2020_public_v7.csv (microdata)
# - RECS 2020 Codebook for Public File - v7.xlsx
# - HC tables (HC 2.x, HC 6.x, HC 10.x)
# - Methodology documents
```

### 3. Run the Analysis Pipeline

```bash
# Step 1: Data Preparation
python src/01_data_prep.py

# Step 2: Descriptive Statistics and Validation
python src/02_descriptive_validation.py

# Step 3: Train XGBoost Model
python src/03_xgboost_model.py

# Step 4: SHAP Analysis
python src/04_shap_analysis.py

# Step 5: Retrofit Scenarios
python src/05_retrofit_scenarios.py

# Step 6: NSGA-II Optimization (optional - computationally intensive)
python src/06_nsga2_optimization.py

# Step 7: Tipping Point Analysis
python src/07_tipping_point_maps.py
```

---

## 📂 Project Structure

```
heat_pump_retrofit/
├── data/                       # RECS 2020 data files
│   ├── recs2020_public_v7.csv  # Main microdata
│   ├── RECS 2020 Codebook...   # Variable definitions
│   ├── HC 2.*.xlsx             # Housing characteristics tables
│   ├── HC 6.*.xlsx             # Space heating tables
│   ├── HC 10.*.xlsx            # Square footage tables
│   └── *.pdf                   # Methodology documents
│
├── src/                        # Source code
│   ├── 01_data_prep.py         # Data loading and preprocessing
│   ├── 02_descriptive_validation.py  # Statistics and validation
│   ├── 03_xgboost_model.py     # Thermal intensity model
│   ├── 04_shap_analysis.py     # SHAP interpretation
│   ├── 05_retrofit_scenarios.py     # Retrofit definitions
│   ├── 06_nsga2_optimization.py     # Multi-objective optimization
│   └── 07_tipping_point_maps.py     # Viability analysis
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_XGBoost_Experiments.ipynb  # Model experiments
│   └── 03_SHAP_Visualization.ipynb   # SHAP visualizations
│
├── output/                     # Generated outputs
│   ├── figures/                # PNG/PDF figures
│   ├── tables/                 # CSV/LaTeX tables
│   └── models/                 # Saved models
│
├── results/                    # Analysis results
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧪 Methodology

### 1. Data Preparation (`01_data_prep.py`)
- Load RECS 2020 public-use microdata
- Filter for gas-heated homes
- Compute thermal intensity: `I = E_heat / (A_heated × HDD65)`
- Create envelope efficiency classes (poor/medium/good)
- Engineer features for modeling

### 2. Descriptive Validation (`02_descriptive_validation.py`)
- Compute weighted statistics using `NWEIGHT`
- Validate against official RECS HC tables
- Generate Tables 1-2 and Figures 2-4

### 3. XGBoost Model (`03_xgboost_model.py`)
- Train XGBoost regressor for thermal intensity
- 60/20/20 train/val/test split with stratification
- Hyperparameter tuning with cross-validation
- Generate Table 3 and Figure 5

### 4. SHAP Analysis (`04_shap_analysis.py`)
- Compute SHAP values for model interpretation
- Identify key drivers of thermal intensity
- Generate Table 4 and Figures 6-7

### 5. Retrofit Scenarios (`05_retrofit_scenarios.py`)
- Define retrofit measures (air sealing, insulation, windows)
- Define heat pump options (standard, cold-climate)
- Calculate costs and emissions for all combinations
- Generate Table 5

### 6. NSGA-II Optimization (`06_nsga2_optimization.py`)
- Minimize: (1) annualized cost, (2) CO₂ emissions
- Find Pareto-optimal retrofit + HP combinations
- Generate Table 6 and Figure 8

### 7. Tipping Point Analysis (`07_tipping_point_maps.py`)
- Build scenario grid: HDD × electricity price × envelope class
- Identify economic and environmental tipping points
- Generate Table 7 and Figures 9-11

---

## 📊 Key Outputs

### Tables
| Table | Description |
|-------|-------------|
| Table 1 | Variable definitions and sources |
| Table 2 | Sample characteristics by division/envelope |
| Table 3 | XGBoost model performance metrics |
| Table 4 | SHAP feature importance ranking |
| Table 5 | Retrofit and HP assumptions |
| Table 6 | NSGA-II configuration |
| Table 7 | Tipping point summary |

### Figures
| Figure | Description |
|--------|-------------|
| Fig. 1 | Study workflow schematic |
| Fig. 2 | Climate and envelope overview |
| Fig. 3 | Thermal intensity distribution |
| Fig. 4 | Validation against RECS tables |
| Fig. 5 | Predicted vs. observed thermal intensity |
| Fig. 6 | Global SHAP feature importance |
| Fig. 7 | SHAP dependence plots |
| Fig. 8 | Pareto fronts from NSGA-II |
| Fig. 9 | Tipping point heatmaps |
| Fig. 10 | U.S. map of HP viability |
| Fig. 11 | Sensitivity analysis |

---

## 🔑 Key Findings (Expected)

1. **Envelope quality** (draftiness, insulation) is a primary driver of heating intensity
2. Heat pumps become economically viable when:
   - Electricity prices are low relative to gas
   - Buildings are in moderate climates (HDD 3000-5000)
   - Envelope retrofits are bundled
3. Cold climate heat pumps extend viability to colder regions
4. Grid decarbonization significantly improves HP environmental benefits

---

## 📚 Data Sources

### Primary Source (Cite in Publications)
U.S. Energy Information Administration (EIA). 2020 Residential Energy Consumption Survey (RECS) Public-Use Microdata.  
https://www.eia.gov/consumption/residential/data/2020/

### Repository Mirror
https://github.com/Fateme9977/DataR/tree/main/data

---

## 🛠️ Dependencies

- Python 3.9+
- pandas, numpy
- scikit-learn, xgboost
- shap
- matplotlib, seaborn
- openpyxl (for Excel files)
- joblib

See `requirements.txt` for full list with versions.

---

## 📝 Citation

If you use this code or methodology, please cite:

```bibtex
@thesis{fafa2024heatpump,
  title={Techno-Economic Feasibility and Optimization of Heat Pump Retrofits 
         in Aging U.S. Housing Stock Using RECS 2020 Microdata},
  author={Fafa},
  year={2024},
  school={K. N. Toosi University of Technology},
  department={Mechanical Engineering, Energy Conversion}
}
```

---

## 📧 Contact

- **Author:** Fafa
- **GitHub:** [Fateme9977](https://github.com/Fateme9977)
- **Institution:** K. N. Toosi University of Technology

---

## 📄 License

This project is for academic research purposes. Please cite appropriately if using code or methodology.

The RECS 2020 data is public-use data from the U.S. Energy Information Administration and should be cited as the original data source.
