# 🚀 RECS 2020 Heat Pump Retrofit Analysis - Quick Start Guide

> **For:** Fafa (Fateme9977) - Master's Thesis  
> **Ready to use:** ✅ All scripts implemented and documented

---

## 📋 What You Have

### ✅ Complete Analysis Pipeline (7 Scripts)

```
recs_analysis/
├── 01_data_prep.py              # Load & clean RECS data
├── 02_descriptive_validation.py # Statistics & validation
├── 03_xgboost_model.py          # Thermal intensity model
├── 04_shap_analysis.py          # Model interpretation
├── 05_retrofit_scenarios.py     # Define scenarios
├── 06_nsga2_optimization.py     # Multi-objective optimization
├── 07_tipping_point_maps.py     # Tipping point analysis
└── run_complete_pipeline.py     # Master script (runs all)
```

### ✅ Documentation

- **`README_RECS_2020.md`**: Full documentation (16 pages)
- **`RECS_PROJECT_SUMMARY.md`**: Implementation summary
- **`QUICK_START_GUIDE.md`**: This file
- **`requirements_recs.txt`**: Python dependencies

### ✅ Output Structure

```
recs_output/
├── tables/          # All tables (CSV + TXT)
├── figures/         # All figures (PNG, 300 DPI)
├── models/          # Trained XGBoost model
└── *.csv            # Cleaned datasets
```

---

## ⚡ Quick Start (3 Steps)

### Step 1: Get RECS 2020 Data

Download from: https://github.com/Fateme9977/DataR/tree/main/data

**Required file:**
- `recs2020_public_v*.csv` (main microdata)

**Place in:** `/workspace/data/`

```bash
# Create data directory if needed
mkdir -p /workspace/data

# Download RECS microdata (manual or wget/curl)
# Then place in /workspace/data/
```

### Step 2: Install Dependencies

```bash
cd /workspace
pip install -r requirements_recs.txt
```

**Key packages:**
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- shap (model interpretation)
- pymoo (optimization)

### Step 3: Run the Pipeline

```bash
cd /workspace/recs_analysis
python run_complete_pipeline.py
```

**This will:**
1. ✅ Check prerequisites
2. ✅ Run all 7 analysis steps in order
3. ✅ Generate all tables and figures
4. ✅ Create summary reports

**Time:** ~20–40 minutes total

---

## 📊 Expected Outputs

### Tables

| File | Description |
|------|-------------|
| `table2_sample_characteristics.csv` | Sample stats by region/envelope |
| `table3_model_performance.csv` | XGBoost RMSE, MAE, R² |
| `table4_shap_feature_importance.csv` | Feature importance ranking |
| `table5a_retrofit_measures.csv` | Retrofit costs and savings |
| `table5b_heat_pump_options.csv` | HP costs and COP |
| `table5c_fuel_prices_emissions.csv` | Price and emission scenarios |
| `table6_nsga2_configuration.csv` | Optimization settings |
| `table7_tipping_point_summary.csv` | Viability by division |

### Figures

| File | Description |
|------|-------------|
| `figure2_climate_envelope_overview.png` | HDD dist + envelope shares |
| `figure3_thermal_intensity_distribution.png` | Intensity by envelope/climate |
| `figure5_predicted_vs_observed.png` | XGBoost model fit |
| `figure6_global_shap_importance.png` | SHAP feature importance |
| `figure7_shap_dependence.png` | SHAP dependence plots |
| `figure8_pareto_fronts.png` | Cost vs emissions trade-offs |
| `figure9_tipping_point_heatmap.png` | Viability in HDD-price-envelope space |
| `figure10_us_divisions_viability.png` | Regional viability map |

**All figures are 300 DPI, publication-ready!**

---

## 🎯 For Your Thesis

### Methodology Chapter

Use these scripts as your methods:

1. **Section 3.1 - Data:** Describe `01_data_prep.py`
2. **Section 3.2 - Model:** Describe `03_xgboost_model.py`
3. **Section 3.3 - Interpretation:** Describe `04_shap_analysis.py`
4. **Section 3.4 - Optimization:** Describe `06_nsga2_optimization.py`
5. **Section 3.5 - Tipping Points:** Describe `07_tipping_point_maps.py`

### Results Chapter

Present all tables and figures:

- **Table 2**: Sample characteristics
- **Table 3**: Model performance
- **Table 4**: Key drivers (SHAP)
- **Table 5**: Assumptions
- **Table 6**: Optimization setup
- **Table 7**: Tipping points
- **Figures 2–10**: All visualizations

### Discussion Chapter

Interpret results from:
- `table7_tipping_point_summary.csv`
- `final_tipping_point_summary.txt`
- Pareto front CSVs

---

## 🔧 Customization

### Change Assumptions

**Edit `05_retrofit_scenarios.py`:**

```python
# Line ~70: Retrofit costs
self.retrofit_measures = {
    'attic_insulation': {
        'cost_per_sqft': 1.50,  # ← Change this
        'intensity_reduction_pct': 15.0,  # ← Or this
        ...
    }
}

# Line ~110: Fuel prices
self.fuel_prices = {
    'natural_gas': {'medium': 12.0},  # ← Adjust
    'electricity': {'medium': 0.13},  # ← Adjust
}
```

**Then rerun:**

```bash
python 05_retrofit_scenarios.py
python 06_nsga2_optimization.py
python 07_tipping_point_maps.py
```

### Focus on Specific Region

**Edit `01_data_prep.py`, line ~125:**

```python
# After filtering for gas heating, add:
df = df[df['DIVISION'] == 1]  # New England only
```

### Adjust Model Complexity

**Edit `03_xgboost_model.py`, line ~160:**

```python
params = {
    'max_depth': 6,       # ← Increase for more complex model
    'n_estimators': 300,  # ← Increase for better fit
    ...
}
```

---

## 🐛 Troubleshooting

### Issue: "RECS file not found"

**Solution:**
```bash
# Check if file exists
ls /workspace/data/*.csv

# If not, download from:
# https://github.com/Fateme9977/DataR/tree/main/data
```

### Issue: "Missing column: FUELHEAT"

**Solution:**
- Open RECS 2020 codebook
- Find correct variable name
- Edit `01_data_prep.py`, line ~100

### Issue: "SHAP is very slow"

**Solution:**
- Edit `04_shap_analysis.py`, line ~90
- Reduce `max_samples = 1000` to `max_samples = 500`

### Issue: "Import error: pymoo"

**Solution:**
```bash
pip install pymoo
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `README_RECS_2020.md` | Complete documentation (READ THIS FIRST) |
| `RECS_PROJECT_SUMMARY.md` | Implementation summary |
| `QUICK_START_GUIDE.md` | This file |
| Script docstrings | Detailed function documentation |

---

## 🎓 Citation

### For Your Thesis

```
Data Source:
U.S. Energy Information Administration. (2023). 
Residential Energy Consumption Survey (RECS) 2020.
Retrieved from https://www.eia.gov/consumption/residential/data/2020/
```

### For the Code

```
Analysis Pipeline: Fafa (2024)
K. N. Toosi University of Technology
GitHub: Fateme9977
```

---

## ✅ Checklist for Thesis

- [ ] Download RECS 2020 data
- [ ] Install Python dependencies
- [ ] Run complete pipeline
- [ ] Review all outputs (tables + figures)
- [ ] Validate results (compare with RECS official tables)
- [ ] Customize parameters if needed
- [ ] Run sensitivity analysis (vary prices, costs)
- [ ] Write methodology chapter
- [ ] Present results chapter
- [ ] Interpret findings in discussion
- [ ] Prepare appendix with all tables
- [ ] Create supplementary materials (code repository)

---

## 🚀 Next Actions

1. **Today:** Download RECS data → Run pipeline
2. **This week:** Review outputs → Validate results
3. **Next week:** Customize scenarios → Write methods
4. **This month:** Complete results → Draft thesis

---

## 📞 Need Help?

1. **Check `README_RECS_2020.md`** (comprehensive guide)
2. **Read script comments** (detailed inline docs)
3. **Review RECS documentation** (in `data/` folder)
4. **Open GitHub issue** (for bugs)
5. **Contact:** Fafa @ GitHub (Fateme9977)

---

## 🎉 You're All Set!

Everything is ready to run. Just:

1. Get the data
2. Run the pipeline
3. Use the outputs in your thesis

**Good luck with your research! 🎓**

---

*Quick Start Guide v1.0*  
*Created: December 2024*  
*Author: Fafa (Fateme9977)*
