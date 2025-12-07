"""
Heat Pump Retrofit Analysis Package
====================================

This package contains modules for analyzing techno-economic feasibility
and optimization of heat pump retrofits in U.S. housing stock using
RECS 2020 microdata.

Modules:
--------
- 01_data_prep: Data loading and preprocessing
- 02_descriptive_validation: Descriptive statistics and validation
- 03_xgboost_model: XGBoost thermal intensity model
- 04_shap_analysis: SHAP interpretation
- 05_retrofit_scenarios: Retrofit and heat pump scenario definitions
- 06_nsga2_optimization: Multi-objective optimization with NSGA-II
- 07_tipping_point_maps: Tipping point analysis and mapping

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

__version__ = "1.0.0"
__author__ = "Fafa"

# Allow importing from submodules
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"
