# Multi-Objective Optimization Framework for Building Energy Management

**Target Journal:** Applied Energy (Q1)

## Overview

This repository implements a comprehensive Multi-Objective Optimization framework for building energy management, combining physics-aware feature engineering, a generalizable Digital Twin surrogate model, and NSGA-II optimization with TOPSIS-based decision making.

## Key Contributions

1. **Physics-Aware Digital Twin**: Stacking Ensemble Model serving as a fast surrogate for building thermal physics
2. **Multi-Source Data Harmonization**: Unified feature space across heterogeneous building typologies
3. **Thermodynamic Feature Engineering**: Enthalpy, thermal inertia, and Delta-T features from mechanical engineering perspective
4. **Multi-Objective HVAC Optimization**: NSGA-II optimization for optimal thermostat setpoints
5. **ASHRAE 55-Compliant Discomfort Modeling**: PPD-proxy for thermal comfort assessment
6. **MCDM-Based Solution Selection**: TOPSIS for identifying optimal trade-off solutions

## Methodology

### 1. Multi-Source Data Harmonization & Physics-Invariant Feature Alignment

To enable robust generalization across heterogeneous building typologies (e.g., UCI AEP, Pecan Street, NREL ResStock), a rigorous harmonization protocol is implemented:

- **Schema Standardization**: Maps inconsistent variable identifiers to standardized nomenclature (e.g., 'Outdoor Temp' → `T_out`)
- **Temporal Synchronization**: Resamples time-series to unified frequency (Δt=1h) using aggregation functions
- **Physics-Based Feature Synthesis**: Derives Indoor-Outdoor Temperature Gradient (ΔT) as the primary driving force for heat flux (Fourier's Law: Q ∝ ΔT)

### 2. Thermodynamic Feature Engineering

Physics-based features reflecting thermodynamic states:

- **Enthalpy (h)**: Calculated from Temperature and Relative Humidity to represent total energy content of air
  - Formula: `h = cp_air * T + ω * (hfg + cp_vapor * T)`
- **Thermal Inertia Lag**: Exponential Moving Average (EMA) on indoor temperature representing building envelope heat storage capacity
- **Delta-T**: Difference between Outdoor and Indoor temperature (T_out - T_in), representing driving force for conductive heat transfer

### 3. Generalizable Digital Twin (Surrogate Modeling)

- **Model**: Stacking Ensemble (XGBoost, LightGBM, CatBoost → Ridge meta-learner)
- **Validation**: Cross-Validation across multiple datasets
- **Role**: Fast proxy replacing computationally expensive simulation software (EnergyPlus) in optimization loop

### 4. Multi-Objective Optimization Problem Formulation (NSGA-II)

**Decision Variables:**
- `T_set_heat`: Heating setpoint (lower bound) [18-24°C]
- `T_set_cool`: Cooling setpoint (upper bound) [20-26°C]

**Constraints:**
- Deadband: `T_cool - T_heat ≥ 2°C` (prevents short-cycling)
- Actuator limits: `18°C ≤ T_set ≤ 26°C`

**Objective Functions:**
- **J₁**: Minimize Energy Consumption (`Σ E_pred`)
- **J₂**: Minimize Thermal Discomfort (ASHRAE 55 compliance proxy)
  - `J₂ = Σ(|T_in - T_optimal| + λ · |RH_in - 50%|)`

### 5. Multi-Criteria Decision Making (MCDM)

- **Method**: TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
- **Purpose**: Selects "Knee Point" representing best engineering compromise
- **Output**: Optimal setpoints balancing energy savings and occupant comfort

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pymoo >= 0.6.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- catboost >= 1.2.0
- requests >= 2.31.0
- joblib >= 1.3.0
- tqdm >= 4.65.0
- pyyaml >= 6.0

## Usage

### Quick Start

```bash
python main.py
```

### Configuration

Edit `config.yaml` to customize:
- Data source and paths
- Feature engineering parameters
- Digital Twin model configuration
- Optimization parameters (NSGA-II)
- MCDM weights

### Pipeline Execution

The main script (`main.py`) executes the complete pipeline:

1. **Data Acquisition & Harmonization**: Downloads and harmonizes data
2. **Feature Engineering**: Creates thermodynamic features
3. **Digital Twin Training**: Trains stacking ensemble surrogate model
4. **Optimization**: Runs NSGA-II to find Pareto-optimal solutions
5. **MCDM**: Applies TOPSIS to select optimal solution
6. **Visualization**: Generates Pareto front plots and saves results

### Output

Results are saved to:
- `results/optimization_results.json`: Complete optimization results
- `results/digital_twin_model.pkl`: Trained Digital Twin model
- `figures/pareto_front.png`: Pareto front visualization

## Project Structure

```
.
├── main.py                 # Main execution script
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── data_harmonization.py    # Multi-source data harmonization
│   ├── feature_engineering.py   # Thermodynamic feature engineering
│   ├── digital_twin.py          # Stacking Ensemble Digital Twin
│   ├── optimization.py          # NSGA-II optimization
│   ├── mcdm.py                 # TOPSIS decision making
│   └── utils.py               # Utility functions
├── data/                    # Data directory (created automatically)
├── results/                 # Results directory (created automatically)
└── figures/                 # Figures directory (created automatically)
```

## Scientific Justification

### Data Harmonization

According to **Fourier's Law of Conduction** (Q ∝ ΔT), the temperature difference is the primary driving force for heat flux through the building envelope. By explicitly calculating ΔT, we provide the model with a "common thermodynamic language," allowing it to learn fundamental heat transfer laws irrespective of the specific building's location or baseline temperature.

### Feature Engineering

- **Enthalpy**: Represents total energy content (sensible + latent), crucial for HVAC load analysis
- **Thermal Inertia**: Captures building envelope's heat storage capacity, causing temperature lag behind setpoint changes
- **Delta-T**: Directly represents driving force for conductive heat transfer

### Digital Twin

The stacking ensemble model serves as a computationally efficient surrogate, enabling real-time or near-real-time optimization without requiring expensive simulation software.

### Optimization

NSGA-II is a proven multi-objective evolutionary algorithm capable of finding diverse Pareto-optimal solutions, enabling exploration of the energy-comfort trade-off space.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{building_energy_optimization,
  title={Multi-Objective Optimization Framework for Building Energy Management using Physics-Aware Digital Twins},
  author={Your Name},
  journal={Applied Energy},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- Dataset: [P2 Repository](https://github.com/Fateme9977/P2)
- Optimization Algorithm: NSGA-II (pymoo library)
- Reference: ASHRAE Standard 55 for thermal comfort
