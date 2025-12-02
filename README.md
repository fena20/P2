# Surrogate-Assisted Optimization for Residential Buildings

## Overview

This project implements a complete research pipeline for optimizing residential building energy consumption using surrogate-assisted optimization. Instead of using slow, physics-based simulations (like EnergyPlus) within the optimization loop, this system uses a fast, deep-learning surrogate model trained on the BDG2 dataset to predict building behavior instantly, allowing a Genetic Algorithm to find optimal solutions in real-time.

## Research Strategy

### Core Concept

The system replaces computationally expensive physics-based simulations with a fast surrogate model (LSTM or XGBoost) that predicts building energy consumption and indoor temperature based on weather conditions and HVAC settings. This enables real-time optimization using Genetic Algorithms.

### Four-Phase Approach

1. **Phase 1: Data Curation & Pre-processing**
   - Filter BDG2 metadata for residential/lodging buildings
   - Merge meter readings with weather data
   - Clean missing values and normalize data

2. **Phase 2: Surrogate Model Development**
   - Train neural network (LSTM) or gradient boosting (XGBoost) model
   - Predict energy consumption and indoor temperature
   - Input: [Outdoor Temp, Solar Radiation, Humidity, Hour, Day of Week, HVAC Setpoint]
   - Output: [Next Hour Energy Consumption, Next Hour Indoor Temp]

3. **Phase 3: Optimization Framework**
   - Genetic Algorithm generates potential thermostat schedules
   - Surrogate model evaluates each schedule instantly
   - Objective: Minimize J = C_energy + w · D_comfort

4. **Phase 4: Results & Comparative Analysis**
   - Compare optimized control against baseline
   - Generate performance metrics and visualizations
   - Calculate energy, cost, and comfort improvements

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline

```bash
python main.py
```

This will execute all four phases and generate:
- Tables 1-4 (printed to console)
- Figure 1: Framework diagram
- Figure 2: Daily optimization profile
- Figure 3: Pareto front
- Trained surrogate model

### Run Individual Phases

```bash
# Phase 1: Data curation
python phase1_data_curation.py

# Phase 2: Surrogate model training
python phase2_surrogate_model.py

# Phase 3: Optimization
python phase3_optimization.py

# Phase 4: Results and visualization
python phase4_results_visualization.py
```

## Project Structure

```
.
├── main.py                          # Main execution script
├── phase1_data_curation.py          # Data preprocessing module
├── phase2_surrogate_model.py        # Surrogate model (LSTM/XGBoost)
├── phase3_optimization.py           # Genetic Algorithm optimizer
├── phase4_results_visualization.py   # Results analysis and plots
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Key Features

### Data Processing
- Automatic filtering of residential buildings
- Weather and meter data merging
- Temporal feature extraction
- Data normalization (0-1 scaling)

### Surrogate Models
- **LSTM**: Sequential model for time-series prediction
- **XGBoost**: Fast gradient boosting for regression
- Multi-output prediction (energy + temperature)

### Optimization
- Genetic Algorithm with customizable parameters
- Multi-objective optimization (cost vs. comfort)
- Time-of-use energy pricing support
- Comfort constraints (PMV approximation)

### Visualizations
- Framework flowchart
- Daily optimization profiles
- Pareto front analysis
- Comparative performance metrics

## Tables Generated

### Table 1: Building Characteristics
Characteristics of selected case study buildings including floor area, climate zone, and year built.

### Table 2: Input Variables
Input features for the prediction model with units, sources, and relevance.

### Table 3: Optimization Constraints
Objective function, decision variables, constraints, and algorithm parameters.

### Table 4: Comparative Results
Performance comparison between baseline and optimized controllers showing improvements.

## Figures Generated

### Figure 1: Framework Diagram
Schematic showing data flow from BDG2 dataset through surrogate model to optimization.

### Figure 2: Daily Optimization Profile
Three-panel plot showing:
- Outdoor temperature
- Optimized vs. baseline setpoint schedule
- Energy consumption comparison

### Figure 3: Pareto Front
Scatter plot demonstrating trade-off between energy cost and resident comfort.

## Expected Results

Based on the optimization framework, typical improvements include:
- **Energy Reduction**: 10-20%
- **Cost Reduction**: 15-25%
- **Comfort Violation Reduction**: 50-80%
- **Computational Time**: 99%+ reduction (seconds vs. hours)

## Configuration

### Model Selection
In `main.py`, change the model type:
```python
surrogate = SurrogateModel(model_type='xgboost')  # or 'lstm'
```

### Optimization Parameters
In `phase3_optimization.py`, adjust:
- `population_size`: Number of GA individuals (default: 50)
- `generations`: Number of GA generations (default: 100)
- `comfort_weight`: Weight for comfort in objective (default: 0.5)

### Energy Pricing
Modify `energy_price` in `BuildingOptimizer` to implement different pricing schemes (flat rate, time-of-use, etc.).

## Notes

- The current implementation uses simulated BDG2 data for demonstration
- For real BDG2 data, modify `load_metadata()`, `load_weather_data()`, and `load_meter_data()` methods
- The surrogate model can be saved and loaded for reuse
- All visualizations are saved as high-resolution PNG files (300 DPI)

## Dependencies

- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- tensorflow >= 2.13.0 (for LSTM)
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- deap >= 1.4.0 (for Genetic Algorithm)

## License

This project is part of a research study on surrogate-assisted building optimization.

## Citation

If you use this code in your research, please cite appropriately and acknowledge the BDG2 dataset.
