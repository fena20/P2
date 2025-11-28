"""
Example usage script for the Multi-Objective Building Energy Optimization Framework

This script demonstrates how to use individual components of the framework
for custom workflows.
"""

import pandas as pd
import numpy as np
from src.data_harmonization import DataHarmonizer
from src.feature_engineering import ThermodynamicFeatureEngineer
from src.digital_twin import DigitalTwin
from src.optimization import HVACOptimizationProblem, NSGA2Optimizer
from src.mcdm import TOPSIS, ParetoFrontAnalyzer
from src.utils import load_config

# Load configuration
config = load_config('config.yaml')

# Example 1: Data Harmonization
print("=" * 80)
print("Example 1: Data Harmonization")
print("=" * 80)

# Load sample data
df = pd.read_csv('data/energydata_complete.csv')  # Adjust path as needed

# Create harmonizer
harmonizer = DataHarmonizer(config.get('feature_engineering', {}))

# Harmonize data
df_harmonized = harmonizer.harmonize(df)
print(f"Harmonized data shape: {df_harmonized.shape}")
print(f"Columns: {list(df_harmonized.columns)[:10]}...")  # Show first 10 columns

# Example 2: Feature Engineering
print("\n" + "=" * 80)
print("Example 2: Thermodynamic Feature Engineering")
print("=" * 80)

feature_engineer = ThermodynamicFeatureEngineer(config)
df_engineered = feature_engineer.engineer_features(df_harmonized)
print(f"Engineered features: {[col for col in df_engineered.columns if col not in df_harmonized.columns]}")

# Example 3: Digital Twin Training
print("\n" + "=" * 80)
print("Example 3: Digital Twin Training")
print("=" * 80)

# Prepare data
target_col = 'Appliances' if 'Appliances' in df_engineered.columns else 'E_load'
feature_cols = [col for col in df_engineered.columns 
                if col not in [target_col, 'date', 'Date']]
X = df_engineered[feature_cols].select_dtypes(include=[np.number])
y = df_engineered[target_col]

# Remove missing values
valid_mask = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_mask]
y = y[valid_mask]

# Train Digital Twin
digital_twin = DigitalTwin(config)
metrics = digital_twin.train(X, y, validation_split=0.2)
print(f"Training completed. Validation RMSE: {metrics.get('val_rmse', 'N/A'):.4f}")

# Example 4: Optimization (requires trained Digital Twin)
print("\n" + "=" * 80)
print("Example 4: Multi-Objective Optimization")
print("=" * 80)

# Prepare historical data
historical_data = df_engineered[feature_cols].select_dtypes(include=[np.number])
historical_data = historical_data.ffill().bfill()

# Create optimization problem
problem = HVACOptimizationProblem(
    digital_twin=digital_twin.predict,
    historical_data=historical_data.iloc[:1000],  # Use subset for faster demo
    config=config
)

# Run optimization (with smaller population for demo)
optimizer_config = config.get('optimization', {}).copy()
optimizer_config['population_size'] = 20  # Smaller for demo
optimizer_config['n_generations'] = 10    # Fewer generations for demo

optimizer = NSGA2Optimizer({'optimization': optimizer_config})
res, pareto_front, pareto_solutions = optimizer.optimize(problem, verbose=False)

print(f"Found {len(pareto_front)} Pareto-optimal solutions")
print(f"Energy range: {pareto_front[:, 0].min():.2f} - {pareto_front[:, 0].max():.2f} kWh")
print(f"Discomfort range: {pareto_front[:, 1].min():.2f} - {pareto_front[:, 1].max():.2f}")

# Example 5: MCDM (TOPSIS)
print("\n" + "=" * 80)
print("Example 5: Multi-Criteria Decision Making (TOPSIS)")
print("=" * 80)

topsis = TOPSIS(weights={'energy': 0.5, 'discomfort': 0.5})
best_idx, relative_closeness, topsis_results = topsis.rank_solutions(pareto_front)

optimal_solution = pareto_solutions[best_idx]
optimal_objectives = pareto_front[best_idx]

print(f"Optimal Solution:")
print(f"  T_set_heat: {optimal_solution[0]:.2f} °C")
print(f"  T_set_cool: {optimal_solution[1]:.2f} °C")
print(f"  Energy: {optimal_objectives[0]:.2f} kWh")
print(f"  Discomfort: {optimal_objectives[1]:.2f}")
print(f"  Relative Closeness: {relative_closeness[best_idx]:.4f}")

print("\n" + "=" * 80)
print("Example usage complete!")
print("=" * 80)
