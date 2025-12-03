"""
Quick Demo Script - Generate Tables and Sample Visualizations
Without full training (uses synthetic results)
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from pi_drl_tables import TableGenerator
from pi_drl_visualization import ResultVisualizer

def main():
    print("="*80)
    print("Quick Demo - Generating Tables and Sample Visualizations")
    print("="*80)
    
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    # Generate Tables
    print("\n[1/2] Generating Tables...")
    generator = TableGenerator(save_dir=os.path.join(output_dir, "tables"))
    
    # Table 1: Hyperparameters
    generator.table1_simulation_hyperparameters()
    
    # Table 2: Performance Comparison (with example values)
    generator.table2_performance_comparison(
        baseline_cost=120.50,
        pi_drl_cost=85.30,
        baseline_discomfort=45.2,
        pi_drl_discomfort=28.5,
        baseline_cycles=150,
        pi_drl_cycles=60,
        baseline_peak_load=3.0,
        pi_drl_peak_load=2.1
    )
    
    # Table 3: Ablation Study
    generator.table3_ablation_study(
        pi_drl_with_cycling={'cost': 85.30, 'discomfort': 28.5, 'cycles': 60},
        pi_drl_without_cycling={'cost': 75.20, 'discomfort': 25.0, 'cycles': 450},
        baseline={'cost': 120.50, 'discomfort': 45.2, 'cycles': 150}
    )
    
    # Generate Sample Visualizations
    print("\n[2/2] Generating Sample Visualizations...")
    visualizer = ResultVisualizer(save_dir=os.path.join(output_dir, "figures"))
    
    # Create synthetic data for visualizations
    n_samples = 200
    time_hours = np.arange(n_samples) / 60.0
    
    # PI-DRL: More stable (fewer switches)
    pi_drl_actions = []
    current_action = 0
    for i in range(n_samples):
        if i % 30 == 0:  # Change every 30 minutes (stable)
            current_action = 1 - current_action
        pi_drl_actions.append(current_action)
    pi_drl_actions = np.array(pi_drl_actions)
    pi_drl_temps = 22.0 + 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 60) + np.random.normal(0, 0.3, n_samples)
    
    # Baseline: Frequent switching
    baseline_actions = []
    current_action = 0
    for i in range(n_samples):
        if np.random.random() < 0.1:  # Frequent random switches
            current_action = 1 - current_action
        baseline_actions.append(current_action)
    baseline_actions = np.array(baseline_actions)
    baseline_temps = 22.0 + 1.0 * np.sin(2 * np.pi * np.arange(n_samples) / 60) + np.random.normal(0, 0.5, n_samples)
    
    # Figure 1: System Heartbeat
    visualizer.figure1_system_heartbeat(
        pi_drl_actions, pi_drl_temps,
        baseline_actions, baseline_temps,
        time_hours
    )
    
    # Figure 3: Radar Chart
    baseline_metrics = {
        'energy_cost': 100,
        'comfort_violation': 100,
        'equipment_cycles': 100,
        'peak_load': 100,
        'carbon': 100
    }
    pi_drl_metrics = {
        'energy_cost': 80,
        'comfort_violation': 75,
        'equipment_cycles': 60,
        'peak_load': 70,
        'carbon': 85
    }
    visualizer.figure3_multi_objective_radar(baseline_metrics, pi_drl_metrics)
    
    # Figure 4: Energy Carpet Plot
    pi_drl_power = np.array([3.0 if a == 1 else 0.0 for a in pi_drl_actions])
    baseline_power = np.array([3.0 if a == 1 else 0.0 for a in baseline_actions])
    visualizer.figure4_energy_carpet_plot(baseline_power, pi_drl_power, time_hours)
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print(f"Tables saved to: {os.path.join(output_dir, 'tables')}")
    print(f"Figures saved to: {os.path.join(output_dir, 'figures')}")
    print("="*80)

if __name__ == "__main__":
    main()
