"""
Main Execution Script: Surrogate-Assisted Optimization for Residential Buildings
Complete pipeline from data curation to results visualization
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from phase1_data_curation import BDG2DataProcessor
from phase2_surrogate_model import SurrogateModel, generate_input_variables_table
from phase3_optimization import BuildingOptimizer, generate_optimization_constraints_table
from phase4_results_visualization import ResultsAnalyzer


def main():
    """Execute complete research pipeline"""
    
    print("="*80)
    print("Surrogate-Assisted Optimization for Residential Buildings")
    print("="*80)
    
    # ========================================================================
    # PHASE 1: Data Curation & Pre-processing
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Data Curation & Pre-processing")
    print("="*80)
    
    processor = BDG2DataProcessor()
    
    # Load and filter metadata
    metadata = processor.load_metadata("metadata.csv")
    residential_buildings = processor.filter_residential_buildings()
    
    # Generate Table 1
    table1 = processor.generate_building_characteristics_table(residential_buildings.head(10))
    print("\nTable 1: Characteristics of Selected Case Study Buildings")
    print("-"*80)
    print(table1.to_string(index=False))
    
    # Process buildings for training
    print("\nProcessing buildings for training...")
    sample_buildings = residential_buildings['building_id'].head(5).tolist()
    results = processor.process_multiple_buildings(sample_buildings, 
                                                   start_date='2020-06-01', 
                                                   end_date='2020-08-31')
    
    print(f"\nTotal processed data points: {len(results['combined_data'])}")
    
    # Select one building for optimization demonstration
    demo_building_id = sample_buildings[0]
    demo_data = results['individual_buildings'][demo_building_id]['data']
    
    # ========================================================================
    # PHASE 2: Surrogate Model Development
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Surrogate Model Development")
    print("="*80)
    
    # Generate Table 2
    table2 = generate_input_variables_table()
    print("\nTable 2: Input Variables for the Prediction Model")
    print("-"*80)
    print(table2.to_string(index=False))
    
    # Train surrogate model
    print("\nTraining surrogate model (XGBoost)...")
    surrogate = SurrogateModel(model_type='xgboost')  # Using XGBoost for faster training
    
    X, y = surrogate.prepare_features(demo_data)
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    metrics = surrogate.train(X, y, validation_split=0.2, epochs=30)
    
    # Save model
    surrogate.save_model("surrogate_model_xgboost.pkl")
    print("\nSurrogate model saved.")
    
    # ========================================================================
    # PHASE 3: Optimization Framework
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Optimization Framework (MPC + Genetic Algorithm)")
    print("="*80)
    
    # Generate Table 3
    table3 = generate_optimization_constraints_table()
    print("\nTable 3: Objective Function & Optimization Constraints")
    print("-"*80)
    print(table3.to_string(index=False))
    
    # Prepare weather forecast for next 24 hours
    forecast = demo_data.tail(24).copy()
    
    # Create optimizer
    print("\nInitializing optimizer...")
    optimizer = BuildingOptimizer(surrogate, forecast, comfort_weight=0.5)
    
    # Run optimization
    print("\nRunning Genetic Algorithm optimization...")
    print("(This may take a few minutes)")
    optimized_results = optimizer.optimize(verbose=True)
    
    # Baseline comparison
    print("\nEvaluating baseline control strategy...")
    baseline_results = optimizer.baseline_control(fixed_setpoint=23.0)
    
    # ========================================================================
    # PHASE 4: Results & Comparative Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Results & Comparative Analysis")
    print("="*80)
    
    # Analyze results
    analyzer = ResultsAnalyzer(baseline_results, optimized_results, forecast)
    
    # Generate Table 4
    table4 = analyzer.generate_comparative_results_table()
    print("\nTable 4: Comparative Results")
    print("-"*80)
    print(table4.to_string(index=False))
    
    # Calculate improvements
    improvements = analyzer.calculate_improvements()
    print("\nImprovement Summary:")
    print(f"  Energy Reduction: {improvements['energy_improvement']:.1f}%")
    print(f"  Cost Reduction: {improvements['cost_improvement']:.1f}%")
    print(f"  Comfort Improvement: {improvements['comfort_improvement']:.1f}%")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_framework_diagram('figure1_framework.png')
    analyzer.plot_daily_optimization_profile('figure2_daily_profile.png')
    analyzer.plot_pareto_front(save_path='figure3_pareto_front.png')
    
    print("\n" + "="*80)
    print("Pipeline execution completed successfully!")
    print("="*80)
    print("\nGenerated outputs:")
    print("  - Table 1: Building characteristics")
    print("  - Table 2: Input variables")
    print("  - Table 3: Optimization constraints")
    print("  - Table 4: Comparative results")
    print("  - Figure 1: Framework diagram (figure1_framework.png)")
    print("  - Figure 2: Daily optimization profile (figure2_daily_profile.png)")
    print("  - Figure 3: Pareto front (figure3_pareto_front.png)")
    print("  - Surrogate model: surrogate_model_xgboost.pkl")
    print("="*80)


if __name__ == "__main__":
    main()
