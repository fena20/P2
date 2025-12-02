#!/usr/bin/env python3
"""
Main execution script for the complete research framework
Runs all phases sequentially
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from phase1_data_curation import main as phase1_main
from phase2_surrogate_model import main as phase2_main
from phase3_optimization import main as phase3_main
from phase4_comparative_analysis import main as phase4_main
from generate_visualizations import main as viz_main


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(" " * ((80 - len(text)) // 2) + text)
    print("="*80 + "\n")


def main():
    """Execute complete research framework"""
    start_time = time.time()
    
    print_header("SURROGATE-ASSISTED OPTIMIZATION FOR RESIDENTIAL BUILDINGS")
    print("Complete Research Framework Execution")
    print("="*80)
    
    try:
        # Phase 1: Data Curation & Pre-processing
        print_header("PHASE 1: DATA CURATION & PRE-PROCESSING")
        phase1_main()
        
        # Phase 2: Surrogate Model Development
        print_header("PHASE 2: SURROGATE MODEL DEVELOPMENT")
        phase2_main()
        
        # Phase 3: Optimization Framework
        print_header("PHASE 3: OPTIMIZATION FRAMEWORK")
        phase3_main()
        
        # Phase 4: Comparative Analysis
        print_header("PHASE 4: COMPARATIVE ANALYSIS")
        phase4_main()
        
        # Generate Visualizations
        print_header("GENERATING VISUALIZATIONS")
        viz_main()
        
        # Success summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print_header("EXECUTION COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        
        print("Generated Outputs:")
        print("\n1. Data Files:")
        print("   - data/building_metadata.csv")
        print("   - data/processed_data.csv")
        print("   - data/Res_*.csv (individual building data)")
        
        print("\n2. Models:")
        print("   - results/surrogate_model_lstm/")
        print("   - results/surrogate_model_xgboost/")
        
        print("\n3. Results:")
        print("   - results/optimization_results.json")
        print("   - results/pareto_frontier_data.csv")
        print("   - results/annual_savings_projection.txt")
        
        print("\n4. Tables (for paper):")
        print("   - tables/table1_building_characteristics.csv")
        print("   - tables/table2_input_variables.csv")
        print("   - tables/table3_optimization_parameters.csv")
        print("   - tables/table4_comparative_results.csv")
        print("   - tables/extended_scenario_analysis.csv")
        
        print("\n5. Figures (for paper):")
        print("   - figures/figure1_framework_flowchart.png")
        print("   - figures/figure2_daily_optimization_profile.png")
        print("   - figures/figure3_pareto_front.png")
        
        print("\n" + "="*80)
        print("All outputs are ready for inclusion in your research paper!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
