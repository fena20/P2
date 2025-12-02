"""
Utility script to export tables in various formats (CSV, LaTeX, Markdown)
for easy inclusion in research papers
"""

import pandas as pd
from phase1_data_curation import BDG2DataProcessor
from phase2_surrogate_model import generate_input_variables_table
from phase3_optimization import generate_optimization_constraints_table
from phase4_results_visualization import ResultsAnalyzer
from phase2_surrogate_model import SurrogateModel
from phase3_optimization import BuildingOptimizer


def export_table_csv(df: pd.DataFrame, filename: str):
    """Export table to CSV"""
    df.to_csv(filename, index=False)
    print(f"Exported {filename}")


def export_table_latex(df: pd.DataFrame, filename: str):
    """Export table to LaTeX format"""
    latex_str = df.to_latex(index=False, float_format="%.2f", 
                            caption=filename.replace('.tex', '').replace('_', ' ').title(),
                            label=f"tab:{filename.replace('.tex', '').replace('_', '-')}")
    
    with open(filename, 'w') as f:
        f.write(latex_str)
    print(f"Exported {filename}")


def export_table_markdown(df: pd.DataFrame, filename: str):
    """Export table to Markdown format"""
    markdown_str = df.to_markdown(index=False)
    
    with open(filename, 'w') as f:
        f.write(markdown_str)
    print(f"Exported {filename}")


def generate_all_tables():
    """Generate all tables and export in multiple formats"""
    
    print("="*80)
    print("Generating and Exporting All Tables")
    print("="*80)
    
    # Table 1: Building Characteristics
    print("\nGenerating Table 1...")
    processor = BDG2DataProcessor()
    metadata = processor.load_metadata("metadata.csv")
    residential_buildings = processor.filter_residential_buildings()
    table1 = processor.generate_building_characteristics_table(residential_buildings.head(10))
    
    export_table_csv(table1, "table1_building_characteristics.csv")
    export_table_latex(table1, "table1_building_characteristics.tex")
    export_table_markdown(table1, "table1_building_characteristics.md")
    
    # Table 2: Input Variables
    print("\nGenerating Table 2...")
    table2 = generate_input_variables_table()
    
    export_table_csv(table2, "table2_input_variables.csv")
    export_table_latex(table2, "table2_input_variables.tex")
    export_table_markdown(table2, "table2_input_variables.md")
    
    # Table 3: Optimization Constraints
    print("\nGenerating Table 3...")
    table3 = generate_optimization_constraints_table()
    
    export_table_csv(table3, "table3_optimization_constraints.csv")
    export_table_latex(table3, "table3_optimization_constraints.tex")
    export_table_markdown(table3, "table3_optimization_constraints.md")
    
    # Table 4: Comparative Results (requires running optimization)
    print("\nGenerating Table 4 (requires optimization run)...")
    
    # Load data and train model
    building_id = residential_buildings['building_id'].iloc[0]
    df, scalers = processor.process_building(building_id)
    
    # Train surrogate model
    surrogate = SurrogateModel(model_type='xgboost')
    X, y = surrogate.prepare_features(df)
    surrogate.train(X, y, validation_split=0.2, epochs=30)
    
    # Run optimization
    forecast = df.tail(24).copy()
    optimizer = BuildingOptimizer(surrogate, forecast, comfort_weight=0.5)
    optimized_results = optimizer.optimize(verbose=False)
    baseline_results = optimizer.baseline_control(fixed_setpoint=23.0)
    
    # Generate Table 4
    analyzer = ResultsAnalyzer(baseline_results, optimized_results, forecast)
    table4 = analyzer.generate_comparative_results_table()
    
    export_table_csv(table4, "table4_comparative_results.csv")
    export_table_latex(table4, "table4_comparative_results.tex")
    export_table_markdown(table4, "table4_comparative_results.md")
    
    print("\n" + "="*80)
    print("All tables exported successfully!")
    print("="*80)
    print("\nGenerated files:")
    print("  CSV files: table1-4_*.csv")
    print("  LaTeX files: table1-4_*.tex")
    print("  Markdown files: table1-4_*.md")


if __name__ == "__main__":
    generate_all_tables()
