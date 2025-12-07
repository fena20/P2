"""
Generate Figure 1 - Workflow Schematic for RECS 2020 Heat Pump Retrofit Analysis

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology

This script creates a visual workflow diagram showing all analysis steps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_workflow_diagram():
    """
    Create Figure 1: Overall study workflow schematic
    """
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Define colors
    color_data = '#e8f4f8'
    color_method = '#fff4e6'
    color_output = '#e8f5e9'
    color_arrow = '#666666'
    
    # Helper function to create boxes
    def add_box(x, y, width, height, text, color, fontsize=10, fontweight='normal'):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center',
               fontsize=fontsize, fontweight=fontweight,
               wrap=True)
    
    # Helper function for arrows
    def add_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', 
            mutation_scale=20,
            linewidth=2,
            color=color_arrow
        )
        ax.add_patch(arrow)
    
    # Title
    ax.text(5, 15.5, 'RECS 2020 Heat Pump Retrofit Analysis Workflow',
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Y positions for each step
    y_start = 14
    y_step = 2
    
    # Step 0: Data Source
    y = y_start
    add_box(1, y, 8, 0.8, 'RECS 2020 Microdata\n(EIA Public-Use File)', color_data, 11, 'bold')
    add_arrow(5, y, 5, y - 0.5)
    
    # Step 1: Data Preparation
    y -= y_step
    add_box(1.5, y, 7, 1.2, 
           'Step 1: Data Preparation\n' +
           '• Filter gas-heated homes\n' +
           '• Construct thermal intensity (I = E/A/HDD)\n' +
           '• Engineer features & envelope classes',
           color_method, 9)
    add_arrow(5, y, 5, y - 0.5)
    
    # Output 1
    y -= 0.8
    add_box(1, y-0.3, 3.5, 0.6, 'Output: Cleaned dataset', color_output, 8)
    add_arrow(5, y-0.3, 5, y - 1.0)
    
    # Step 2: Validation
    y -= 1.2
    add_box(1.5, y, 7, 1.0,
           'Step 2: Descriptive Statistics & Validation\n' +
           '• Weighted statistics (NWEIGHT)\n' +
           '• Compare with RECS official tables',
           color_method, 9)
    add_arrow(5, y, 5, y - 0.5)
    
    # Output 2
    y -= 0.8
    add_box(1, y-0.3, 3.5, 0.6, 'Output: Table 2, Figures 2-3', color_output, 8)
    add_arrow(5, y-0.3, 5, y - 1.0)
    
    # Step 3: ML Model
    y -= 1.2
    add_box(1.5, y, 7, 1.0,
           'Step 3: XGBoost Thermal Intensity Model\n' +
           '• Train gradient boosting model\n' +
           '• Evaluate overall & by subgroups',
           color_method, 9)
    add_arrow(5, y, 5, y - 0.5)
    
    # Output 3
    y -= 0.8
    add_box(1, y-0.3, 3.5, 0.6, 'Output: Table 3, Figure 5, Model', color_output, 8)
    add_arrow(5, y-0.3, 5, y - 1.0)
    
    # Step 4: SHAP
    y -= 1.2
    add_box(1.5, y, 7, 1.0,
           'Step 4: SHAP Interpretation\n' +
           '• Compute SHAP values\n' +
           '• Feature importance & dependence',
           color_method, 9)
    add_arrow(5, y, 5, y - 0.5)
    
    # Output 4
    y -= 0.8
    add_box(1, y-0.3, 3.5, 0.6, 'Output: Table 4, Figures 6-7', color_output, 8)
    add_arrow(5, y-0.3, 5, y - 1.0)
    
    # Step 5: Scenarios
    y -= 1.2
    add_box(1.5, y, 7, 1.0,
           'Step 5: Retrofit & Heat Pump Scenarios\n' +
           '• Define measures, costs, performance\n' +
           '• Fuel prices & emissions',
           color_method, 9)
    add_arrow(5, y, 5, y - 0.5)
    
    # Output 5
    y -= 0.8
    add_box(1, y-0.3, 3.5, 0.6, 'Output: Table 5 (a,b,c)', color_output, 8)
    add_arrow(5, y-0.3, 5, y - 1.0)
    
    # Step 6: Optimization
    y -= 1.2
    add_box(1.5, y, 7, 1.0,
           'Step 6: NSGA-II Optimization\n' +
           '• Multi-objective (cost vs emissions)\n' +
           '• Generate Pareto fronts',
           color_method, 9)
    add_arrow(5, y, 5, y - 0.5)
    
    # Output 6
    y -= 0.8
    add_box(1, y-0.3, 3.5, 0.6, 'Output: Table 6, Figure 8', color_output, 8)
    add_arrow(5, y-0.3, 5, y - 1.0)
    
    # Step 7: Tipping Points
    y -= 1.2
    add_box(1.5, y, 7, 1.0,
           'Step 7: Tipping Point Analysis\n' +
           '• Scenario grid (HDD × price × envelope)\n' +
           '• Identify viability zones',
           color_method, 9)
    add_arrow(5, y, 5, y - 0.5)
    
    # Final Output
    y -= 0.8
    add_box(1, y-0.3, 8, 0.7,
           'Final Output: Table 7, Figures 9-10\nPolicy-Ready Tipping Point Maps',
           color_output, 10, 'bold')
    
    # Add legend
    legend_y = 0.5
    add_box(0.5, legend_y, 1.2, 0.4, 'Data', color_data, 8)
    add_box(2, legend_y, 1.2, 0.4, 'Method', color_method, 8)
    add_box(3.5, legend_y, 1.2, 0.4, 'Output', color_output, 8)
    
    plt.tight_layout()
    
    # Save
    output_path = '../recs_output/figures/figure1_workflow_schematic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    print("\n✓ Figure 1 (workflow schematic) created successfully!")
    print("\nInclude this figure in your thesis Methods chapter to show the overall approach.")


def main():
    """
    Main execution
    """
    print("=" * 80)
    print("Generating Figure 1: Workflow Schematic")
    print("=" * 80)
    
    import os
    os.makedirs('../recs_output/figures', exist_ok=True)
    
    create_workflow_diagram()


if __name__ == "__main__":
    main()
