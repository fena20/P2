#!/usr/bin/env python3
"""
Verification Script for HVAC PI-DRL Deliverable
Checks that all required outputs exist and are valid.
"""

import os
import sys
from pathlib import Path

def verify_deliverable():
    """Verify all deliverable components."""
    
    print("=" * 70)
    print("HVAC PI-DRL DELIVERABLE VERIFICATION")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Check main script
    print("1. Checking main script...")
    script_path = Path("hvac_pidrl.py")
    if script_path.exists():
        size_kb = script_path.stat().st_size / 1024
        print(f"   ✅ hvac_pidrl.py exists ({size_kb:.1f} KB)")
        
        # Check for key components
        with open(script_path) as f:
            content = f.read()
            
        checks = {
            'if __name__ == "__main__":': "Main execution block",
            'class SafetyHVACEnv': "Environment class",
            'class BaselineThermostat': "Baseline controller",
            'def train_pi_drl': "Training function",
            'def evaluate_controller': "Evaluation function",
            'def create_all_figures': "Figure generation",
            'def create_all_tables': "Table generation",
        }
        
        for key, desc in checks.items():
            if key in content:
                print(f"   ✅ {desc} present")
            else:
                print(f"   ❌ {desc} MISSING")
                all_passed = False
    else:
        print(f"   ❌ hvac_pidrl.py NOT FOUND")
        all_passed = False
    
    print()
    
    # Check output directory
    print("2. Checking output directory...")
    output_dir = Path("output")
    if output_dir.exists():
        print(f"   ✅ output/ directory exists")
    else:
        print(f"   ❌ output/ directory NOT FOUND")
        all_passed = False
        return all_passed
    
    print()
    
    # Check figures
    print("3. Checking figures (7 required)...")
    required_figures = [
        "Figure_1_Micro_Dynamics.png",
        "Figure_2_Safety_Verification.png",
        "Figure_3_Policy_Heatmap.png",
        "Figure_4_Multi_Objective_Radar.png",
        "Figure_5_Robustness.png",
        "Figure_6_Comfort_Distribution.png",
        "Figure_7_Price_Response.png",
    ]
    
    for fig in required_figures:
        fig_path = output_dir / fig
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            print(f"   ✅ {fig} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {fig} NOT FOUND")
            all_passed = False
    
    print()
    
    # Check tables
    print("4. Checking tables (4 required)...")
    required_tables = [
        "Table_1_System_Parameters.csv",
        "Table_2_Performance_Summary.csv",
        "Table_3_Grid_Impact.csv",
        "Table_4_Safety_Shield_Activity.csv",
    ]
    
    for table in required_tables:
        table_path = output_dir / table
        if table_path.exists():
            # Count lines
            with open(table_path) as f:
                lines = len(f.readlines())
            print(f"   ✅ {table} ({lines} rows)")
        else:
            print(f"   ❌ {table} NOT FOUND")
            all_passed = False
    
    print()
    
    # Check documentation
    print("5. Checking documentation...")
    docs = [
        "HVAC_PIDRL_README.md",
        "DELIVERABLE_SUMMARY.md",
    ]
    
    for doc in docs:
        doc_path = Path(doc)
        if doc_path.exists():
            size_kb = doc_path.stat().st_size / 1024
            print(f"   ✅ {doc} ({size_kb:.1f} KB)")
        else:
            print(f"   ⚠️  {doc} not found (optional)")
    
    print()
    
    # Summary
    print("=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Deliverable is complete!")
        print()
        print("To run the pipeline:")
        print("  $ python3 hvac_pidrl.py")
        print()
        print("Expected runtime: 3-5 minutes")
        print("Expected output: 7 PNG figures + 4 CSV tables in output/")
    else:
        print("❌ SOME CHECKS FAILED - Please review above")
        return False
    
    print("=" * 70)
    print()
    
    return all_passed


if __name__ == "__main__":
    success = verify_deliverable()
    sys.exit(0 if success else 1)
