#!/usr/bin/env python
"""
Master script to run the complete RECS 2020 heat pump retrofit analysis pipeline

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology

This script runs all 7 analysis steps in sequence.
Use this for automated execution of the entire workflow.
"""

import sys
import subprocess
from pathlib import Path
import time


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 100)
    print(f"  {text}")
    print("=" * 100 + "\n")


def run_script(script_path, step_number, step_name):
    """
    Run a Python script and handle errors
    
    Parameters
    ----------
    script_path : Path
        Path to the script to run
    step_number : int
        Step number (1-7)
    step_name : str
        Step name for logging
    
    Returns
    -------
    success : bool
        True if script ran successfully
    """
    print_banner(f"STEP {step_number}/7: {step_name}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Step {step_number} completed successfully in {elapsed:.1f} seconds")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR in Step {step_number}:")
        print(e.stdout)
        print(e.stderr)
        
        elapsed = time.time() - start_time
        print(f"\nStep {step_number} failed after {elapsed:.1f} seconds")
        
        return False
    
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR in Step {step_number}:")
        print(str(e))
        return False


def check_prerequisites():
    """Check if required files and directories exist"""
    print_banner("Checking Prerequisites")
    
    issues = []
    
    # Check for data directory
    data_dir = Path('../data')
    if not data_dir.exists():
        issues.append("data/ directory not found")
    
    # Check for RECS microdata file
    if data_dir.exists():
        recs_files = list(data_dir.glob('recs2020_public*.csv'))
        if len(recs_files) == 0:
            issues.append("RECS 2020 microdata CSV not found in data/")
        else:
            print(f"✓ Found RECS microdata: {recs_files[0].name}")
    
    # Check for output directory
    output_dir = Path('../recs_output')
    if not output_dir.exists():
        print("Creating recs_output/ directory...")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for required Python packages
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'xgboost', 'shap', 'pymoo'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"Missing Python packages: {', '.join(missing_packages)}")
        print("\n✗ Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
    else:
        print("✓ All required Python packages are installed")
    
    if issues:
        print("\n✗ Prerequisites check FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease resolve these issues before running the pipeline.")
        print("See README_RECS_2020.md for setup instructions.")
        return False
    
    print("\n✓ All prerequisites satisfied")
    return True


def main():
    """
    Main execution function
    """
    print_banner("RECS 2020 Heat Pump Retrofit Analysis - Complete Pipeline")
    print("Author: Fafa (GitHub: Fateme9977)")
    print("Institution: K. N. Toosi University of Technology")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Define the pipeline steps
    steps = [
        ('01_data_prep.py', 'Data Preparation'),
        ('02_descriptive_validation.py', 'Descriptive Statistics & Validation'),
        ('03_xgboost_model.py', 'XGBoost Model Training'),
        ('04_shap_analysis.py', 'SHAP Interpretation'),
        ('05_retrofit_scenarios.py', 'Retrofit Scenario Definitions'),
        ('06_nsga2_optimization.py', 'NSGA-II Optimization'),
        ('07_tipping_point_maps.py', 'Tipping Point Analysis'),
    ]
    
    # Track overall progress
    overall_start = time.time()
    successful_steps = 0
    
    # Run each step
    for i, (script_name, step_name) in enumerate(steps, 1):
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            print(f"\n✗ ERROR: Script not found: {script_path}")
            break
        
        success = run_script(script_path, i, step_name)
        
        if success:
            successful_steps += 1
        else:
            print(f"\n✗ Pipeline halted at Step {i} due to errors.")
            print("Please review the error messages above and fix the issues.")
            break
        
        # Pause between steps
        if i < len(steps):
            time.sleep(1)
    
    # Final summary
    overall_elapsed = time.time() - overall_start
    
    print_banner("PIPELINE SUMMARY")
    
    print(f"Steps completed: {successful_steps}/{len(steps)}")
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
    
    if successful_steps == len(steps):
        print("\n🎉 SUCCESS! All steps completed successfully!")
        print("\nOutputs are available in: recs_output/")
        print("  - Tables: recs_output/tables/")
        print("  - Figures: recs_output/figures/")
        print("  - Models: recs_output/models/")
        print("\nNext steps:")
        print("  1. Review all outputs")
        print("  2. Validate results against RECS official tables")
        print("  3. Write up results in your thesis/paper")
        print("\nSee README_RECS_2020.md for full documentation.")
    else:
        print("\n✗ Pipeline incomplete. Please fix errors and rerun.")
        sys.exit(1)


if __name__ == "__main__":
    main()
