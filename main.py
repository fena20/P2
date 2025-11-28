"""
Multi-Objective Optimization Framework for Building Energy Management

Main execution script for the complete pipeline:
1. Data Acquisition & Harmonization
2. Thermodynamic Feature Engineering
3. Digital Twin Development (Stacking Ensemble)
4. Multi-Objective Optimization (NSGA-II)
5. Multi-Criteria Decision Making (TOPSIS)
6. Generalization & Robustness Verification

Target Journal: Applied Energy (Q1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Dict
warnings.filterwarnings('ignore')

from src.data_harmonization import DataHarmonizer, load_and_harmonize_data
from src.feature_engineering import ThermodynamicFeatureEngineer
from src.digital_twin import DigitalTwin
from src.optimization import HVACOptimizationProblem, NSGA2Optimizer
from src.mcdm import TOPSIS, ParetoFrontAnalyzer
from src.utils import (
    load_config, ensure_dir, download_data,
    plot_pareto_front, save_results
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BuildingEnergyOptimizationPipeline:
    """
    Complete pipeline for Multi-Objective Building Energy Optimization.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_harmonizer = None
        self.feature_engineer = None
        self.digital_twin = None
        self.optimizer = None
        self.topsis = None
        
        # Create output directories
        results_dir = self.config.get('output', {}).get('results_dir', 'results')
        figures_dir = self.config.get('output', {}).get('figures_dir', 'figures')
        ensure_dir(results_dir)
        ensure_dir(figures_dir)
        ensure_dir('data')
        
    def step1_data_acquisition(self) -> pd.DataFrame:
        """
        Step 1: Data Acquisition & Harmonization
        
        Downloads and harmonizes data from multiple sources to create
        a unified feature space.
        """
        logger.info("=" * 80)
        logger.info("STEP 1: DATA ACQUISITION & HARMONIZATION")
        logger.info("=" * 80)
        
        data_config = self.config.get('data', {})
        source_url = data_config.get('source_url')
        local_path = data_config.get('local_path', 'data/energydata_complete.csv')
        
        # Download data if needed
        if source_url:
            download_data(source_url, local_path)
        
        # Load and harmonize data
        logger.info("Harmonizing data...")
        self.data_harmonizer = DataHarmonizer(self.config.get('feature_engineering', {}))
        df = pd.read_csv(local_path)
        
        # CRITICAL FIX: Explicitly coerce all columns to numeric types (except datetime)
        # This prevents 'agg function failed' errors during resampling
        datetime_col = 'date'  # Default datetime column name
        if datetime_col in df.columns:
            # Keep datetime column as is for now
            pass
        
        # Coerce all other columns to numeric, handling errors gracefully
        for col in df.columns:
            if col != datetime_col and col.lower() != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_harmonized = self.data_harmonizer.harmonize(df, datetime_col=datetime_col)
        
        logger.info(f"Data harmonization complete. Shape: {df_harmonized.shape}")
        return df_harmonized
    
    def step2_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Thermodynamic Feature Engineering
        
        Engineers physics-aware features:
        - Enthalpy (h): Air enthalpy from T and RH
        - Thermal Inertia: EMA on indoor temperature
        - Delta-T: Indoor-Outdoor temperature difference
        """
        logger.info("=" * 80)
        logger.info("STEP 2: THERMODYNAMIC FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        logger.info("Engineering thermodynamic features...")
        self.feature_engineer = ThermodynamicFeatureEngineer(self.config)
        df_engineered = self.feature_engineer.engineer_features(df)
        
        logger.info(f"Feature engineering complete. Shape: {df_engineered.shape}")
        logger.info(f"Features: {list(df_engineered.columns)}")
        
        return df_engineered
    
    def step3_digital_twin(self, df: pd.DataFrame) -> DigitalTwin:
        """
        Step 3: Digital Twin Development
        
        Trains a Stacking Ensemble Model as a fast surrogate for building
        thermal physics, validated via Cross-Validation.
        """
        logger.info("=" * 80)
        logger.info("STEP 3: DIGITAL TWIN DEVELOPMENT")
        logger.info("=" * 80)
        
        # Prepare features and target
        target_col = self.config.get('data', {}).get('target_column', 'Appliances')
        
        # Find target column (handle different naming)
        if target_col not in df.columns:
            if 'Appliances' in df.columns:
                target_col = 'Appliances'
            elif 'E_load' in df.columns:
                target_col = 'E_load'
            else:
                raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Select feature columns (exclude target and datetime)
        exclude_cols = [target_col, 'date', 'Date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Remove rows with missing values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Train Digital Twin
        self.digital_twin = DigitalTwin(self.config)
        metrics = self.digital_twin.train(X, y, validation_split=0.2)
        
        # Evaluate on full dataset
        logger.info("Evaluating Digital Twin...")
        eval_metrics = self.digital_twin.evaluate(X, y)
        
        # Save model
        model_path = 'results/digital_twin_model.pkl'
        self.digital_twin.save(model_path)
        
        logger.info("Digital Twin training complete.")
        return self.digital_twin
    
    def step4_optimization(self, df: pd.DataFrame) -> tuple:
        """
        Step 4: Multi-Objective Optimization (NSGA-II)
        
        Optimizes HVAC control setpoints to minimize:
        - Energy Consumption (J1)
        - Thermal Discomfort (J2)
        """
        logger.info("=" * 80)
        logger.info("STEP 4: MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)")
        logger.info("=" * 80)
        
        # Prepare historical data for simulation
        target_col = self.config.get('data', {}).get('target_column', 'Appliances')
        exclude_cols = [target_col, 'date', 'Date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        historical_data = df[feature_cols].select_dtypes(include=[np.number])
        historical_data = historical_data.ffill().bfill()
        
        # Create optimization problem
        problem = HVACOptimizationProblem(
            digital_twin=self.digital_twin.predict,
            historical_data=historical_data,
            config=self.config
        )
        
        # Run optimization
        self.optimizer = NSGA2Optimizer(self.config)
        res, pareto_front, pareto_solutions = self.optimizer.optimize(problem, verbose=True)
        
        logger.info(f"Optimization complete. Found {len(pareto_front)} Pareto-optimal solutions")
        
        return res, pareto_front, pareto_solutions
    
    def step5_mcdm(self, pareto_front: np.ndarray, 
                   pareto_solutions: np.ndarray) -> Dict:
        """
        Step 5: Multi-Criteria Decision Making (TOPSIS)
        
        Selects optimal solution from Pareto front using TOPSIS.
        """
        logger.info("=" * 80)
        logger.info("STEP 5: MULTI-CRITERIA DECISION MAKING (TOPSIS)")
        logger.info("=" * 80)
        
        # Analyze Pareto front
        analyzer = ParetoFrontAnalyzer()
        tradeoff_analysis = analyzer.analyze_tradeoff(pareto_front)
        knee_idx = analyzer.find_knee_point(pareto_front)
        
        logger.info("Pareto Front Analysis:")
        logger.info(f"  Energy range: {tradeoff_analysis.get('energy_range', 0):.2f} kWh")
        logger.info(f"  Discomfort range: {tradeoff_analysis.get('discomfort_range', 0):.2f}")
        logger.info(f"  Trade-off ratio: {tradeoff_analysis.get('tradeoff_ratio', 0):.4f}")
        logger.info(f"  Knee point index: {knee_idx}")
        
        # Apply TOPSIS
        mcdm_config = self.config.get('mcdm', {})
        weights = mcdm_config.get('weights', {'energy': 0.5, 'discomfort': 0.5})
        
        self.topsis = TOPSIS(weights=weights)
        best_idx, relative_closeness, topsis_results = self.topsis.rank_solutions(pareto_front)
        
        optimal_solution = pareto_solutions[best_idx]
        optimal_objectives = pareto_front[best_idx]
        
        logger.info(f"\nOptimal Solution (TOPSIS):")
        logger.info(f"  T_set_heat: {optimal_solution[0]:.2f} °C")
        logger.info(f"  T_set_cool: {optimal_solution[1]:.2f} °C")
        logger.info(f"  Energy: {optimal_objectives[0]:.2f} kWh")
        logger.info(f"  Discomfort: {optimal_objectives[1]:.2f}")
        
        results = {
            'optimal_setpoints': {
                'T_set_heat': float(optimal_solution[0]),
                'T_set_cool': float(optimal_solution[1])
            },
            'optimal_objectives': {
                'energy': float(optimal_objectives[0]),
                'discomfort': float(optimal_objectives[1])
            },
            'pareto_front': pareto_front,
            'pareto_solutions': pareto_solutions,
            'topsis_results': topsis_results,
            'tradeoff_analysis': tradeoff_analysis
        }
        
        return results
    
    def step6_visualization(self, results: Dict):
        """
        Step 6: Visualization & Results Export
        
        Creates visualizations and saves results.
        """
        logger.info("=" * 80)
        logger.info("STEP 6: VISUALIZATION & RESULTS EXPORT")
        logger.info("=" * 80)
        
        pareto_front = results['pareto_front']
        optimal_objectives = results['optimal_objectives']
        optimal_obj_array = np.array([[optimal_objectives['energy'], 
                                      optimal_objectives['discomfort']]])
        
        # Plot Pareto front
        fig_path = self.config.get('output', {}).get('figures_dir', 'figures') + '/pareto_front.png'
        plot_pareto_front(
            pareto_front,
            optimal_solution=optimal_obj_array[0],
            save_path=fig_path,
            title="Pareto Front: Energy Consumption vs. Thermal Discomfort"
        )
        
        # Save results
        results_path = self.config.get('output', {}).get('results_dir', 'results') + '/optimization_results.json'
        save_results(results, results_path)
        
        logger.info("Visualization and results export complete.")
    
    def run_complete_pipeline(self):
        """
        Execute the complete optimization pipeline.
        """
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-OBJECTIVE BUILDING ENERGY OPTIMIZATION FRAMEWORK")
        logger.info("Target Journal: Applied Energy (Q1)")
        logger.info("=" * 80 + "\n")
        
        try:
            # Step 1: Data Acquisition & Harmonization
            df_harmonized = self.step1_data_acquisition()
            
            # Step 2: Feature Engineering
            df_engineered = self.step2_feature_engineering(df_harmonized)
            
            # Step 3: Digital Twin
            self.step3_digital_twin(df_engineered)
            
            # Step 4: Optimization
            res, pareto_front, pareto_solutions = self.step4_optimization(df_engineered)
            
            # Step 5: MCDM
            results = self.step5_mcdm(pareto_front, pareto_solutions)
            
            # Step 6: Visualization
            self.step6_visualization(results)
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE EXECUTION COMPLETE")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise


def main():
    """
    Main entry point.
    """
    pipeline = BuildingEnergyOptimizationPipeline()
    results = pipeline.run_complete_pipeline()
    return results


if __name__ == "__main__":
    results = main()
