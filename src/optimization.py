"""
Multi-Objective Optimization Module (NSGA-II)

This module implements the Multi-Objective Optimization problem formulation
for deriving optimal HVAC control strategies (thermostat setpoints).

Optimization Engine: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
Objectives:
- Minimize Energy Consumption (J1)
- Minimize Thermal Discomfort (J2) - ASHRAE 55 Compliance
"""

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from typing import Dict, Callable, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HVACOptimizationProblem(Problem):
    """
    Multi-Objective Optimization Problem for HVAC Control.
    
    Decision Variables:
    - T_set_heat: Heating setpoint (lower bound) [°C]
    - T_set_cool: Cooling setpoint (upper bound) [°C]
    
    Objectives:
    - J1: Minimize Energy Consumption
    - J2: Minimize Thermal Discomfort (ASHRAE 55 compliance)
    
    Constraints:
    - Deadband: T_cool - T_heat >= 2°C
    - Actuator limits: 18°C <= T_set <= 26°C
    """
    
    def __init__(self, digital_twin: Any, 
                 historical_data: pd.DataFrame,
                 config: Dict):
        """
        Initialize optimization problem.
        
        Args:
            digital_twin: Trained Digital Twin model object (must have .predict() method)
            historical_data: Historical building data for simulation
            config: Configuration dictionary
        """
        self.digital_twin = digital_twin
        self.historical_data = historical_data.copy()
        self.config = config
        
        opt_config = config.get('optimization', {})
        dv_config = opt_config.get('decision_variables', {})
        
        # Decision variable bounds
        T_heat_config = dv_config.get('T_set_heat', {})
        T_cool_config = dv_config.get('T_set_cool', {})
        
        self.T_heat_min = T_heat_config.get('lower_bound', 18.0)
        self.T_heat_max = T_heat_config.get('upper_bound', 24.0)
        self.T_cool_min = T_cool_config.get('lower_bound', 20.0)
        self.T_cool_max = T_cool_config.get('upper_bound', 26.0)
        
        # Constraints
        self.deadband_min = opt_config.get('deadband_min', 2.0)
        
        # Discomfort parameters
        discomfort_config = opt_config.get('discomfort', {})
        self.temp_weight = discomfort_config.get('temp_weight', 1.0)
        self.rh_weight = discomfort_config.get('rh_weight', 0.5)
        self.optimal_temp = discomfort_config.get('optimal_temp', 22.0)
        self.optimal_rh = discomfort_config.get('optimal_rh', 50.0)
        
        # Problem definition for pymoo
        n_var = 2  # T_set_heat, T_set_cool
        n_obj = 2  # Energy, Discomfort
        n_constr = 1  # Deadband constraint
        
        xl = np.array([self.T_heat_min, self.T_cool_min])
        xu = np.array([self.T_heat_max, self.T_cool_max])
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        
    def _simulate_hvac_control(self, T_set_heat: float, T_set_cool: float) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Simulate HVAC control with given setpoints.
        
        Args:
            T_set_heat: Heating setpoint [°C]
            T_set_cool: Cooling setpoint [°C]
            
        Returns:
            Tuple of (simulated_data, energy_predictions)
        """
        sim_data = self.historical_data.copy()
        
        # Find temperature columns
        t_in_cols = [col for col in sim_data.columns if 'T_in' in col or col == 'T1']
        if not t_in_cols:
            t_in_cols = ['T1'] if 'T1' in sim_data.columns else []
        
        if not t_in_cols:
            logger.warning("Could not find indoor temperature column for HVAC simulation")
            return sim_data, np.zeros(len(sim_data))
        
        T_in = sim_data[t_in_cols[0]]
        
        # Simulate HVAC operation based on setpoints
        # Simple rule-based control: heat if T < T_set_heat, cool if T > T_set_cool
        heating_active = (T_in < T_set_heat).astype(int)
        cooling_active = (T_in > T_set_cool).astype(int)
        
        # Update indoor temperature (simplified dynamics)
        # In reality, this would be more complex, but for optimization we use the surrogate
        sim_data['T_set_heat'] = T_set_heat
        sim_data['T_set_cool'] = T_set_cool
        sim_data['heating_active'] = heating_active
        sim_data['cooling_active'] = cooling_active
        
        # Predict energy consumption using Digital Twin
        # Prepare features for prediction
        feature_cols = [col for col in sim_data.columns 
                       if col not in ['Appliances', 'E_load', 'date']]
        X_sim = sim_data[feature_cols]
        
        # Ensure all required features are present
        try:
            energy_pred = self.digital_twin.predict(X_sim)
        except Exception as e:
            logger.warning(f"Prediction error: {e}. Using baseline energy.")
            energy_pred = sim_data.get('Appliances', np.zeros(len(sim_data)))
            if isinstance(energy_pred, pd.Series):
                energy_pred = energy_pred.values
        
        return sim_data, energy_pred
    
    def _calculate_discomfort(self, sim_data: pd.DataFrame) -> float:
        """
        Calculate thermal discomfort index (ASHRAE 55 compliance proxy).
        
        J2 = Σ(|T_in - T_optimal| + λ · |RH_in - 50%|)
        
        Args:
            sim_data: Simulated building data
            
        Returns:
            Discomfort index
        """
        # Find temperature and humidity columns
        t_in_cols = [col for col in sim_data.columns if 'T_in' in col or col == 'T1']
        rh_in_cols = [col for col in sim_data.columns if 'RH_in' in col or 'RH_1' in col]
        
        if not t_in_cols:
            return 0.0
        
        T_in = sim_data[t_in_cols[0]]
        temp_discomfort = np.abs(T_in - self.optimal_temp)
        
        if rh_in_cols:
            RH_in = sim_data[rh_in_cols[0]]
            rh_discomfort = np.abs(RH_in - self.optimal_rh)
            discomfort = self.temp_weight * temp_discomfort + self.rh_weight * rh_discomfort
        else:
            discomfort = self.temp_weight * temp_discomfort
        
        return discomfort.sum()
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives and constraints for given decision variables.
        
        Args:
            X: Decision variable array (n_pop x n_var)
            out: Output dictionary to populate
        """
        n_pop = X.shape[0]
        f1 = np.zeros(n_pop)  # Energy consumption
        f2 = np.zeros(n_pop)  # Discomfort
        
        for i in range(n_pop):
            T_set_heat = X[i, 0]
            T_set_cool = X[i, 1]
            
            # Simulate HVAC control
            sim_data, energy_pred = self._simulate_hvac_control(T_set_heat, T_set_cool)
            
            # Objective 1: Minimize Energy Consumption
            f1[i] = np.sum(energy_pred)
            
            # Objective 2: Minimize Thermal Discomfort
            f2[i] = self._calculate_discomfort(sim_data)
        
        out["F"] = np.column_stack([f1, f2])
        
        # Constraint: Deadband (T_cool - T_heat >= deadband_min)
        g = self.deadband_min - (X[:, 1] - X[:, 0])  # g <= 0 means constraint satisfied
        out["G"] = g.reshape(-1, 1)


class NSGA2Optimizer:
    """
    NSGA-II Optimizer wrapper for HVAC control optimization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize optimizer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        opt_config = config.get('optimization', {})
        
        self.population_size = opt_config.get('population_size', 50)
        self.n_generations = opt_config.get('n_generations', 100)
        
    def optimize(self, problem: HVACOptimizationProblem, 
                  verbose: bool = True) -> Tuple:
        """
        Run NSGA-II optimization.
        
        Args:
            problem: HVACOptimizationProblem instance
            verbose: Whether to print progress
            
        Returns:
            Tuple of (result, pareto_front)
        """
        logger.info("Starting NSGA-II optimization...")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Generations: {self.n_generations}")
        
        # Initialize NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Run optimization
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.n_generations),
            verbose=verbose,
            seed=42
        )
        
        # Extract Pareto front
        pareto_front = res.F
        pareto_solutions = res.X
        
        logger.info(f"Optimization completed. Found {len(pareto_front)} Pareto-optimal solutions")
        
        return res, pareto_front, pareto_solutions
