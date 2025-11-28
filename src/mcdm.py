"""
Multi-Criteria Decision Making (MCDM) Module

This module implements TOPSIS (Technique for Order of Preference by Similarity
to Ideal Solution) for selecting the optimal solution from the Pareto front.

The "Knee Point" represents the best engineering compromise where significant
energy is saved with negligible loss in occupant comfort.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TOPSIS:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    
    Selects the optimal solution from Pareto front by finding the solution
    closest to the ideal point and farthest from the negative-ideal point.
    """
    
    def __init__(self, weights: Optional[Dict] = None):
        """
        Initialize TOPSIS with criteria weights.
        
        Args:
            weights: Dictionary with weights for each objective
                    (default: equal weights)
        """
        self.weights = weights or {'energy': 0.5, 'discomfort': 0.5}
        self.ideal_solution = None
        self.negative_ideal_solution = None
        
    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize decision matrix using vector normalization.
        
        Args:
            matrix: Decision matrix (n_alternatives x n_criteria)
            
        Returns:
            Normalized matrix
        """
        # Vector normalization: rij = xij / sqrt(sum(xij^2))
        norm_matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=0))
        return norm_matrix
    
    def calculate_ideal_solutions(self, normalized_matrix: np.ndarray, 
                                  benefit_criteria: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ideal and negative-ideal solutions.
        
        For minimization problems:
        - Ideal: minimum values (best case)
        - Negative-ideal: maximum values (worst case)
        
        Args:
            normalized_matrix: Normalized decision matrix
            benefit_criteria: List indicating if each criterion is benefit (True) or cost (False)
            
        Returns:
            Tuple of (ideal_solution, negative_ideal_solution)
        """
        ideal = np.zeros(normalized_matrix.shape[1])
        negative_ideal = np.zeros(normalized_matrix.shape[1])
        
        for j in range(normalized_matrix.shape[1]):
            if benefit_criteria[j]:  # Maximize (benefit)
                ideal[j] = np.max(normalized_matrix[:, j])
                negative_ideal[j] = np.min(normalized_matrix[:, j])
            else:  # Minimize (cost)
                ideal[j] = np.min(normalized_matrix[:, j])
                negative_ideal[j] = np.max(normalized_matrix[:, j])
        
        return ideal, negative_ideal
    
    def calculate_distances(self, normalized_matrix: np.ndarray,
                           ideal: np.ndarray, negative_ideal: np.ndarray,
                           weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate distances to ideal and negative-ideal solutions.
        
        Args:
            normalized_matrix: Normalized decision matrix
            ideal: Ideal solution vector
            negative_ideal: Negative-ideal solution vector
            weights: Criteria weights
            
        Returns:
            Tuple of (distances_to_ideal, distances_to_negative_ideal)
        """
        n_alternatives = normalized_matrix.shape[0]
        d_plus = np.zeros(n_alternatives)
        d_minus = np.zeros(n_alternatives)
        
        for i in range(n_alternatives):
            # Weighted Euclidean distance to ideal
            d_plus[i] = np.sqrt(np.sum(weights * (normalized_matrix[i, :] - ideal) ** 2))
            
            # Weighted Euclidean distance to negative-ideal
            d_minus[i] = np.sqrt(np.sum(weights * (normalized_matrix[i, :] - negative_ideal) ** 2))
        
        return d_plus, d_minus
    
    def rank_solutions(self, pareto_front: np.ndarray, 
                      weights: Optional[Dict] = None) -> Tuple[int, np.ndarray, Dict]:
        """
        Rank solutions using TOPSIS and select the best one.
        
        Args:
            pareto_front: Pareto front solutions (n_solutions x n_objectives)
            weights: Optional weights dictionary (overrides initialization)
            
        Returns:
            Tuple of (best_solution_index, relative_closeness_scores, results_dict)
        """
        if weights is None:
            weights = self.weights
        
        # Convert weights to array
        weight_array = np.array([weights.get('energy', 0.5), 
                                weights.get('discomfort', 0.5)])
        weight_array = weight_array / weight_array.sum()  # Normalize
        
        # Normalize decision matrix
        normalized_matrix = self.normalize(pareto_front)
        
        # Both objectives are minimization (cost criteria)
        benefit_criteria = [False, False]  # Energy (minimize), Discomfort (minimize)
        
        # Calculate ideal solutions
        ideal, negative_ideal = self.calculate_ideal_solutions(
            normalized_matrix, benefit_criteria
        )
        
        self.ideal_solution = ideal
        self.negative_ideal_solution = negative_ideal
        
        # Calculate distances
        d_plus, d_minus = self.calculate_distances(
            normalized_matrix, ideal, negative_ideal, weight_array
        )
        
        # Calculate relative closeness
        relative_closeness = d_minus / (d_plus + d_minus + 1e-10)
        
        # Best solution is the one with highest relative closeness
        best_idx = np.argmax(relative_closeness)
        
        results = {
            'best_solution_index': best_idx,
            'relative_closeness': relative_closeness,
            'distances_to_ideal': d_plus,
            'distances_to_negative_ideal': d_minus,
            'ideal_solution': ideal,
            'negative_ideal_solution': negative_ideal
        }
        
        logger.info(f"TOPSIS selected solution {best_idx} as optimal")
        logger.info(f"Relative closeness: {relative_closeness[best_idx]:.4f}")
        
        return best_idx, relative_closeness, results


class ParetoFrontAnalyzer:
    """
    Analyzes Pareto front and identifies trade-off characteristics.
    """
    
    @staticmethod
    def find_knee_point(pareto_front: np.ndarray) -> int:
        """
        Find knee point on Pareto front using distance-based method.
        
        The knee point represents the solution with maximum curvature change,
        indicating the best trade-off between objectives.
        
        Args:
            pareto_front: Pareto front solutions (n_solutions x n_objectives)
            
        Returns:
            Index of knee point solution
        """
        if len(pareto_front) < 3:
            return 0
        
        # Normalize objectives to [0, 1] for fair comparison
        normalized = pareto_front.copy()
        for j in range(pareto_front.shape[1]):
            min_val = np.min(pareto_front[:, j])
            max_val = np.max(pareto_front[:, j])
            if max_val > min_val:
                normalized[:, j] = (pareto_front[:, j] - min_val) / (max_val - min_val)
        
        # Calculate distances from each point to the line connecting extremes
        # Knee point maximizes this distance
        extreme_1 = normalized[0]
        extreme_2 = normalized[-1]
        
        # Vector from extreme_1 to extreme_2
        line_vec = extreme_2 - extreme_1
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-10:
            return len(pareto_front) // 2
        
        distances = []
        for i in range(len(normalized)):
            point = normalized[i]
            # Vector from extreme_1 to point
            point_vec = point - extreme_1
            
            # Projection of point_vec onto line_vec
            projection = np.dot(point_vec, line_vec) / (line_length ** 2) * line_vec
            
            # Perpendicular distance
            perpendicular = point_vec - projection
            distance = np.linalg.norm(perpendicular)
            distances.append(distance)
        
        knee_idx = np.argmax(distances)
        return knee_idx
    
    @staticmethod
    def analyze_tradeoff(pareto_front: np.ndarray) -> Dict:
        """
        Analyze trade-off characteristics of Pareto front.
        
        Args:
            pareto_front: Pareto front solutions
            
        Returns:
            Dictionary with trade-off analysis results
        """
        if len(pareto_front) == 0:
            return {}
        
        # Calculate ranges
        energy_range = np.max(pareto_front[:, 0]) - np.min(pareto_front[:, 0])
        discomfort_range = np.max(pareto_front[:, 1]) - np.min(pareto_front[:, 1])
        
        # Calculate trade-off ratio
        tradeoff_ratio = energy_range / (discomfort_range + 1e-10)
        
        # Find extreme solutions
        min_energy_idx = np.argmin(pareto_front[:, 0])
        min_discomfort_idx = np.argmin(pareto_front[:, 1])
        
        analysis = {
            'energy_range': energy_range,
            'discomfort_range': discomfort_range,
            'tradeoff_ratio': tradeoff_ratio,
            'min_energy_solution': min_energy_idx,
            'min_discomfort_solution': min_discomfort_idx,
            'n_solutions': len(pareto_front)
        }
        
        return analysis
