"""
Utility functions for the Multi-Objective Building Energy Optimization Framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os
import yaml

def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_data(url: str, save_path: str) -> str:
    """
    Download data from URL if not already present.
    
    Args:
        url: URL to download from
        save_path: Path to save the file
        
    Returns:
        Path to downloaded file
    """
    import requests
    
    if os.path.exists(save_path):
        print(f"Data already exists at {save_path}")
        return save_path
    
    ensure_dir(os.path.dirname(save_path))
    
    print(f"Downloading data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Data downloaded to {save_path}")
    return save_path

def plot_pareto_front(pareto_front: np.ndarray, 
                     optimal_solution: Optional[np.ndarray] = None,
                     save_path: Optional[str] = None,
                     title: str = "Pareto Front: Energy vs. Discomfort"):
    """
    Plot Pareto front with optional optimal solution highlight.
    
    Args:
        pareto_front: Pareto front solutions (n_solutions x 2)
        optimal_solution: Optional optimal solution to highlight
        save_path: Optional path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by first objective for better visualization
    sorted_indices = np.argsort(pareto_front[:, 0])
    sorted_front = pareto_front[sorted_indices]
    
    plt.plot(sorted_front[:, 0], sorted_front[:, 1], 
             'o-', label='Pareto Front', linewidth=2, markersize=8)
    
    if optimal_solution is not None:
        plt.plot(optimal_solution[0], optimal_solution[1], 
                'r*', markersize=20, label='Optimal Solution (TOPSIS)')
    
    plt.xlabel('Energy Consumption [kWh]', fontsize=12)
    plt.ylabel('Thermal Discomfort Index', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_optimization_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot optimization history (convergence).
    
    Args:
        history: Optimization history dictionary
        save_path: Optional path to save figure
    """
    # This would require storing history during optimization
    # Placeholder for future implementation
    pass

def _convert_to_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float_)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_results(results: Dict, filepath: str):
    """
    Save optimization results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save results
    """
    ensure_dir(os.path.dirname(filepath))
    
    # Convert numpy arrays and types to Python native types for JSON serialization
    results_serializable = _convert_to_serializable(results)
    
    import json
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filepath}")
