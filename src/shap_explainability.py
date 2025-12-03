"""
SHAP Explainability Module for PI-DRL Framework
Provides interpretability for the PPO agent's decisions

Target Journal: Applied Energy (Q1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

from stable_baselines3 import PPO


# Publication-quality settings
def setup_shap_style():
    """Configure matplotlib for SHAP visualizations."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
    })


class SHAPExplainer:
    """
    SHAP-based explainability for PI-DRL agent.
    
    Provides:
    1. Feature importance analysis
    2. Decision explanation
    3. Summary plots for publication
    """
    
    def __init__(
        self,
        agent: PPO,
        feature_names: List[str] = None,
        save_dir: str = './figures'
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            agent: Trained PPO agent
            feature_names: List of state feature names
            save_dir: Directory to save figures
        """
        self.agent = agent
        self.feature_names = feature_names or [
            'Indoor Temp (°C)',
            'Outdoor Temp (°C)',
            'Solar Radiation (W/m²)',
            'Electricity Price ($/kWh)',
            'Last Action',
            'Time Index'
        ]
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        setup_shap_style()
        
    def _policy_function(self, X: np.ndarray) -> np.ndarray:
        """
        Wrapper for agent's policy for SHAP.
        
        Args:
            X: Input observations [batch, features]
            
        Returns:
            Action probabilities or values
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        actions = []
        for obs in X:
            action, _ = self.agent.predict(obs.astype(np.float32), deterministic=True)
            actions.append(float(action))
        
        return np.array(actions)
    
    def compute_shap_values(
        self,
        background_data: np.ndarray,
        test_data: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, object]:
        """
        Compute SHAP values for test observations.
        
        Args:
            background_data: Background dataset for SHAP
            test_data: Observations to explain
            n_samples: Number of background samples
            
        Returns:
            SHAP values and explainer object
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Using permutation importance instead.")
            return self._permutation_importance(background_data, test_data)
        
        # Sample background data
        if len(background_data) > n_samples:
            idx = np.random.choice(len(background_data), n_samples, replace=False)
            background_sample = background_data[idx]
        else:
            background_sample = background_data
        
        # Create KernelExplainer (model-agnostic)
        explainer = shap.KernelExplainer(
            self._policy_function,
            background_sample
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(test_data)
        
        return shap_values, explainer
    
    def _permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        n_repeats: int = 10
    ) -> Tuple[np.ndarray, None]:
        """
        Alternative: Permutation-based feature importance.
        """
        base_pred = self._policy_function(X)
        n_features = X.shape[1]
        importance = np.zeros((len(X), n_features))
        
        for j in range(n_features):
            X_permuted = X.copy()
            for _ in range(n_repeats):
                # Shuffle feature j
                X_permuted[:, j] = np.random.permutation(X[:, j])
                permuted_pred = self._policy_function(X_permuted)
                importance[:, j] += np.abs(base_pred - permuted_pred)
            importance[:, j] /= n_repeats
        
        return importance, None
    
    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        save_name: str = 'fig_shap_importance.png'
    ) -> str:
        """
        Create feature importance bar chart.
        
        Args:
            shap_values: SHAP values [n_samples, n_features]
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Mean absolute SHAP values
        mean_importance = np.abs(shap_values).mean(axis=0)
        
        # Sort by importance
        sorted_idx = np.argsort(mean_importance)
        sorted_features = [self.feature_names[i] for i in sorted_idx]
        sorted_importance = mean_importance[sorted_idx]
        
        # Plot
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_features)))
        bars = ax.barh(sorted_features, sorted_importance, color=colors)
        
        ax.set_xlabel('Mean |SHAP Value|', fontweight='bold')
        ax.set_title('Feature Importance for HVAC Control Decisions', 
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, sorted_importance):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SHAP] Feature importance saved: {filepath}")
        return filepath
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        save_name: str = 'fig_shap_summary.png'
    ) -> str:
        """
        Create SHAP summary plot (beeswarm).
        
        Args:
            shap_values: SHAP values
            X: Input data
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        if not SHAP_AVAILABLE:
            return self._plot_alternative_summary(shap_values, X, save_name)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use SHAP's built-in plotting
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False,
            plot_size=(10, 6)
        )
        
        plt.title('SHAP Summary: Feature Impact on Control Decisions',
                 fontweight='bold', fontsize=13)
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SHAP] Summary plot saved: {filepath}")
        return filepath
    
    def _plot_alternative_summary(
        self,
        importance: np.ndarray,
        X: np.ndarray,
        save_name: str
    ) -> str:
        """Alternative summary when SHAP not available."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_importance = np.abs(importance).mean(axis=0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_importance
        }).sort_values('Importance', ascending=True)
        
        # Plot
        ax.barh(df['Feature'], df['Importance'], color='steelblue')
        ax.set_xlabel('Mean Absolute Impact', fontweight='bold')
        ax.set_title('Feature Importance (Permutation-based)',
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_decision_explanation(
        self,
        obs: np.ndarray,
        shap_values: np.ndarray,
        save_name: str = 'fig_shap_decision.png'
    ) -> str:
        """
        Explain a single decision using waterfall plot.
        
        Args:
            obs: Single observation
            shap_values: SHAP values for this observation
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Get action
        action, _ = self.agent.predict(obs.astype(np.float32), deterministic=True)
        action_name = 'ON' if action == 1 else 'OFF'
        
        # Sort by absolute impact
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]
        
        # Plot waterfall-style
        colors = ['#E63946' if v > 0 else '#1D3557' for v in shap_values[sorted_idx]]
        y_pos = np.arange(len(self.feature_names))
        
        ax.barh(y_pos, shap_values[sorted_idx], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in sorted_idx])
        
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (impact on decision)', fontweight='bold')
        ax.set_title(f'Decision Explanation: Action = {action_name}',
                    fontweight='bold', fontsize=13)
        
        # Add state values
        for i, idx in enumerate(sorted_idx):
            val = obs[idx]
            ax.text(0.02, i, f'({val:.2f})', va='center', fontsize=9,
                   transform=ax.get_yaxis_transform())
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E63946', label='Push toward ON'),
            Patch(facecolor='#1D3557', label='Push toward OFF')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SHAP] Decision explanation saved: {filepath}")
        return filepath
    
    def plot_dependence(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_idx: int = 1,  # Outdoor temp by default
        interaction_idx: int = 5,  # Time index
        save_name: str = 'fig_shap_dependence.png'
    ) -> str:
        """
        Create SHAP dependence plot.
        
        Shows how a feature's value affects the model output.
        
        Args:
            shap_values: SHAP values
            X: Input data
            feature_idx: Index of main feature
            interaction_idx: Index of interaction feature
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        feature_name = self.feature_names[feature_idx]
        interaction_name = self.feature_names[interaction_idx]
        
        scatter = ax.scatter(
            X[:, feature_idx],
            shap_values[:, feature_idx],
            c=X[:, interaction_idx],
            cmap='coolwarm',
            alpha=0.6,
            s=20
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(interaction_name, fontweight='bold')
        
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel(feature_name, fontweight='bold')
        ax.set_ylabel(f'SHAP Value for {feature_name}', fontweight='bold')
        ax.set_title(f'SHAP Dependence: {feature_name}\n(colored by {interaction_name})',
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SHAP] Dependence plot saved: {filepath}")
        return filepath
    
    def generate_all_explanations(
        self,
        background_data: np.ndarray,
        test_data: np.ndarray
    ) -> Dict[str, str]:
        """
        Generate all SHAP explanation figures.
        
        Args:
            background_data: Background dataset
            test_data: Test observations
            
        Returns:
            Dictionary of saved file paths
        """
        print("\n" + "=" * 60)
        print("Generating SHAP Explainability Analysis")
        print("=" * 60)
        
        saved_files = {}
        
        # Compute SHAP values
        print("Computing SHAP values...")
        shap_values, explainer = self.compute_shap_values(
            background_data,
            test_data,
            n_samples=100
        )
        
        # Feature importance
        print("1. Feature Importance...")
        saved_files['importance'] = self.plot_feature_importance(shap_values)
        
        # Summary plot
        print("2. Summary Plot...")
        saved_files['summary'] = self.plot_summary(shap_values, test_data)
        
        # Decision explanation (first sample)
        print("3. Decision Explanation...")
        saved_files['decision'] = self.plot_decision_explanation(
            test_data[0],
            shap_values[0]
        )
        
        # Dependence plot
        print("4. Dependence Plot...")
        saved_files['dependence'] = self.plot_dependence(
            shap_values,
            test_data,
            feature_idx=1,  # Outdoor temp
            interaction_idx=5  # Time
        )
        
        print("=" * 60)
        print(f"SHAP analysis complete. Files saved to: {self.save_dir}")
        print("=" * 60)
        
        return saved_files


def generate_sample_observations(
    env,
    n_samples: int = 500,
    agent=None
) -> np.ndarray:
    """
    Generate sample observations by running environment.
    
    Args:
        env: SmartHomeEnv instance
        n_samples: Number of samples to collect
        agent: Optional agent for action selection
        
    Returns:
        Array of observations [n_samples, n_features]
    """
    observations = []
    obs, _ = env.reset()
    
    for _ in range(n_samples):
        observations.append(obs.copy())
        
        if agent is not None:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    return np.array(observations)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("SHAP Explainability Module")
    print("=" * 60)
    
    if not SHAP_AVAILABLE:
        print("SHAP not installed. Please install with: pip install shap")
    else:
        print("SHAP is available!")
    
    print("\nUsage:")
    print("  from shap_explainability import SHAPExplainer")
    print("  explainer = SHAPExplainer(agent, feature_names)")
    print("  explainer.generate_all_explanations(bg_data, test_data)")
