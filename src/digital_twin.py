"""
Generalizable Digital Twin (Surrogate Modeling)

This module implements a Stacking Ensemble Model that acts as a fast, accurate
proxy for the building's thermal physics, replacing heavy simulation software
(like EnergyPlus) in the optimization loop.

The model is validated via Cross-Validation across multiple datasets to ensure
generalization capability.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Using XGBoost and LightGBM only.")
import logging
import joblib
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DigitalTwin:
    """
    Stacking Ensemble Model serving as a fast surrogate for building thermal physics.
    
    This model replaces computationally expensive simulation software in the
    optimization loop, enabling real-time or near-real-time optimization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Digital Twin with configuration.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        dt_config = config.get('digital_twin', {})
        
        # Base models configuration
        self.base_models_config = dt_config.get('base_models', [])
        
        # Meta-model configuration
        self.meta_model_config = dt_config.get('meta_model', {})
        
        # Cross-validation parameters
        self.cv_folds = dt_config.get('cv_folds', 5)
        self.random_state = dt_config.get('random_state', 42)
        
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def _create_base_models(self) -> list:
        """
        Create base models for stacking ensemble.
        
        Returns:
            List of (name, model) tuples
        """
        base_models = []
        
        for model_config in self.base_models_config:
            model_name = model_config['name']
            params = model_config.get('params', {})
            
            if model_name == 'xgboost':
                model = xgb.XGBRegressor(**params)
                base_models.append(('xgb', model))
            
            elif model_name == 'lightgbm':
                model = lgb.LGBMRegressor(**params, verbose=-1)
                base_models.append(('lgb', model))
            
            elif model_name == 'catboost' and CATBOOST_AVAILABLE:
                model = cb.CatBoostRegressor(**params, verbose=False)
                base_models.append(('cat', model))
        
        # Ensure at least two base models
        if len(base_models) < 2:
            logger.warning("Less than 2 base models available. Adding default models.")
            if not any('xgb' in name for name, _ in base_models):
                base_models.append(('xgb', xgb.XGBRegressor(random_state=self.random_state)))
            if not any('lgb' in name for name, _ in base_models):
                base_models.append(('lgb', lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)))
        
        return base_models
    
    def _create_meta_model(self):
        """
        Create meta-model for stacking ensemble.
        
        Returns:
            Meta-model instance
        """
        meta_name = self.meta_model_config.get('name', 'ridge')
        meta_params = self.meta_model_config.get('params', {})
        
        if meta_name == 'ridge':
            return Ridge(**meta_params)
        else:
            return Ridge(**meta_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: Optional[float] = None) -> Dict:
        """
        Train the stacking ensemble model.
        
        Args:
            X: Feature matrix
            y: Target variable (energy consumption)
            validation_split: Optional validation split ratio
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training Digital Twin (Stacking Ensemble)...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Create base models
        base_models = self._create_base_models()
        meta_model = self._create_meta_model()
        
        # Create stacking ensemble
        self.model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=self.cv_folds,
            n_jobs=-1
        )
        
        # Split data if validation_split provided
        if validation_split:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train on training set
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred_val = self.model.predict(X_val)
            
            metrics = {
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'val_mae': mean_absolute_error(y_val, y_pred_val),
                'val_r2': r2_score(y_val, y_pred_val)
            }
            
            logger.info(f"Validation RMSE: {metrics['val_rmse']:.4f}")
            logger.info(f"Validation MAE: {metrics['val_mae']:.4f}")
            logger.info(f"Validation R²: {metrics['val_r2']:.4f}")
        else:
            # Train on full dataset
            self.model.fit(X, y)
            metrics = {}
        
        # Cross-validation on full dataset
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        cv_rmse = np.sqrt(-cv_scores)
        metrics['cv_rmse_mean'] = cv_rmse.mean()
        metrics['cv_rmse_std'] = cv_rmse.std()
        
        logger.info(f"Cross-Validation RMSE: {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_mean']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict energy consumption using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted energy consumption
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100  # Mean Absolute Percentage Error
        }
        
        logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
        logger.info(f"Test MAE: {metrics['mae']:.4f}")
        logger.info(f"Test R²: {metrics['r2']:.4f}")
        logger.info(f"Test MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        logger.info(f"Model loaded from {filepath}")
