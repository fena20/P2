"""
03_xgboost_model.py
====================
XGBoost Model for Thermal Intensity Prediction

This module trains and evaluates an XGBoost regressor to predict
thermal intensity (I = E_heat / (A × HDD)) based on building,
envelope, climate, and system characteristics.

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import joblib
from typing import Tuple, Dict, List

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned dataset."""
    filepath = OUTPUT_DIR / "03_gas_heated_clean.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found. Run 01_data_prep.py first.")
    return pd.read_csv(filepath)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Prepare features and target for modeling.
    
    Returns:
        X: Feature matrix
        y: Target variable (Thermal_Intensity_I)
        encoders: Dictionary of label encoders for categorical variables
    """
    logger.info("Preparing features for modeling...")
    
    # Define feature columns
    numeric_features = [
        'HDD65',
        'A_heated',
        'building_age',
        'heating_equip_age',
        'log_sqft',
    ]
    
    categorical_features = [
        'TYPEHUQ',       # Housing type
        'YEARMADERANGE', # Year built range
        'DRAFTY',        # Draftiness
        'ADQINSUL',      # Insulation adequacy
        'TYPEGLASS',     # Window glass type
        'EQUIPM',        # Heating equipment type
        'REGIONC',       # Census region
        'DIVISION',      # Census division
        'envelope_class',
        'climate_zone',
    ]
    
    # Check which features are available
    available_numeric = [f for f in numeric_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    
    logger.info(f"Available numeric features: {available_numeric}")
    logger.info(f"Available categorical features: {available_categorical}")
    
    # Create feature matrix
    X = df[available_numeric + available_categorical].copy()
    y = df['Thermal_Intensity_I'].copy()
    
    # Remove rows with missing target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Encode categorical variables
    encoders = {}
    for col in available_categorical:
        if X[col].dtype == 'object' or col in ['envelope_class', 'climate_zone']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('missing').astype(str))
            encoders[col] = le
        else:
            X[col] = X[col].fillna(-1)
    
    # Fill missing numeric values
    for col in available_numeric:
        X[col] = X[col].fillna(X[col].median())
    
    logger.info(f"Final feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, encoders


def split_data(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
               test_size: float = 0.2, val_size: float = 0.2,
               stratify_col: str = 'REGIONC') -> Tuple:
    """
    Split data into train/validation/test sets.
    
    Uses stratification by region or climate zone to ensure
    representative samples in each split.
    """
    logger.info("Splitting data into train/val/test sets...")
    
    # Get stratification variable
    valid_idx = y.notna()
    strat = df.loc[valid_idx.index, stratify_col].fillna(-1) if stratify_col in df.columns else None
    
    # Get sample weights
    weights = df.loc[valid_idx.index, 'NWEIGHT'].values if 'NWEIGHT' in df.columns else None
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test, strat_trainval, _ = train_test_split(
        X, y, strat if strat is not None else y,
        test_size=test_size,
        random_state=42,
        stratify=strat if strat is not None else None
    )
    
    if weights is not None:
        w_trainval, w_test = train_test_split(
            weights[valid_idx.values],
            test_size=test_size,
            random_state=42
        )
    else:
        w_trainval, w_test = None, None
    
    # Second split: train vs val
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=adjusted_val_size,
        random_state=42,
        stratify=strat_trainval if strat is not None else None
    )
    
    if w_trainval is not None:
        w_train, w_val = train_test_split(
            w_trainval,
            test_size=adjusted_val_size,
            random_state=42
        )
    else:
        w_train, w_val = None, None
    
    logger.info(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            w_train, w_val, w_test)


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        sample_weight: np.ndarray = None,
                        params: dict = None) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor with early stopping.
    """
    logger.info("Training XGBoost model...")
    
    # Default hyperparameters
    default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
    }
    
    if params:
        default_params.update(params)
    
    model = xgb.XGBRegressor(**default_params)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score (validation): {model.best_score:.4f}")
    
    return model


def evaluate_model(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series,
                   sample_weight: np.ndarray = None, set_name: str = "Test") -> dict:
    """
    Evaluate model performance.
    """
    y_pred = model.predict(X)
    
    metrics = {
        'set': set_name,
        'n_samples': len(y),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred),
        'mape': np.mean(np.abs((y - y_pred) / y)) * 100,
    }
    
    # Weighted metrics (if weights available)
    if sample_weight is not None:
        weighted_se = sample_weight * (y - y_pred)**2
        metrics['weighted_rmse'] = np.sqrt(np.sum(weighted_se) / np.sum(sample_weight))
    
    logger.info(f"{set_name} Performance:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics


def evaluate_by_subgroups(model: xgb.XGBRegressor, X: pd.DataFrame, 
                          y: pd.Series, df: pd.DataFrame,
                          groupby_cols: list) -> pd.DataFrame:
    """
    Evaluate model performance by subgroups (division, envelope class, etc.)
    """
    logger.info("Evaluating performance by subgroups...")
    
    y_pred = model.predict(X)
    
    results = []
    
    for col in groupby_cols:
        if col in df.columns:
            for group in df.loc[X.index, col].unique():
                mask = df.loc[X.index, col] == group
                if mask.sum() > 10:  # Minimum sample size
                    y_group = y[mask]
                    y_pred_group = y_pred[mask.values]
                    
                    results.append({
                        'Group Variable': col,
                        'Group': group,
                        'N': len(y_group),
                        'RMSE': np.sqrt(mean_squared_error(y_group, y_pred_group)),
                        'MAE': mean_absolute_error(y_group, y_pred_group),
                        'R²': r2_score(y_group, y_pred_group),
                    })
    
    return pd.DataFrame(results)


def generate_figure5_predictions(y_true: pd.Series, y_pred: np.ndarray,
                                 groups: pd.Series = None,
                                 group_name: str = 'Division'):
    """
    Generate Figure 5: Predicted vs. observed scatter plot.
    """
    logger.info("Generating Figure 5: Predicted vs. observed")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if groups is not None:
        for group in groups.unique():
            mask = groups == group
            ax.scatter(
                y_true[mask], y_pred[mask.values],
                alpha=0.5, label=group, s=20
            )
        ax.legend(title=group_name, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Add 45-degree line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=2, label='Perfect prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Metrics annotation
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.annotate(
        f'R² = {r2:.3f}\nRMSE = {rmse:.2f}',
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax.set_xlabel('Observed Thermal Intensity (BTU/sqft/HDD)', fontsize=12)
    ax.set_ylabel('Predicted Thermal Intensity (BTU/sqft/HDD)', fontsize=12)
    ax.set_title('Predicted vs. Observed Thermal Intensity', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure5_predicted_vs_observed.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure5_predicted_vs_observed.pdf", bbox_inches='tight')
    plt.close()


def generate_table3_model_performance(train_metrics: dict, val_metrics: dict,
                                      test_metrics: dict, 
                                      subgroup_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 3: Model performance summary.
    """
    logger.info("Generating Table 3: Model performance")
    
    # Overall metrics
    overall = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    overall.to_csv(TABLES_DIR / "table3_overall_performance.csv", index=False)
    
    # Subgroup metrics
    subgroup_metrics.to_csv(TABLES_DIR / "table3_subgroup_performance.csv", index=False)
    
    # Combined table for LaTeX
    combined = overall[['set', 'n_samples', 'rmse', 'mae', 'r2']]
    combined.columns = ['Set', 'N', 'RMSE', 'MAE', 'R²']
    combined.to_latex(
        TABLES_DIR / "table3_model_performance.tex",
        index=False,
        float_format="%.3f",
        caption="XGBoost model performance for thermal intensity prediction.",
        label="tab:model_performance"
    )
    
    return combined


def cross_validate_model(X: pd.DataFrame, y: pd.Series, 
                         n_splits: int = 5) -> dict:
    """
    Perform k-fold cross-validation.
    """
    logger.info(f"Performing {n_splits}-fold cross-validation...")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(
        model, X, y,
        cv=n_splits,
        scoring='r2',
        n_jobs=-1
    )
    
    rmse_scores = cross_val_score(
        model, X, y,
        cv=n_splits,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    cv_results = {
        'r2_mean': cv_scores.mean(),
        'r2_std': cv_scores.std(),
        'rmse_mean': -rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
    }
    
    logger.info(f"CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    logger.info(f"CV RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
    
    return cv_results


def run_xgboost_pipeline() -> dict:
    """
    Main function to run the complete XGBoost modeling pipeline.
    """
    logger.info("=" * 60)
    logger.info("XGBoost Thermal Intensity Modeling Pipeline")
    logger.info("=" * 60)
    
    # Load data
    df = load_processed_data()
    
    # Prepare features
    X, y, encoders = prepare_features(df)
    
    # Split data
    (X_train, X_val, X_test, y_train, y_val, y_test,
     w_train, w_val, w_test) = split_data(X, y, df)
    
    # Cross-validation
    cv_results = cross_validate_model(X, y)
    
    # Train model
    model = train_xgboost_model(X_train, y_train, X_val, y_val, sample_weight=w_train)
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, w_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, w_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, w_test, "Test")
    
    # Subgroup evaluation
    subgroup_metrics = evaluate_by_subgroups(
        model, X_test, y_test, df.loc[X_test.index],
        ['division_name', 'envelope_class', 'climate_zone']
    )
    
    # Generate outputs
    y_pred_test = model.predict(X_test)
    
    # Figure 5: Predicted vs observed
    groups = df.loc[X_test.index, 'division_name'] if 'division_name' in df.columns else None
    generate_figure5_predictions(y_test, y_pred_test, groups)
    
    # Table 3: Performance metrics
    generate_table3_model_performance(train_metrics, val_metrics, test_metrics, subgroup_metrics)
    
    # Save model
    model_path = MODELS_DIR / "xgboost_thermal_intensity.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save encoders
    encoders_path = MODELS_DIR / "label_encoders.joblib"
    joblib.dump(encoders, encoders_path)
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(TABLES_DIR / "feature_importance_xgboost.csv", index=False)
    
    logger.info("=" * 60)
    logger.info("XGBoost modeling complete!")
    logger.info("=" * 60)
    
    return {
        'model': model,
        'encoders': encoders,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'cv_results': cv_results,
        'subgroup_metrics': subgroup_metrics,
        'feature_importance': importance_df,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_test,
    }


if __name__ == "__main__":
    results = run_xgboost_pipeline()
    
    print("\n" + "=" * 60)
    print("MODEL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTest Set Performance:")
    print(f"  R²: {results['test_metrics']['r2']:.4f}")
    print(f"  RMSE: {results['test_metrics']['rmse']:.4f}")
    print(f"  MAE: {results['test_metrics']['mae']:.4f}")
    
    print(f"\nCross-Validation:")
    print(f"  R²: {results['cv_results']['r2_mean']:.4f} ± {results['cv_results']['r2_std']:.4f}")
    
    print(f"\nTop 10 Important Features:")
    print(results['feature_importance'].head(10).to_string())
