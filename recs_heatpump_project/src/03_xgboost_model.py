"""
Step 3: XGBoost Model for Thermal Intensity Prediction
=======================================================

Goal: Train XGBoost model to predict thermal intensity from building/envelope/climate features.

This script:
1. Loads cleaned data
2. Prepares features and target
3. Splits data (60/20/20) stratified by region/climate
4. Trains XGBoost regressor with sample weights
5. Evaluates performance (RMSE, MAE, R²) overall and by subgroups
6. Saves model and evaluation metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CLEANED_DATA = OUTPUT_DIR / "recs2020_gas_heated_cleaned.csv"
MODELS_DIR = OUTPUT_DIR / "models"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# RECS variables
VARIABLES = {
    'target': 'thermal_intensity',
    'weight': 'NWEIGHT',
    'division': 'DIVISION',
    'region': 'REGIONC',
    'envelope_class': 'envelope_class',
    'hdd65': 'HDD65',
}


def load_data(filepath):
    """Load cleaned dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records")
    return df


def prepare_features_target(df):
    """
    Prepare feature matrix and target vector.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector (thermal intensity)
    weights : pd.Series
        Sample weights
    """
    print("\nPreparing features and target...")
    
    # Target
    if VARIABLES['target'] not in df.columns:
        raise ValueError(f"Target variable '{VARIABLES['target']}' not found")
    
    y = df[VARIABLES['target']].copy()
    
    # Remove rows with missing target
    valid_mask = y.notna()
    df = df[valid_mask].copy()
    y = y[valid_mask]
    
    print(f"Valid records with target: {len(y):,}")
    
    # Sample weights
    if VARIABLES['weight'] in df.columns:
        weights = df[VARIABLES['weight']].copy()
    else:
        print("WARNING: Sample weights not found. Using uniform weights.")
        weights = pd.Series(1.0, index=df.index)
    
    # Feature selection
    # Exclude target, weights, and non-feature columns
    exclude_cols = [
        VARIABLES['target'],
        VARIABLES['weight'],
        # Add other non-feature columns as needed
    ]
    
    # Select numeric features (excluding target and weights)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # Add categorical features (one-hot encode)
    categorical_cols = []
    for col in ['envelope_class', 'DIVISION', 'TYPEHUQ', 'REGIONC']:
        if col in df.columns:
            categorical_cols.append(col)
    
    # Create feature matrix
    X_numeric = df[feature_cols].copy()
    
    # One-hot encode categoricals
    X_categorical = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, drop_first=True)
    
    X = pd.concat([X_numeric, X_categorical], axis=1)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"Features: {len(X.columns)}")
    print(f"  - Numeric: {len(feature_cols)}")
    print(f"  - Categorical (one-hot): {len(X_categorical.columns)}")
    
    return X, y, weights


def stratified_split(X, y, weights, stratify_var=None, test_size=0.2, val_size=0.2):
    """
    Split data into train/validation/test sets with optional stratification.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    weights : pd.Series
        Sample weights
    stratify_var : pd.Series, optional
        Variable to stratify on (e.g., region, climate zone)
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set (of remaining after test split)
        
    Returns
    -------
    dict
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test
    """
    print(f"\nSplitting data (train/val/test: {1-test_size-val_size:.0%}/{val_size:.0%}/{test_size:.0%})...")
    
    # First split: train+val vs test
    if stratify_var is not None:
        X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
            X, y, weights,
            test_size=test_size,
            stratify=stratify_var,
            random_state=42
        )
    else:
        X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
            X, y, weights,
            test_size=test_size,
            random_state=42
        )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    if stratify_var is not None:
        stratify_temp = stratify_var.loc[X_temp.index] if hasattr(stratify_var, 'loc') else None
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_temp, y_temp, w_temp,
            test_size=val_size_adjusted,
            stratify=stratify_temp,
            random_state=42
        )
    else:
        X_train, X_val, y_val, y_train, w_val, w_train = train_test_split(
            X_temp, y_temp, w_temp,
            test_size=val_size_adjusted,
            random_state=42
        )
    
    print(f"Train: {len(X_train):,} samples")
    print(f"Validation: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'w_train': w_train, 'w_val': w_val, 'w_test': w_test,
    }


def train_xgboost(X_train, y_train, w_train, X_val, y_val, w_val):
    """
    Train XGBoost regressor.
    
    Parameters
    ----------
    X_train, y_train, w_train : Training data
    X_val, y_val, w_val : Validation data
        
    Returns
    -------
    xgb.XGBRegressor
        Trained model
    """
    print("\nTraining XGBoost model...")
    
    # XGBoost parameters (tune as needed)
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        early_stopping_rounds=50,
        verbose=100
    )
    
    print("✓ Model training complete")
    
    return model


def evaluate_model(model, X, y, weights=None, name=""):
    """
    Evaluate model performance.
    
    Parameters
    ----------
    model : Trained model
    X, y : Test data
    weights : Sample weights
    name : Name for reporting
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred, sample_weight=weights))
    mae = mean_absolute_error(y, y_pred, sample_weight=weights)
    r2 = r2_score(y, y_pred, sample_weight=weights)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }
    
    if name:
        print(f"\n{name} Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
    
    return metrics


def evaluate_by_subgroups(model, X_test, y_test, w_test, df_test, division_var, envelope_var):
    """
    Evaluate model performance by subgroups.
    
    Parameters
    ----------
    model : Trained model
    X_test, y_test, w_test : Test data
    df_test : Original dataframe for test set
    division_var : Variable name for division
    envelope_var : Variable name for envelope class
        
    Returns
    -------
    pd.DataFrame
        Metrics by subgroup
    """
    print("\nEvaluating by subgroups...")
    
    results = []
    
    # Overall
    metrics = evaluate_model(model, X_test, y_test, w_test, "Overall")
    results.append({
        'group_type': 'overall',
        'group_value': 'all',
        **metrics
    })
    
    # By division
    if division_var in df_test.columns:
        for div in df_test[division_var].unique():
            mask = df_test[division_var] == div
            if mask.sum() > 10:  # Minimum sample size
                metrics = evaluate_model(
                    model,
                    X_test[mask],
                    y_test[mask],
                    w_test[mask] if w_test is not None else None,
                    f"Division {div}"
                )
                results.append({
                    'group_type': 'division',
                    'group_value': str(div),
                    **metrics
                })
    
    # By envelope class
    if envelope_var in df_test.columns:
        for env_class in df_test[envelope_var].unique():
            mask = df_test[envelope_var] == env_class
            if mask.sum() > 10:
                metrics = evaluate_model(
                    model,
                    X_test[mask],
                    y_test[mask],
                    w_test[mask] if w_test is not None else None,
                    f"Envelope {env_class}"
                )
                results.append({
                    'group_type': 'envelope',
                    'group_value': str(env_class),
                    **metrics
                })
    
    results_df = pd.DataFrame(results)
    return results_df


def main():
    """Main modeling pipeline."""
    print("=" * 70)
    print("RECS 2020 Heat Pump Retrofit Project - XGBoost Modeling")
    print("=" * 70)
    
    # Load data
    df = load_data(CLEANED_DATA)
    
    # Prepare features and target
    X, y, weights = prepare_features_target(df)
    
    # Stratified split
    stratify_var = df.loc[X.index, VARIABLES['division']] if VARIABLES['division'] in df.columns else None
    splits = stratified_split(X, y, weights, stratify_var=stratify_var)
    
    # Train model
    model = train_xgboost(
        splits['X_train'], splits['y_train'], splits['w_train'],
        splits['X_val'], splits['y_val'], splits['w_val']
    )
    
    # Evaluate on test set
    test_metrics = evaluate_model(
        model, splits['X_test'], splits['y_test'], splits['w_test'], "Test Set"
    )
    
    # Evaluate by subgroups
    df_test = df.loc[splits['X_test'].index].copy()
    subgroup_metrics = evaluate_by_subgroups(
        model, splits['X_test'], splits['y_test'], splits['w_test'],
        df_test, VARIABLES['division'], VARIABLES['envelope_class']
    )
    
    # Save model
    model_file = MODELS_DIR / "xgboost_thermal_intensity.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved to {model_file}")
    
    # Save metrics
    metrics_file = OUTPUT_DIR / "table3_xgboost_performance.csv"
    subgroup_metrics.to_csv(metrics_file, index=False)
    print(f"✓ Performance metrics saved to {metrics_file}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_file = OUTPUT_DIR / "feature_importance.csv"
    feature_importance.to_csv(importance_file, index=False)
    print(f"✓ Feature importance saved to {importance_file}")
    
    print("\n" + "=" * 70)
    print("Modeling complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review performance metrics")
    print("2. Proceed to Step 4: SHAP analysis")


if __name__ == "__main__":
    main()
