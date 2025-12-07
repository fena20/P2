"""
Step 4: SHAP Interpretation
============================

Goal: Interpret XGBoost model using SHAP values to understand feature contributions.

This script:
1. Loads trained XGBoost model
2. Computes SHAP values
3. Generates visualizations:
   - Global feature importance
   - Dependence plots for key variables
   - Summary plots
4. Uses insights to refine envelope classes and identify retrofit targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "xgboost_thermal_intensity.pkl"
CLEANED_DATA = OUTPUT_DIR / "recs2020_gas_heated_cleaned.csv"


def load_model_and_data():
    """Load trained model and cleaned data."""
    print("Loading model and data...")
    
    # Load model
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}\n"
            f"Please run 03_xgboost_model.py first"
        )
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Loaded model from {MODEL_FILE}")
    
    # Load data
    df = pd.read_csv(CLEANED_DATA)
    
    # Prepare features (same as in Step 3)
    # TODO: Refactor to use shared feature preparation function
    exclude_cols = ['thermal_intensity', 'NWEIGHT']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    categorical_cols = []
    for col in ['envelope_class', 'DIVISION', 'TYPEHUQ', 'REGIONC']:
        if col in df.columns:
            categorical_cols.append(col)
    
    X_numeric = df[feature_cols].copy()
    X_categorical = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, drop_first=True)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    X = X.fillna(X.median())
    
    # Use subset for SHAP (SHAP can be slow on large datasets)
    n_samples = min(1000, len(X))
    X_sample = X.sample(n=n_samples, random_state=42)
    
    print(f"✓ Prepared feature matrix: {len(X.columns)} features")
    print(f"  Using {n_samples:,} samples for SHAP computation")
    
    return model, X, X_sample, df


def compute_shap_values(model, X_sample):
    """
    Compute SHAP values for the model.
    
    Parameters
    ----------
    model : Trained XGBoost model
    X_sample : pd.DataFrame
        Sample of features for SHAP computation
        
    Returns
    -------
    shap.Explanation
        SHAP values
    """
    print("\nComputing SHAP values...")
    print("  (This may take a few minutes...)")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer(X_sample)
    
    print("✓ SHAP values computed")
    
    return shap_values, explainer


def plot_global_importance(shap_values, X_sample, output_dir):
    """
    Plot global feature importance based on mean absolute SHAP values.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values
    X_sample : pd.DataFrame
        Feature matrix
    output_dir : Path
        Output directory for figures
    """
    print("\nGenerating global feature importance plot...")
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': mean_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Plot top 20 features
    top_n = 20
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_features['mean_abs_shap'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    
    output_file = output_dir / "figure6_shap_global_importance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved to {output_file}")
    
    # Save importance table
    table_file = OUTPUT_DIR / "table4_shap_feature_importance.csv"
    importance_df['normalized_importance'] = (importance_df['mean_abs_shap'] / 
                                             importance_df['mean_abs_shap'].sum() * 100)
    importance_df.to_csv(table_file, index=False)
    print(f"✓ Saved importance table to {table_file}")
    
    return importance_df


def plot_summary(shap_values, X_sample, output_dir):
    """
    Plot SHAP summary plot (beeswarm).
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values
    X_sample : pd.DataFrame
        Feature matrix
    output_dir : Path
        Output directory
    """
    print("\nGenerating SHAP summary plot...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout()
    
    output_file = output_dir / "figure6_shap_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved to {output_file}")


def plot_dependence_plots(shap_values, X_sample, key_features, output_dir):
    """
    Plot SHAP dependence plots for key features.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values
    X_sample : pd.DataFrame
        Feature matrix
    key_features : list
        List of feature names to plot
    output_dir : Path
        Output directory
    """
    print("\nGenerating SHAP dependence plots...")
    
    n_features = len(key_features)
    fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 5))
    
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(key_features):
        if feature in X_sample.columns:
            shap.plots.scatter(
                shap_values[:, feature],
                show=False,
                ax=axes[i]
            )
            axes[i].set_title(f'SHAP Dependence: {feature}', fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f'{feature}\nnot found', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    
    output_file = output_dir / "figure7_shap_dependence.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved to {output_file}")


def identify_key_drivers(importance_df, top_n=10):
    """
    Identify key drivers from SHAP importance.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    top_n : int
        Number of top features to return
        
    Returns
    -------
    list
        List of top feature names
    """
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    print(f"\nTop {top_n} drivers of thermal intensity:")
    for i, feat in enumerate(top_features, 1):
        importance = importance_df[importance_df['feature'] == feat]['mean_abs_shap'].values[0]
        print(f"  {i}. {feat}: {importance:.4f}")
    
    return top_features


def main():
    """Main SHAP analysis pipeline."""
    print("=" * 70)
    print("RECS 2020 Heat Pump Retrofit Project - SHAP Analysis")
    print("=" * 70)
    
    # Load model and data
    model, X, X_sample, df = load_model_and_data()
    
    # Compute SHAP values
    shap_values, explainer = compute_shap_values(model, X_sample)
    
    # Global importance
    importance_df = plot_global_importance(shap_values, X_sample, FIGURES_DIR)
    
    # Summary plot
    plot_summary(shap_values, X_sample, FIGURES_DIR)
    
    # Dependence plots for key features
    # Identify key features (prioritize interpretable ones)
    key_features = []
    for feat in importance_df['feature'].head(10):
        # Prefer original features over one-hot encoded
        if not any(x in feat for x in ['_', 'envelope_class', 'DIVISION', 'TYPEHUQ']):
            key_features.append(feat)
        elif feat.startswith('envelope_class_') or feat.startswith('DIVISION_'):
            key_features.append(feat)
        if len(key_features) >= 3:
            break
    
    # Also try common RECS variables
    common_features = ['DRAFTY', 'YEARMADE', 'HDD65', 'TOTSQFT_EN']
    for feat in common_features:
        if feat in X_sample.columns and feat not in key_features:
            key_features.append(feat)
            if len(key_features) >= 3:
                break
    
    if key_features:
        plot_dependence_plots(shap_values, X_sample, key_features[:3], FIGURES_DIR)
    
    # Identify key drivers
    top_drivers = identify_key_drivers(importance_df, top_n=10)
    
    print("\n" + "=" * 70)
    print("SHAP Analysis Complete!")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. Review feature importance to identify retrofit priorities")
    print("2. Use dependence plots to understand non-linear effects")
    print("3. Refine envelope classes based on SHAP insights")
    print("\nNext steps:")
    print("1. Review SHAP visualizations")
    print("2. Proceed to Step 5: Retrofit & Heat Pump Scenarios")


if __name__ == "__main__":
    main()
