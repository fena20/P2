"""
04_shap_analysis.py
====================
SHAP Interpretation for Thermal Intensity Model

This module computes SHAP values to interpret the XGBoost model
and identify key drivers of thermal intensity.

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

import shap
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"


def load_model_and_data():
    """Load trained model and processed data."""
    logger.info("Loading model and data...")
    
    # Load model
    model_path = MODELS_DIR / "xgboost_thermal_intensity.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run 03_xgboost_model.py first.")
    model = joblib.load(model_path)
    
    # Load encoders
    encoders_path = MODELS_DIR / "label_encoders.joblib"
    encoders = joblib.load(encoders_path) if encoders_path.exists() else {}
    
    # Load data
    data_path = OUTPUT_DIR / "03_gas_heated_clean.csv"
    df = pd.read_csv(data_path)
    
    return model, encoders, df


def prepare_features_for_shap(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Prepare feature matrix matching the model's expected format."""
    
    # Same feature preparation as in training
    numeric_features = ['HDD65', 'A_heated', 'building_age', 'heating_equip_age', 'log_sqft']
    categorical_features = [
        'TYPEHUQ', 'YEARMADERANGE', 'DRAFTY', 'ADQINSUL', 'TYPEGLASS',
        'EQUIPM', 'REGIONC', 'DIVISION', 'envelope_class', 'climate_zone'
    ]
    
    available_numeric = [f for f in numeric_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    
    X = df[available_numeric + available_categorical].copy()
    
    # Encode categoricals
    for col in available_categorical:
        if col in encoders:
            X[col] = encoders[col].transform(X[col].fillna('missing').astype(str))
        elif X[col].dtype == 'object':
            X[col] = pd.factorize(X[col].fillna('missing'))[0]
        else:
            X[col] = X[col].fillna(-1)
    
    # Fill missing numerics
    for col in available_numeric:
        X[col] = X[col].fillna(X[col].median())
    
    return X


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 5000):
    """
    Compute SHAP values using TreeExplainer.
    
    For large datasets, we sample to keep computation tractable.
    """
    logger.info(f"Computing SHAP values for {min(len(X), sample_size)} samples...")
    
    # Sample if dataset is large
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    logger.info(f"SHAP values computed. Shape: {shap_values.shape}")
    
    return shap_values, X_sample, explainer


def generate_table4_feature_importance(shap_values: np.ndarray, 
                                       feature_names: list) -> pd.DataFrame:
    """
    Generate Table 4: SHAP-based feature importance ranking.
    """
    logger.info("Generating Table 4: SHAP feature importance")
    
    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap,
        'Importance (%)': mean_abs_shap / mean_abs_shap.sum() * 100
    }).sort_values('Mean |SHAP|', ascending=False)
    
    # Add interpretation hints
    interpretations = {
        'HDD65': 'Climate severity - more heating in colder climates',
        'DRAFTY': 'Air leakage - drafty homes lose more heat',
        'ADQINSUL': 'Insulation quality - better insulation reduces heat loss',
        'YEARMADERANGE': 'Building age - older homes typically less efficient',
        'A_heated': 'Floor area - larger homes may have different intensities',
        'TYPEGLASS': 'Window quality - better glazing reduces heat loss',
        'TYPEHUQ': 'Housing type - apartments vs detached homes differ',
        'building_age': 'Years since construction',
        'EQUIPM': 'Heating system type and efficiency',
        'DIVISION': 'Geographic region effects',
        'REGIONC': 'Census region',
        'envelope_class': 'Overall envelope quality classification',
        'climate_zone': 'Climate zone category',
        'log_sqft': 'Log-transformed floor area',
        'heating_equip_age': 'Age of heating equipment',
    }
    
    importance_df['Interpretation'] = importance_df['Feature'].map(
        lambda x: interpretations.get(x, '')
    )
    
    # Save
    importance_df.to_csv(TABLES_DIR / "table4_shap_feature_importance.csv", index=False)
    importance_df[['Feature', 'Mean |SHAP|', 'Importance (%)']].to_latex(
        TABLES_DIR / "table4_shap_feature_importance.tex",
        index=False,
        float_format="%.3f",
        caption="Global feature importance based on mean absolute SHAP values.",
        label="tab:shap_importance"
    )
    
    return importance_df


def generate_figure6_global_importance(shap_values: np.ndarray, 
                                       X_sample: pd.DataFrame):
    """
    Generate Figure 6: Global SHAP feature importance (beeswarm plot).
    """
    logger.info("Generating Figure 6: Global SHAP importance")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title('Global Feature Importance (SHAP Values)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure6_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure6_shap_summary.pdf", bbox_inches='tight')
    plt.close()
    
    # Also create bar plot version
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title('Mean |SHAP| Value (Average Impact on Model Output)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure6_shap_bar.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure6_shap_bar.pdf", bbox_inches='tight')
    plt.close()


def generate_figure7_dependence_plots(shap_values: np.ndarray,
                                      X_sample: pd.DataFrame,
                                      key_features: list = None):
    """
    Generate Figure 7: SHAP dependence plots for key drivers.
    """
    logger.info("Generating Figure 7: SHAP dependence plots")
    
    if key_features is None:
        # Select top features based on importance
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-6:][::-1]
        key_features = [X_sample.columns[i] for i in top_idx if X_sample.columns[i] in X_sample.columns][:3]
    
    # Filter to available features
    key_features = [f for f in key_features if f in X_sample.columns]
    
    if len(key_features) == 0:
        logger.warning("No key features available for dependence plots")
        return
    
    # Create subplot grid
    n_features = min(len(key_features), 3)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
    
    if n_features == 1:
        axes = [axes]
    
    feature_titles = {
        'DRAFTY': '(a) Draftiness',
        'YEARMADERANGE': '(b) Year Built',
        'HDD65': '(c) Heating Degree Days',
        'A_heated': '(d) Heated Floor Area',
        'ADQINSUL': '(e) Insulation Quality',
        'TYPEGLASS': '(f) Window Glass Type',
        'building_age': '(g) Building Age',
        'envelope_class': '(h) Envelope Class',
    }
    
    for i, feature in enumerate(key_features[:n_features]):
        plt.sca(axes[i])
        feature_idx = list(X_sample.columns).index(feature)
        shap.dependence_plot(
            feature_idx, shap_values, X_sample,
            show=False, ax=axes[i]
        )
        axes[i].set_title(feature_titles.get(feature, feature), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure7_shap_dependence.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure7_shap_dependence.pdf", bbox_inches='tight')
    plt.close()
    
    # Individual plots for each key feature
    for feature in key_features[:6]:
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_idx = list(X_sample.columns).index(feature)
        shap.dependence_plot(feature_idx, shap_values, X_sample, show=False, ax=ax)
        ax.set_title(f'SHAP Dependence: {feature}', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_dependence_{feature}.png", dpi=300, bbox_inches='tight')
        plt.close()


def analyze_envelope_class_drivers(shap_values: np.ndarray, X_sample: pd.DataFrame,
                                   df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which features drive thermal intensity for each envelope class.
    """
    logger.info("Analyzing drivers by envelope class...")
    
    results = []
    
    if 'envelope_class' in df.columns:
        # Get envelope classes for sampled data
        sample_idx = X_sample.index
        env_classes = df.loc[sample_idx, 'envelope_class']
        
        for env_class in ['poor', 'medium', 'good']:
            mask = (env_classes == env_class).values
            if mask.sum() > 10:
                class_shap = shap_values[mask]
                mean_abs = np.abs(class_shap).mean(axis=0)
                
                for i, feature in enumerate(X_sample.columns):
                    results.append({
                        'Envelope Class': env_class,
                        'Feature': feature,
                        'Mean |SHAP|': mean_abs[i],
                    })
    
    if results:
        analysis_df = pd.DataFrame(results)
        
        # Pivot for easier viewing
        pivot_df = analysis_df.pivot(
            index='Feature', 
            columns='Envelope Class', 
            values='Mean |SHAP|'
        )
        pivot_df.to_csv(TABLES_DIR / "shap_by_envelope_class.csv")
        
        return pivot_df
    
    return pd.DataFrame()


def generate_interaction_analysis(shap_values: np.ndarray, X_sample: pd.DataFrame):
    """
    Generate SHAP interaction analysis for key feature pairs.
    """
    logger.info("Generating interaction analysis...")
    
    # Top interaction pairs based on typical building physics
    interaction_pairs = [
        ('HDD65', 'DRAFTY'),
        ('HDD65', 'ADQINSUL'),
        ('A_heated', 'DRAFTY'),
        ('YEARMADERANGE', 'DRAFTY'),
    ]
    
    for feat1, feat2 in interaction_pairs:
        if feat1 in X_sample.columns and feat2 in X_sample.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            feat1_idx = list(X_sample.columns).index(feat1)
            
            shap.dependence_plot(
                feat1_idx, shap_values, X_sample,
                interaction_index=feat2, show=False, ax=ax
            )
            ax.set_title(f'SHAP Dependence: {feat1} × {feat2}', fontsize=14)
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / f"shap_interaction_{feat1}_{feat2}.png",
                dpi=300, bbox_inches='tight'
            )
            plt.close()


def generate_waterfall_examples(model, X_sample: pd.DataFrame, df: pd.DataFrame,
                                n_examples: int = 3):
    """
    Generate waterfall plots for example households.
    """
    logger.info("Generating waterfall plot examples...")
    
    explainer = shap.TreeExplainer(model)
    
    # Select diverse examples
    sample_idx = X_sample.index
    
    examples = []
    
    # Try to get one example from each envelope class
    if 'envelope_class' in df.columns:
        for env_class in ['poor', 'medium', 'good']:
            mask = df.loc[sample_idx, 'envelope_class'] == env_class
            if mask.sum() > 0:
                idx = X_sample[mask.values].index[0]
                examples.append((idx, env_class))
    
    if len(examples) == 0:
        examples = [(X_sample.index[i], f'Example {i+1}') for i in range(min(n_examples, len(X_sample)))]
    
    for idx, label in examples[:n_examples]:
        row = X_sample.loc[[idx]]
        shap_values_single = explainer.shap_values(row)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_single[0],
                base_values=explainer.expected_value,
                data=row.iloc[0],
                feature_names=row.columns.tolist()
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall: {label}', fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_waterfall_{label.replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()


def run_shap_analysis() -> dict:
    """
    Main function to run complete SHAP analysis.
    """
    logger.info("=" * 60)
    logger.info("SHAP Interpretation Analysis")
    logger.info("=" * 60)
    
    # Load model and data
    model, encoders, df = load_model_and_data()
    
    # Prepare features
    X = prepare_features_for_shap(df, encoders)
    
    # Compute SHAP values
    shap_values, X_sample, explainer = compute_shap_values(model, X)
    
    # Table 4: Feature importance
    importance_df = generate_table4_feature_importance(shap_values, X_sample.columns.tolist())
    
    # Figure 6: Global importance
    generate_figure6_global_importance(shap_values, X_sample)
    
    # Figure 7: Dependence plots
    # Select features based on what's available
    key_features = ['DRAFTY', 'HDD65', 'YEARMADERANGE', 'A_heated', 'ADQINSUL', 'building_age']
    key_features = [f for f in key_features if f in X_sample.columns][:3]
    generate_figure7_dependence_plots(shap_values, X_sample, key_features)
    
    # Envelope class analysis
    envelope_analysis = analyze_envelope_class_drivers(shap_values, X_sample, df)
    
    # Interaction analysis
    generate_interaction_analysis(shap_values, X_sample)
    
    # Waterfall examples
    generate_waterfall_examples(model, X_sample, df)
    
    logger.info("=" * 60)
    logger.info("SHAP analysis complete!")
    logger.info("=" * 60)
    
    return {
        'shap_values': shap_values,
        'X_sample': X_sample,
        'explainer': explainer,
        'importance': importance_df,
        'envelope_analysis': envelope_analysis,
    }


if __name__ == "__main__":
    results = run_shap_analysis()
    
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS SUMMARY")
    print("=" * 60)
    print("\nTop 10 Important Features (by SHAP):")
    print(results['importance'].head(10).to_string())
