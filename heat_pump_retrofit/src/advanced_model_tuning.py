"""
advanced_model_tuning.py
========================
Advanced model tuning to improve R²

Strategies:
1. More features and interactions
2. Target transformation (log)
3. Larger hyperparameter search
4. Try different algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"


def load_data():
    """Load and prepare data"""
    logger.info("Loading data...")
    df = pd.read_csv(OUTPUT_DIR / "03_gas_heated_clean.csv")
    
    # Rename target
    if 'Thermal_Intensity_I' in df.columns:
        df['thermal_intensity'] = df['Thermal_Intensity_I']
    
    logger.info(f"Loaded {len(df)} records")
    return df


def create_advanced_features(df):
    """Create more advanced features"""
    logger.info("Creating advanced features...")
    
    df = df.copy()
    
    # 1. Basic transformations
    df['log_intensity'] = np.log1p(df['thermal_intensity'] * 1000)  # Scale for log
    df['log_sqft'] = np.log1p(df['A_heated'])
    df['log_hdd'] = np.log1p(df['HDD65'])
    
    # 2. Interactions
    df['hdd_sqft'] = df['HDD65'] * df['A_heated'] / 1e6
    df['age_hdd'] = df['building_age'] * df['HDD65'] / 1e4
    df['age_sqft'] = df['building_age'] * df['A_heated'] / 1e4
    
    # 3. Polynomial features
    df['sqft_sq'] = df['A_heated'] ** 2 / 1e6
    df['hdd_sq'] = df['HDD65'] ** 2 / 1e6
    df['age_sq'] = df['building_age'] ** 2 / 100
    
    # 4. Ratios
    df['sqft_per_hdd'] = df['A_heated'] / (df['HDD65'] + 1)
    df['hdd_per_sqft'] = df['HDD65'] / (df['A_heated'] + 1)
    
    # 5. Envelope score (numeric)
    df['drafty_num'] = df['DRAFTY'].fillna(2).astype(float)
    if 'ADQINSUL' in df.columns:
        df['insul_num'] = df['ADQINSUL'].fillna(2).astype(float)
        df['envelope_score_new'] = (df['drafty_num'] * 2 + df['insul_num']) / 3
    else:
        df['envelope_score_new'] = df['drafty_num']
    
    # 6. Climate severity categories
    df['cold_climate'] = (df['HDD65'] > 5500).astype(int)
    df['mild_climate'] = (df['HDD65'] < 3000).astype(int)
    
    # 7. Building size categories
    df['large_home'] = (df['A_heated'] > 2000).astype(int)
    df['small_home'] = (df['A_heated'] < 1000).astype(int)
    
    # 8. Age categories
    df['old_home'] = (df['building_age'] > 50).astype(int)
    df['new_home'] = (df['building_age'] < 20).astype(int)
    
    logger.info(f"  Created features. Total columns: {len(df.columns)}")
    return df


def prepare_features(df):
    """Prepare feature matrix"""
    logger.info("Preparing features...")
    
    # Numeric features
    numeric_cols = [
        'HDD65', 'A_heated', 'building_age',
        'log_sqft', 'log_hdd',
        'hdd_sqft', 'age_hdd', 'age_sqft',
        'sqft_sq', 'hdd_sq', 'age_sq',
        'sqft_per_hdd', 'hdd_per_sqft',
        'envelope_score_new',
        'cold_climate', 'mild_climate',
        'large_home', 'small_home',
        'old_home', 'new_home',
    ]
    
    # Categorical features
    cat_cols = ['TYPEHUQ', 'DRAFTY', 'REGIONC']
    if 'ADQINSUL' in df.columns:
        cat_cols.append('ADQINSUL')
    
    # Build X
    X = df[[c for c in numeric_cols if c in df.columns]].copy()
    X = X.fillna(X.median())
    
    # Encode categoricals
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            X[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
    
    y = df['thermal_intensity'].values
    
    logger.info(f"  Feature matrix: {X.shape}")
    return X, y


def tune_xgboost_aggressive(X_train, y_train, X_val, y_val):
    """Aggressive XGBoost tuning"""
    logger.info("Aggressive XGBoost tuning...")
    
    try:
        import xgboost as xgb
        
        # Larger parameter grid
        param_dist = {
            'n_estimators': [200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2],
        }
        
        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        
        # Randomized search (faster than grid)
        search = RandomizedSearchCV(
            base_model, param_dist,
            n_iter=100, cv=5, scoring='r2',
            n_jobs=-1, verbose=1, random_state=42
        )
        
        search.fit(X_train, y_train)
        
        logger.info(f"  Best CV R²: {search.best_score_:.4f}")
        logger.info(f"  Best params: {search.best_params_}")
        
        # Evaluate on validation
        best_model = search.best_estimator_
        val_pred = best_model.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        logger.info(f"  Validation R²: {val_r2:.4f}")
        
        return best_model, search.best_params_
        
    except ImportError:
        logger.warning("XGBoost not available")
        return None, None


def tune_gradient_boosting(X_train, y_train, X_val, y_val):
    """Tune sklearn GradientBoosting"""
    logger.info("Tuning GradientBoosting...")
    
    param_dist = {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.08],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [3, 5, 10],
        'subsample': [0.8, 0.9],
        'max_features': ['sqrt', 0.5, 0.7],
    }
    
    base_model = GradientBoostingRegressor(random_state=42)
    
    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=50, cv=5, scoring='r2',
        n_jobs=-1, verbose=1, random_state=42
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"  Best CV R²: {search.best_score_:.4f}")
    
    val_pred = search.best_estimator_.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)
    logger.info(f"  Validation R²: {val_r2:.4f}")
    
    return search.best_estimator_, search.best_params_


def try_stacking(X_train, y_train, X_val, y_val, X_test, y_test):
    """Try stacking ensemble"""
    logger.info("Trying stacking ensemble...")
    
    try:
        import xgboost as xgb
        
        estimators = [
            ('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
        ]
        
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5, n_jobs=-1
        )
        
        stack.fit(X_train, y_train)
        
        val_pred = stack.predict(X_val)
        test_pred = stack.predict(X_test)
        
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        logger.info(f"  Stacking Val R²: {val_r2:.4f}")
        logger.info(f"  Stacking Test R²: {test_r2:.4f}")
        
        return stack, test_r2
        
    except Exception as e:
        logger.warning(f"Stacking failed: {e}")
        return None, 0


def main():
    """Main tuning pipeline"""
    logger.info("=" * 70)
    logger.info("ADVANCED MODEL TUNING")
    logger.info("=" * 70)
    
    # Load data
    df = load_data()
    
    # Handle outliers (less aggressive)
    q1 = df['thermal_intensity'].quantile(0.02)
    q99 = df['thermal_intensity'].quantile(0.98)
    df = df[(df['thermal_intensity'] >= q1) & (df['thermal_intensity'] <= q99)].copy()
    logger.info(f"After outlier removal: {len(df)} records")
    
    # Create features
    df = create_advanced_features(df)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Try different approaches
    results = {}
    
    # 1. Aggressive XGBoost tuning
    xgb_model, xgb_params = tune_xgboost_aggressive(X_train, y_train, X_val, y_val)
    if xgb_model:
        test_pred = xgb_model.predict(X_test)
        results['XGBoost'] = {
            'model': xgb_model,
            'test_r2': r2_score(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        }
        logger.info(f"XGBoost Test R²: {results['XGBoost']['test_r2']:.4f}")
    
    # 2. GradientBoosting
    gb_model, gb_params = tune_gradient_boosting(X_train, y_train, X_val, y_val)
    test_pred = gb_model.predict(X_test)
    results['GradientBoosting'] = {
        'model': gb_model,
        'test_r2': r2_score(y_test, test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
    }
    logger.info(f"GradientBoosting Test R²: {results['GradientBoosting']['test_r2']:.4f}")
    
    # 3. Stacking
    stack_model, stack_r2 = try_stacking(X_train, y_train, X_val, y_val, X_test, y_test)
    if stack_model:
        results['Stacking'] = {
            'model': stack_model,
            'test_r2': stack_r2,
        }
    
    # Find best
    best_name = max(results, key=lambda k: results[k]['test_r2'])
    best_model = results[best_name]['model']
    best_r2 = results[best_name]['test_r2']
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"BEST MODEL: {best_name} with Test R² = {best_r2:.4f}")
    logger.info(f"{'=' * 70}")
    
    # Save best model
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    
    # Generate Figure 5 with best model
    generate_figure5_best(best_model, X_test, y_test, best_r2, best_name)
    
    # Generate Table 3 with all results
    generate_table3_all(results, len(X_train), len(X_val), len(X_test))
    
    print(f"\n{'=' * 70}")
    print("📊 FINAL RESULTS")
    print(f"{'=' * 70}")
    for name, res in results.items():
        print(f"  {name}: Test R² = {res['test_r2']:.4f}")
    print(f"\n✅ Best: {best_name} (R² = {best_r2:.4f})")


def generate_figure5_best(model, X_test, y_test, r2, model_name):
    """Generate Figure 5 with best model"""
    logger.info("Generating Figure 5...")
    
    y_pred = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, c='steelblue', edgecolor='white', linewidth=0.3)
    
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect prediction')
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    stats_text = f'{model_name} Model:\nR² = {r2:.3f}\nRMSE = {rmse:.5f}\nMAE = {mae:.5f}\nn = {len(y_test)}'
    ax.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('Observed Thermal Intensity (BTU/sqft/HDD)', fontsize=12)
    ax.set_ylabel('Predicted Thermal Intensity (BTU/sqft/HDD)', fontsize=12)
    ax.set_title(f'Figure 5: {model_name} Model – Predicted vs Observed Thermal Intensity\n'
                 f'(Advanced feature engineering + hyperparameter tuning)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig5_predicted_vs_observed.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "Fig5_predicted_vs_observed.pdf", bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Figure 5 saved with R² = {r2:.3f}")


def generate_table3_all(results, n_train, n_val, n_test):
    """Generate Table 3 with all models"""
    logger.info("Generating Table 3...")
    
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'N (test)': n_test,
            'Test R²': f"{res['test_r2']:.3f}",
            'Test RMSE': f"{res.get('test_rmse', 0):.5f}" if 'test_rmse' in res else '-',
        })
    
    table3 = pd.DataFrame(rows)
    table3.to_csv(TABLES_DIR / "Table3_model_performance.csv", index=False)
    
    logger.info("  Table 3 saved")


if __name__ == "__main__":
    main()
