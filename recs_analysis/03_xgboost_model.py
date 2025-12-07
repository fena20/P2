"""
Step 3 - XGBoost Thermal Intensity Model

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology

This script:
1. Trains XGBoost model to predict thermal intensity
2. Evaluates performance overall and by subgroups
3. Generates Table 3 (model performance metrics)
4. Creates Figure 5 (predicted vs observed scatter plot)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class ThermalIntensityModel:
    """
    XGBoost model for predicting heating thermal intensity
    """
    
    def __init__(self, data_path, output_dir='../recs_output'):
        """
        Initialize model pipeline
        
        Parameters
        ----------
        data_path : str
            Path to prepared RECS data
        output_dir : str
            Output directory for results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / 'models'
        self.figures_dir = self.output_dir / 'figures'
        self.tables_dir = self.output_dir / 'tables'
        
        for d in [self.models_dir, self.figures_dir, self.tables_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.model = None
        self.feature_names = []
        self.label_encoders = {}
        
    def load_data(self):
        """Load prepared RECS data"""
        print("=" * 80)
        print("Loading Prepared RECS Data")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df):,} households")
        
        return self
    
    def prepare_features(self):
        """
        Prepare features for modeling
        """
        print("\n" + "=" * 80)
        print("Preparing Features")
        print("=" * 80)
        
        df = self.df.copy()
        
        # Target variable
        target = 'Thermal_Intensity_I'
        
        if target not in df.columns:
            raise ValueError(f"Target variable '{target}' not found in data")
        
        # Remove rows with missing target
        df = df[df[target].notna()]
        print(f"Samples with valid target: {len(df):,}")
        
        # Define potential features
        # Continuous features
        continuous_features = [
            'heated_sqft', 'hdd65', 'cdd65', 'building_age',
            'total_rooms', 'bedrooms'
        ]
        
        # Categorical features
        categorical_features = [
            'housing_type', 'census_region', 'census_division',
            'drafty', 'envelope_class', 'climate_zone', 
            'size_category', 'main_heating_equipment'
        ]
        
        # Select available features
        available_continuous = [f for f in continuous_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
        
        print(f"\nAvailable continuous features ({len(available_continuous)}):")
        for f in available_continuous:
            print(f"  - {f}")
        
        print(f"\nAvailable categorical features ({len(available_categorical)}):")
        for f in available_categorical:
            print(f"  - {f}")
        
        # Encode categorical features
        df_features = df[available_continuous].copy()
        
        for cat_feat in available_categorical:
            # Fill missing values with 'Unknown'
            df[cat_feat] = df[cat_feat].fillna('Unknown').astype(str)
            
            # Label encode
            le = LabelEncoder()
            df_features[cat_feat] = le.fit_transform(df[cat_feat])
            self.label_encoders[cat_feat] = le
        
        # Handle missing values in continuous features
        for cont_feat in available_continuous:
            median_val = df_features[cont_feat].median()
            df_features[cont_feat] = df_features[cont_feat].fillna(median_val)
        
        self.feature_names = list(df_features.columns)
        
        # Extract features and target
        X = df_features.values
        y = df[target].values
        
        # Store metadata for later use
        metadata_cols = ['NWEIGHT', 'census_division', 'envelope_class', 'climate_zone']
        metadata = df[[c for c in metadata_cols if c in df.columns]].copy()
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        self.X = X
        self.y = y
        self.metadata = metadata
        
        return self
    
    def train_test_split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train/val/test sets
        
        Parameters
        ----------
        test_size : float
            Proportion for test set
        val_size : float
            Proportion for validation set (from remaining after test split)
        random_state : int
            Random seed
        """
        print("\n" + "=" * 80)
        print("Splitting Data")
        print("=" * 80)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
            self.X, self.y, self.metadata,
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
            X_temp, y_temp, meta_temp,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        print(f"Training set:   {len(X_train):,} samples ({len(X_train)/len(self.X)*100:.1f}%)")
        print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(self.X)*100:.1f}%)")
        print(f"Test set:       {len(X_test):,} samples ({len(X_test)/len(self.X)*100:.1f}%)")
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.meta_train, self.meta_val, self.meta_test = meta_train, meta_val, meta_test
        
        return self
    
    def train_model(self, use_sample_weights=True):
        """
        Train XGBoost model
        
        Parameters
        ----------
        use_sample_weights : bool
            Whether to use NWEIGHT as sample weights
        """
        print("\n" + "=" * 80)
        print("Training XGBoost Model")
        print("=" * 80)
        
        # Sample weights
        if use_sample_weights and 'NWEIGHT' in self.meta_train.columns:
            sample_weight_train = self.meta_train['NWEIGHT'].values
            sample_weight_val = self.meta_val['NWEIGHT'].values
            print("Using sample weights (NWEIGHT)")
        else:
            sample_weight_train = None
            sample_weight_val = None
            print("Not using sample weights")
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        print("\nModel parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Train model
        self.model = xgb.XGBRegressor(**params)
        
        self.model.fit(
            self.X_train, self.y_train,
            sample_weight=sample_weight_train,
            eval_set=[(self.X_val, self.y_val)],
            sample_weight_eval_set=[sample_weight_val] if sample_weight_val is not None else None,
            early_stopping_rounds=20,
            verbose=False
        )
        
        print(f"\n✓ Model trained successfully")
        print(f"Best iteration: {self.model.best_iteration}")
        
        # Save model
        model_path = self.models_dir / 'xgboost_thermal_intensity.json'
        self.model.save_model(str(model_path))
        print(f"Model saved to: {model_path}")
        
        return self
    
    def evaluate_model(self):
        """
        Evaluate model performance on test set
        """
        print("\n" + "=" * 80)
        print("Evaluating Model Performance")
        print("=" * 80)
        
        # Predictions
        y_pred_test = self.model.predict(self.X_test)
        
        # Overall metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        mae = mean_absolute_error(self.y_test, y_pred_test)
        r2 = r2_score(self.y_test, y_pred_test)
        
        print("\nOverall Test Set Performance:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.4f}")
        
        # Store predictions
        self.y_pred_test = y_pred_test
        
        # Performance metrics dictionary
        self.performance = {
            'overall': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_samples': len(self.y_test)
            }
        }
        
        return self
    
    def evaluate_by_subgroups(self):
        """
        Evaluate performance by census division and envelope class
        """
        print("\n" + "=" * 80)
        print("Evaluating by Subgroups")
        print("=" * 80)
        
        # By census division
        if 'census_division' in self.meta_test.columns:
            print("\nPerformance by Census Division:")
            divisions = sorted(self.meta_test['census_division'].dropna().unique())
            
            for div in divisions:
                mask = self.meta_test['census_division'] == div
                if mask.sum() > 0:
                    y_true_div = self.y_test[mask]
                    y_pred_div = self.y_pred_test[mask]
                    
                    rmse_div = np.sqrt(mean_squared_error(y_true_div, y_pred_div))
                    mae_div = mean_absolute_error(y_true_div, y_pred_div)
                    r2_div = r2_score(y_true_div, y_pred_div)
                    
                    print(f"  Division {div}: RMSE={rmse_div:.6f}, MAE={mae_div:.6f}, R²={r2_div:.4f}, n={mask.sum()}")
                    
                    self.performance[f'division_{div}'] = {
                        'rmse': rmse_div,
                        'mae': mae_div,
                        'r2': r2_div,
                        'n_samples': int(mask.sum())
                    }
        
        # By envelope class
        if 'envelope_class' in self.meta_test.columns:
            print("\nPerformance by Envelope Class:")
            classes = ['poor', 'medium', 'good']
            
            for cls in classes:
                mask = self.meta_test['envelope_class'] == cls
                if mask.sum() > 0:
                    y_true_cls = self.y_test[mask]
                    y_pred_cls = self.y_pred_test[mask]
                    
                    rmse_cls = np.sqrt(mean_squared_error(y_true_cls, y_pred_cls))
                    mae_cls = mean_absolute_error(y_true_cls, y_pred_cls)
                    r2_cls = r2_score(y_true_cls, y_pred_cls)
                    
                    print(f"  {cls:8s}: RMSE={rmse_cls:.6f}, MAE={mae_cls:.6f}, R²={r2_cls:.4f}, n={mask.sum()}")
                    
                    self.performance[f'envelope_{cls}'] = {
                        'rmse': rmse_cls,
                        'mae': mae_cls,
                        'r2': r2_cls,
                        'n_samples': int(mask.sum())
                    }
        
        return self
    
    def create_table3_performance(self):
        """
        Generate Table 3: Model performance metrics
        """
        print("\n" + "=" * 80)
        print("Generating Table 3: Model Performance")
        print("=" * 80)
        
        # Convert performance dict to DataFrame
        records = []
        for key, metrics in self.performance.items():
            record = {'category': key}
            record.update(metrics)
            records.append(record)
        
        df_table3 = pd.DataFrame(records)
        
        # Save table
        table_path = self.tables_dir / 'table3_model_performance.csv'
        df_table3.to_csv(table_path, index=False)
        print(f"Saved: {table_path}")
        
        # Also save as formatted text
        table_txt_path = self.tables_dir / 'table3_model_performance.txt'
        with open(table_txt_path, 'w') as f:
            f.write("Table 3. Performance metrics (RMSE, MAE, and R²) of the XGBoost\n")
            f.write("thermal intensity model for the overall test set and by subgroups\n")
            f.write("=" * 80 + "\n\n")
            f.write(df_table3.to_string(index=False))
        
        print(f"Saved: {table_txt_path}")
        
        return self
    
    def create_figure5_predicted_vs_observed(self):
        """
        Generate Figure 5: Predicted vs observed thermal intensity
        """
        print("\n" + "=" * 80)
        print("Generating Figure 5: Predicted vs Observed")
        print("=" * 80)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by envelope class if available
        if 'envelope_class' in self.meta_test.columns:
            envelope_classes = self.meta_test['envelope_class'].values
            colors_map = {'poor': '#d62728', 'medium': '#ff7f0e', 'good': '#2ca02c'}
            colors = [colors_map.get(c, '#1f77b4') for c in envelope_classes]
            
            # Plot each class separately for legend
            for cls in ['poor', 'medium', 'good']:
                mask = envelope_classes == cls
                if mask.sum() > 0:
                    ax.scatter(self.y_test[mask], self.y_pred_test[mask],
                             c=colors_map[cls], label=cls.capitalize(), alpha=0.5, s=20)
        else:
            ax.scatter(self.y_test, self.y_pred_test, alpha=0.5, s=20, c='#1f77b4')
        
        # 45-degree line
        min_val = min(self.y_test.min(), self.y_pred_test.min())
        max_val = max(self.y_test.max(), self.y_pred_test.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('Observed Thermal Intensity (BTU/sqft/HDD)', fontsize=13)
        ax.set_ylabel('Predicted Thermal Intensity (BTU/sqft/HDD)', fontsize=13)
        ax.set_title('Figure 5. Predicted vs Observed Heating Thermal Intensity\n(XGBoost Model, Test Set)', 
                    fontsize=14, fontweight='bold')
        
        # Add metrics text
        rmse = self.performance['overall']['rmse']
        mae = self.performance['overall']['mae']
        r2 = self.performance['overall']['r2']
        
        metrics_text = f'RMSE = {rmse:.6f}\nMAE = {mae:.6f}\nR² = {r2:.4f}\nn = {len(self.y_test):,}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=11, fontfamily='monospace')
        
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'figure5_predicted_vs_observed.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.close()
        
        return self
    
    def get_feature_importance(self):
        """
        Extract and save feature importance
        """
        print("\n" + "=" * 80)
        print("Feature Importance")
        print("=" * 80)
        
        importance_scores = self.model.feature_importances_
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(df_importance.head(10).to_string(index=False))
        
        # Save
        importance_path = self.tables_dir / 'feature_importance_xgboost.csv'
        df_importance.to_csv(importance_path, index=False)
        print(f"\nSaved: {importance_path}")
        
        self.feature_importance = df_importance
        
        return self


def main():
    """
    Main execution function
    """
    print("RECS 2020 Heat Pump Retrofit Analysis - XGBoost Modeling")
    print("Author: Fafa (GitHub: Fateme9977)")
    print("=" * 80)
    
    # Path to prepared data
    data_path = '../recs_output/recs2020_gas_heated_prepared.csv'
    
    if not Path(data_path).exists():
        print(f"\n✗ ERROR: Prepared data not found at {data_path}")
        print("Please run 01_data_prep.py first.")
        return
    
    # Initialize and run model
    model = ThermalIntensityModel(data_path=data_path, output_dir='../recs_output')
    
    model.load_data() \
         .prepare_features() \
         .train_test_split_data() \
         .train_model() \
         .evaluate_model() \
         .evaluate_by_subgroups() \
         .create_table3_performance() \
         .create_figure5_predicted_vs_observed() \
         .get_feature_importance()
    
    print("\n" + "=" * 80)
    print("✓ XGBoost modeling completed successfully!")
    print("\nOutputs:")
    print("  - Model: recs_output/models/xgboost_thermal_intensity.json")
    print("  - Table 3: recs_output/tables/table3_model_performance.csv")
    print("  - Figure 5: recs_output/figures/figure5_predicted_vs_observed.png")
    print("\nNext step: Run 04_shap_analysis.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
