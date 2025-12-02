"""
Data Preparation Module for BDG2 Dataset from ASHRAE Competition
Downloads and preprocesses building energy data for Edge AI + Hybrid RL system
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class BDG2DataPreprocessor:
    """
    Preprocessor for BDG2 dataset from ASHRAE Great Energy Predictor III competition.
    Handles downloading, cleaning, feature engineering, and splitting.
    """
    
    def __init__(self, data_dir='data', use_existing=True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.use_existing = use_existing
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        
    def download_bdg2_dataset(self):
        """
        Download BDG2 dataset from ASHRAE competition.
        Note: This is a placeholder - actual download URL may vary.
        For now, we'll use the existing energy dataset and adapt it.
        """
        print("Checking for BDG2 dataset...")
        
        # Check if we already have processed data
        if self.use_existing and (self.data_dir / 'train_processed.csv').exists():
            print("Using existing processed data...")
            return True
            
        # For this implementation, we'll use the existing energy dataset
        # In production, this would download from ASHRAE competition site
        source_file = Path('energydata_complete.csv')
        
        if source_file.exists():
            print(f"Found existing dataset: {source_file}")
            df = pd.read_csv(source_file)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Save raw data
            df.to_csv(self.data_dir / 'raw_data.csv')
            print(f"Saved raw data to {self.data_dir / 'raw_data.csv'}")
            return True
        else:
            print("Warning: No existing dataset found. Please ensure energydata_complete.csv exists.")
            return False
    
    def engineer_features(self, df):
        """
        Create time-based and domain-specific features for energy prediction.
        """
        print("Engineering features...")
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Thermal comfort features (PMV approximation)
        # Using simplified PMV calculation based on temperature and humidity
        indoor_temps = [f'T{i}' for i in range(1, 10) if f'T{i}' in df.columns]
        if indoor_temps:
            df['avg_indoor_temp'] = df[indoor_temps].mean(axis=1)
            df['temp_variance'] = df[indoor_temps].var(axis=1)
        
        # Temperature difference (indoor vs outdoor)
        if 'T_out' in df.columns and 'avg_indoor_temp' in df.columns:
            df['temp_diff'] = df['avg_indoor_temp'] - df['T_out']
        
        # Humidity features
        rh_cols = [col for col in df.columns if col.startswith('RH') and col != 'RH_out']
        if rh_cols:
            df['avg_indoor_rh'] = df[rh_cols].mean(axis=1)
        
        # Energy-related features
        if 'Appliances' in df.columns:
            # Rolling statistics
            df['energy_ma_6h'] = df['Appliances'].rolling(window=6, min_periods=1).mean()
            df['energy_ma_24h'] = df['Appliances'].rolling(window=24, min_periods=1).mean()
            df['energy_std_6h'] = df['Appliances'].rolling(window=6, min_periods=1).std()
        
        # Weather features
        if 'Windspeed' in df.columns:
            df['windspeed_squared'] = df['Windspeed'] ** 2
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    def create_sequences(self, data, sequence_length=24, prediction_horizon=1):
        """
        Create sequences for time-series prediction (LSTM/Transformer).
        
        Args:
            data: DataFrame with features
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of steps ahead to predict
        """
        sequences = []
        targets = []
        
        feature_cols = [col for col in data.columns 
                       if col not in ['Appliances', 'lights']]
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            seq = data[feature_cols].iloc[i:i+sequence_length].values
            target = data['Appliances'].iloc[i+sequence_length:i+sequence_length+prediction_horizon].values
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def preprocess(self, sequence_length=24, prediction_horizon=1, 
                   test_size=0.2, val_size=0.1):
        """
        Main preprocessing pipeline.
        """
        print("="*80)
        print("BDG2 DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Download/load data
        if not self.download_bdg2_dataset():
            raise FileNotFoundError("Dataset not found. Please ensure data is available.")
        
        # Load raw data
        raw_file = self.data_dir / 'raw_data.csv'
        if raw_file.exists():
            df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
        else:
            # Fallback to existing file
            df = pd.read_csv('energydata_complete.csv', index_col=0, parse_dates=True)
            if 'date' in df.columns:
                df.index = pd.to_datetime(df['date'])
                df.drop('date', axis=1, inplace=True, errors='ignore')
        
        print(f"Loaded data: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Feature engineering
        df = self.engineer_features(df)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Separate features and targets
        feature_cols = [col for col in df.columns 
                       if col not in ['Appliances', 'lights']]
        
        # Create sequences
        print(f"\nCreating sequences (lookback={sequence_length}, horizon={prediction_horizon})...")
        X, y = self.create_sequences(df, sequence_length, prediction_horizon)
        
        print(f"Sequences created: X shape={X.shape}, y shape={y.shape}")
        
        # Split data
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        # Temporal split (preserve time order)
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        # Scale features
        n_features = X_train.shape[2]
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = self.scaler_features.fit_transform(X_train_reshaped)
        X_val_scaled = self.scaler_features.transform(X_val_reshaped)
        X_test_scaled = self.scaler_features.transform(X_test_reshaped)
        
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Scale targets
        y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        y_val_scaled = self.scaler_target.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
        y_test_scaled = self.scaler_target.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        
        # Save processed data
        print("\nSaving processed data...")
        np.save(self.data_dir / 'X_train.npy', X_train_scaled)
        np.save(self.data_dir / 'X_val.npy', X_val_scaled)
        np.save(self.data_dir / 'X_test.npy', X_test_scaled)
        np.save(self.data_dir / 'y_train.npy', y_train_scaled)
        np.save(self.data_dir / 'y_val.npy', y_val_scaled)
        np.save(self.data_dir / 'y_test.npy', y_test_scaled)
        
        # Save metadata
        metadata = {
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'n_features': n_features,
            'feature_names': feature_cols,
            'train_size': n_train,
            'val_size': n_val,
            'test_size': n_test
        }
        
        import json
        with open(self.data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nPreprocessing complete!")
        print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
            'y_test': y_test_scaled,
            'metadata': metadata,
            'scaler_features': self.scaler_features,
            'scaler_target': self.scaler_target
        }

if __name__ == "__main__":
    preprocessor = BDG2DataPreprocessor()
    data = preprocessor.preprocess(sequence_length=24, prediction_horizon=1)
    print("\nData preprocessing completed successfully!")
