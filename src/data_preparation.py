"""
Data Preparation Module for Edge AI Building Energy Optimization
Handles BDG2 dataset download, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import urllib.request
import os
import warnings
warnings.filterwarnings('ignore')


class BuildingDataProcessor:
    """
    Comprehensive data processor for building energy datasets.
    Supports BDG2 (ASHRAE) and residential energy datasets.
    """
    
    def __init__(self, dataset_type='bdg2', data_dir='../data'):
        """
        Initialize data processor.
        
        Args:
            dataset_type: 'bdg2' for ASHRAE BDG2 or 'residential' for existing data
            data_dir: Directory to store datasets
        """
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.data = None
        self.features = None
        self.target = None
        
        os.makedirs(data_dir, exist_ok=True)
    
    def download_bdg2_sample(self):
        """
        Download BDG2 sample data from ASHRAE.
        Note: Full BDG2 requires registration. Using sample/synthetic data for demonstration.
        """
        print("Downloading BDG2 sample dataset...")
        
        # For demonstration, we'll use the existing energydata_complete.csv
        # and transform it to simulate multi-building BDG2 format
        source_file = '../energydata_complete.csv'
        
        if not os.path.exists(source_file):
            url = "https://raw.githubusercontent.com/Fateme9977/P2/main/energydata_complete.csv"
            urllib.request.urlretrieve(url, source_file)
            print(f"Downloaded base dataset from GitHub")
        
        # Load and transform to multi-building format
        df = pd.read_csv(source_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Simulate 3 buildings with variations
        buildings = []
        for building_id in range(1, 4):
            building_df = df.copy()
            building_df['building_id'] = building_id
            
            # Add building-specific variations
            noise_factor = 0.1 * building_id
            building_df['Appliances'] = building_df['Appliances'] * (1 + noise_factor * np.random.randn(len(building_df)))
            building_df['lights'] = building_df['lights'] * (1 + noise_factor * np.random.randn(len(building_df)))
            
            # Add building metadata
            building_df['building_type'] = ['Office', 'Retail', 'Educational'][building_id - 1]
            building_df['floor_area'] = [5000, 3000, 8000][building_id - 1]
            building_df['year_built'] = [2010, 2005, 2015][building_id - 1]
            
            buildings.append(building_df)
        
        # Combine all buildings
        bdg2_data = pd.concat(buildings, ignore_index=True)
        bdg2_file = os.path.join(self.data_dir, 'bdg2_sample.csv')
        bdg2_data.to_csv(bdg2_file, index=False)
        
        print(f"BDG2 sample dataset created: {len(bdg2_data)} records across 3 buildings")
        print(f"Saved to: {bdg2_file}")
        
        return bdg2_data
    
    def load_data(self):
        """Load dataset based on type."""
        if self.dataset_type == 'bdg2':
            bdg2_file = os.path.join(self.data_dir, 'bdg2_sample.csv')
            if not os.path.exists(bdg2_file):
                self.data = self.download_bdg2_sample()
            else:
                self.data = pd.read_csv(bdg2_file)
                self.data['date'] = pd.to_datetime(self.data['date'])
        else:
            # Load residential dataset
            res_file = '../energydata_complete.csv'
            if not os.path.exists(res_file):
                url = "https://raw.githubusercontent.com/Fateme9977/P2/main/energydata_complete.csv"
                urllib.request.urlretrieve(url, res_file)
            self.data = pd.read_csv(res_file)
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        print(f"Loaded {len(self.data)} records")
        return self.data
    
    def engineer_features(self):
        """Create temporal and comfort-related features."""
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # Temporal features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Comfort metrics (PMV - Predicted Mean Vote approximation)
        # Simplified comfort index based on temperature and humidity
        if 'T2' in df.columns:  # Living room temperature
            df['comfort_index'] = self._calculate_comfort_index(
                df['T2'], df['RH2'] if 'RH2' in df.columns else 50
            )
        
        # Temperature differentials (indicator of HVAC load)
        if 'T_out' in df.columns:
            temp_cols = [col for col in df.columns if col.startswith('T') and 
                        col not in ['T_out', 'Tdewpoint']]
            if temp_cols:
                df['avg_indoor_temp'] = df[temp_cols].mean(axis=1)
                df['temp_differential'] = df['avg_indoor_temp'] - df['T_out']
        
        # Lagged features for time series prediction
        for lag in [1, 2, 3, 6, 12]:  # 10-min intervals
            df[f'energy_lag_{lag}'] = df['Appliances'].shift(lag)
        
        # Rolling statistics
        df['energy_rolling_mean_6'] = df['Appliances'].rolling(window=6, min_periods=1).mean()
        df['energy_rolling_std_6'] = df['Appliances'].rolling(window=6, min_periods=1).std()
        
        # Drop NaN from lagged features
        df = df.dropna()
        
        self.data = df
        print(f"Feature engineering complete. New shape: {df.shape}")
        return df
    
    def _calculate_comfort_index(self, temperature, humidity):
        """
        Calculate thermal comfort index (simplified PMV).
        Range: -3 (cold) to +3 (hot), 0 is neutral
        """
        # Simplified comfort model
        # Optimal: 20-24°C, 40-60% RH
        temp_deviation = (temperature - 22) / 4  # Normalize around 22°C
        humidity_penalty = np.abs(humidity - 50) / 50  # Penalize extreme humidity
        
        comfort = temp_deviation + 0.3 * humidity_penalty
        return np.clip(comfort, -3, 3)
    
    def prepare_dl_dataset(self, sequence_length=24, test_size=0.2, val_size=0.1):
        """
        Prepare dataset for deep learning models (LSTM/Transformer).
        
        Args:
            sequence_length: Number of timesteps in each sequence
            test_size: Fraction for test set
            val_size: Fraction for validation set
        
        Returns:
            Dictionary with train/val/test splits
        """
        if self.data is None or 'hour' not in self.data.columns:
            self.engineer_features()
        
        df = self.data.copy()
        
        # Select features for deep learning
        feature_cols = [
            # Weather features
            'T_out', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint',
            # Indoor conditions
            'T2', 'RH2', 'T3', 'RH3',
            # Temporal features
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
            # Comfort and derived features
            'temp_differential', 'comfort_index',
            # Lagged features
            'energy_lag_1', 'energy_lag_2', 'energy_lag_3',
            'energy_rolling_mean_6', 'energy_rolling_std_6'
        ]
        
        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        target_col = 'Appliances'
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(df) - sequence_length):
            X_sequences.append(df[feature_cols].iloc[i:i+sequence_length].values)
            y_sequences.append(df[target_col].iloc[i+sequence_length])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), shuffle=False
        )
        
        # Normalize
        # Reshape for scaling
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]
        
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_val_2d = X_val.reshape(-1, X_val.shape[-1])
        X_test_2d = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data
        self.scaler_X.fit(X_train_2d)
        self.scaler_y.fit(y_train.reshape(-1, 1))
        
        # Transform
        X_train_scaled = self.scaler_X.transform(X_train_2d).reshape(X_train.shape)
        X_val_scaled = self.scaler_X.transform(X_val_2d).reshape(X_val.shape)
        X_test_scaled = self.scaler_X.transform(X_test_2d).reshape(X_test.shape)
        
        y_train_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"Deep Learning Dataset Prepared:")
        print(f"  Train: {X_train_scaled.shape}, {y_train_scaled.shape}")
        print(f"  Val:   {X_val_scaled.shape}, {y_val_scaled.shape}")
        print(f"  Test:  {X_test_scaled.shape}, {y_test_scaled.shape}")
        
        return {
            'X_train': X_train_scaled,
            'y_train': y_train_scaled,
            'X_val': X_val_scaled,
            'y_val': y_val_scaled,
            'X_test': X_test_scaled,
            'y_test': y_test_scaled,
            'feature_names': feature_cols,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }
    
    def prepare_rl_environment_data(self):
        """
        Prepare data for RL environment simulation.
        Returns state/action/reward trajectories.
        """
        if self.data is None or 'hour' not in self.data.columns:
            self.engineer_features()
        
        df = self.data.copy()
        
        # RL state features (observable by agent)
        state_cols = [
            'T2', 'RH2',  # Indoor conditions
            'T_out', 'RH_out',  # Outdoor weather
            'hour_sin', 'hour_cos',  # Time of day
            'is_weekend',  # Day type
            'energy_lag_1',  # Recent energy usage
            'comfort_index'  # Comfort state
        ]
        
        state_cols = [col for col in state_cols if col in df.columns]
        
        # Actions: HVAC setpoint adjustments (simulated)
        # We'll create synthetic actions based on observed energy patterns
        df['hvac_action'] = self._simulate_hvac_actions(df)
        df['lighting_action'] = self._simulate_lighting_actions(df)
        
        # Rewards: negative energy consumption + comfort penalty
        df['energy_cost'] = df['Appliances'] / 1000  # kWh
        df['comfort_penalty'] = np.abs(df['comfort_index']) * 0.5
        df['reward'] = -(df['energy_cost'] + df['comfort_penalty'])
        
        print(f"RL Environment Data Prepared:")
        print(f"  States: {len(state_cols)} dimensions")
        print(f"  Episodes: {len(df)} timesteps")
        print(f"  Average reward: {df['reward'].mean():.3f}")
        
        return {
            'states': df[state_cols].values,
            'actions_hvac': df['hvac_action'].values,
            'actions_lighting': df['lighting_action'].values,
            'rewards': df['reward'].values,
            'next_states': np.vstack([df[state_cols].values[1:], df[state_cols].values[-1:]]),
            'state_cols': state_cols,
            'data': df
        }
    
    def _simulate_hvac_actions(self, df):
        """Simulate HVAC control actions based on temperature differential."""
        # Action: -1 (cooling), 0 (off), 1 (heating)
        actions = np.zeros(len(df))
        
        if 'temp_differential' in df.columns:
            temp_diff = df['temp_differential'].values
            actions[temp_diff > 2] = -1  # Cooling when indoor > outdoor
            actions[temp_diff < -2] = 1  # Heating when indoor < outdoor
        
        return actions
    
    def _simulate_lighting_actions(self, df):
        """Simulate lighting control actions based on time and occupancy."""
        # Action: 0 (off), 1 (on)
        actions = np.zeros(len(df))
        
        hours = df['hour'].values
        # Lights on during typical occupancy hours (6am-11pm)
        actions[(hours >= 6) & (hours <= 23)] = 1
        
        # Add some randomness for realism
        actions = actions * (np.random.rand(len(actions)) > 0.2).astype(int)
        
        return actions
    
    def split_federated_data(self, n_clients=3):
        """
        Split data for federated learning simulation.
        Each client represents a different building.
        
        Args:
            n_clients: Number of federated clients (buildings)
        
        Returns:
            List of datasets, one per client
        """
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # If we have building_id, use it
        if 'building_id' in df.columns:
            clients_data = []
            for building_id in df['building_id'].unique()[:n_clients]:
                client_df = df[df['building_id'] == building_id].copy()
                clients_data.append(client_df)
        else:
            # Split temporally
            chunk_size = len(df) // n_clients
            clients_data = [
                df.iloc[i*chunk_size:(i+1)*chunk_size].copy() 
                for i in range(n_clients)
            ]
        
        print(f"Federated Learning Data Split:")
        for i, client_data in enumerate(clients_data):
            print(f"  Client {i+1}: {len(client_data)} records")
        
        return clients_data


if __name__ == "__main__":
    # Test data preparation
    print("="*80)
    print("Testing Building Data Processor")
    print("="*80)
    
    # BDG2 dataset
    processor_bdg2 = BuildingDataProcessor(dataset_type='bdg2', data_dir='../data')
    data_bdg2 = processor_bdg2.load_data()
    data_bdg2 = processor_bdg2.engineer_features()
    
    print("\n" + "="*80)
    print("Preparing Deep Learning Dataset")
    print("="*80)
    dl_data = processor_bdg2.prepare_dl_dataset(sequence_length=24)
    
    print("\n" + "="*80)
    print("Preparing RL Environment Data")
    print("="*80)
    rl_data = processor_bdg2.prepare_rl_environment_data()
    
    print("\n" + "="*80)
    print("Preparing Federated Learning Data")
    print("="*80)
    fl_data = processor_bdg2.split_federated_data(n_clients=3)
    
    print("\n" + "="*80)
    print("Data Preparation Complete!")
    print("="*80)
