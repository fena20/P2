"""
Phase 2: Surrogate Model Development (The "Digital Twin")
Objective: Create a neural network that predicts future energy consumption 
and indoor temperature based on weather and HVAC settings
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
import pickle
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class SurrogateModel:
    """Surrogate model for building energy and temperature prediction"""
    
    def __init__(self, model_type: str = 'lstm', sequence_length: int = 24):
        """
        Args:
            model_type: 'lstm' or 'xgboost'
            sequence_length: Number of hours to use as input sequence (for LSTM)
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.model = None
        self.scalers = None
        self.feature_columns = None
        self.history = None
        
    def prepare_features(self, df: pd.DataFrame, include_hvac_setpoint: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input features and output labels
        
        Input Features: [Outdoor Temp, Solar Radiation, Humidity, Hour of Day, Day of Week, HVAC Setpoint]
        Output Labels: [Next Hour Energy Consumption, Next Hour Indoor Temp]
        """
        df = df.copy()
        
        # Create next-hour targets (shift by -1)
        df['next_energy'] = df['energy_consumption'].shift(-1)
        df['next_indoor_temp'] = df['indoor_temp'].shift(-1)
        
        # Remove last row (no next hour)
        df = df.iloc[:-1]
        
        # Define feature columns
        feature_cols = ['outdoor_temp', 'solar_radiation', 'humidity', 
                       'hour_of_day', 'day_of_week']
        
        if include_hvac_setpoint:
            # If HVAC setpoint not in data, create synthetic one based on indoor temp
            if 'hvac_setpoint' not in df.columns:
                # Simulate HVAC setpoint (typically 1-2°C different from indoor temp)
                df['hvac_setpoint'] = df['indoor_temp'] + np.random.normal(0, 0.5, len(df))
                df['hvac_setpoint'] = np.clip(df['hvac_setpoint'], 19, 26)
            feature_cols.append('hvac_setpoint')
        
        # Extract features and targets
        X = df[feature_cols].values
        y = df[['next_energy', 'next_indoor_temp']].values
        
        # Remove any remaining NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.feature_columns = feature_cols
        
        return X, y
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length-1])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape: Tuple) -> keras.Model:
        """Build LSTM model architecture"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(2)  # Energy consumption and indoor temperature
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_xgboost_model(self) -> xgb.XGBRegressor:
        """Build XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror'
        )
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, 
             epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train the surrogate model"""
        
        if self.model_type == 'lstm':
            # Create sequences
            X_seq, y_seq = self.create_sequences(X, y)
            
            # Split data
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Build model
            self.model = self.build_lstm_model((self.sequence_length, X.shape[1]))
            
            # Train
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Evaluate
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            train_mae = mean_absolute_error(y_train, train_pred, multioutput='raw_values')
            val_mae = mean_absolute_error(y_val, val_pred, multioutput='raw_values')
            
            metrics = {
                'train_mae_energy': train_mae[0],
                'train_mae_temp': train_mae[1],
                'val_mae_energy': val_mae[0],
                'val_mae_temp': val_mae[1]
            }
            
        elif self.model_type == 'xgboost':
            # For XGBoost, use recent history as features
            # Flatten sequences or use rolling features
            X_flat = X
            y_flat = y
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y_flat, test_size=validation_split, random_state=42
            )
            
            # Build model (multi-output)
            self.model = self.build_xgboost_model()
            
            # Train separate models for each output
            self.model_energy = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            )
            self.model_temp = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            )
            
            self.model_energy.fit(X_train, y_train[:, 0])
            self.model_temp.fit(X_train, y_train[:, 1])
            
            # Evaluate
            train_pred_energy = self.model_energy.predict(X_train)
            train_pred_temp = self.model_temp.predict(X_train)
            val_pred_energy = self.model_energy.predict(X_val)
            val_pred_temp = self.model_temp.predict(X_val)
            
            train_pred = np.column_stack([train_pred_energy, train_pred_temp])
            val_pred = np.column_stack([val_pred_energy, val_pred_temp])
            
            train_mae = mean_absolute_error(y_train, train_pred, multioutput='raw_values')
            val_mae = mean_absolute_error(y_val, val_pred, multioutput='raw_values')
            
            metrics = {
                'train_mae_energy': train_mae[0],
                'train_mae_temp': train_mae[1],
                'val_mae_energy': val_mae[0],
                'val_mae_temp': val_mae[1]
            }
        
        print(f"\nTraining completed!")
        print(f"Train MAE - Energy: {metrics['train_mae_energy']:.4f}, Temp: {metrics['train_mae_temp']:.4f}")
        print(f"Val MAE - Energy: {metrics['val_mae_energy']:.4f}, Temp: {metrics['val_mae_temp']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict energy consumption and indoor temperature"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.model_type == 'lstm':
            # Create sequences
            if len(X.shape) == 2:
                # Need to create sequences
                X_seq, _ = self.create_sequences(X, np.zeros((len(X), 2)))
                return self.model.predict(X_seq, verbose=0)
            else:
                return self.model.predict(X, verbose=0)
        
        elif self.model_type == 'xgboost':
            pred_energy = self.model_energy.predict(X)
            pred_temp = self.model_temp.predict(X)
            return np.column_stack([pred_energy, pred_temp])
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model_type == 'lstm':
            self.model.save(filepath)
        else:
            pickle.dump({
                'model_energy': self.model_energy,
                'model_temp': self.model_temp
            }, open(filepath, 'wb'))
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if self.model_type == 'lstm':
            self.model = keras.models.load_model(filepath)
        else:
            models = pickle.load(open(filepath, 'rb'))
            self.model_energy = models['model_energy']
            self.model_temp = models['model_temp']


def generate_input_variables_table() -> pd.DataFrame:
    """Generate Table 2: Input Variables for the Prediction Model"""
    table_data = {
        'Variable Category': [
            'Environmental', 'Environmental', 'Environmental',
            'Temporal', 'Temporal',
            'Control'
        ],
        'Feature Name': [
            'Outdoor Air Temp', 'Global Solar Radiation', 'Relative Humidity',
            'Hour of Day', 'Day of Week',
            'Cooling/Heating Setpoint'
        ],
        'Unit': [
            '°C', 'W/m²', '%',
            '0-23', '1-7',
            '°C'
        ],
        'Source': [
            'BDG2 Weather', 'BDG2 Weather', 'BDG2 Weather',
            'Time Feature', 'Time Feature',
            'Optimization Variable'
        ],
        'Relevance to Proposal': [
            'Climatic data analysis',
            'Impact on heating load',
            'Humidity effects on comfort',
            "Residents' behavioral patterns",
            'Occupancy schedules',
            'System control parameters'
        ]
    }
    
    return pd.DataFrame(table_data)


if __name__ == "__main__":
    # Example usage
    from phase1_data_curation import BDG2DataProcessor
    
    # Load and process data
    processor = BDG2DataProcessor()
    metadata = processor.load_metadata("metadata.csv")
    residential_buildings = processor.filter_residential_buildings()
    
    # Process one building for training
    building_id = residential_buildings['building_id'].iloc[0]
    df, scalers = processor.process_building(building_id)
    
    # Prepare surrogate model
    surrogate = SurrogateModel(model_type='xgboost')  # Faster for demo
    
    # Prepare features
    X, y = surrogate.prepare_features(df)
    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train model
    metrics = surrogate.train(X, y, validation_split=0.2, epochs=30)
    
    # Generate Table 2
    table2 = generate_input_variables_table()
    print("\n" + "="*80)
    print("Table 2: Input Variables for the Prediction Model")
    print("="*80)
    print(table2.to_string(index=False))
    
    # Save model
    surrogate.save_model(f"surrogate_model_{surrogate.model_type}.pkl")
