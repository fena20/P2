"""
Phase 2: Surrogate Model Development (Digital Twin)
Train LSTM and XGBoost models to predict energy consumption and indoor temperature
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json


class SurrogateModel:
    """Digital Twin for predicting building energy consumption and indoor temperature"""
    
    def __init__(self, model_type='lstm'):
        """
        Args:
            model_type: 'lstm' or 'xgboost'
        """
        self.model_type = model_type
        self.model_energy = None
        self.model_temp = None
        self.history = {}
        
        # Feature columns for modeling
        self.input_features = [
            'outdoor_temp', 'solar_radiation', 'humidity', 
            'hour_of_day', 'day_of_week', 'hvac_setpoint'
        ]
        
        self.output_labels = ['energy_consumption', 'indoor_temp']
    
    def prepare_sequences(self, data, sequence_length=24):
        """Prepare sequences for LSTM model (look-back window)"""
        X, y_energy, y_temp = [], [], []
        
        # Group by building to maintain sequence continuity
        for building_id in data['building_id'].unique():
            building_data = data[data['building_id'] == building_id].copy()
            building_data = building_data.sort_values('timestamp')
            
            features = building_data[self.input_features].values
            energy = building_data['energy_consumption'].values
            temp = building_data['indoor_temp'].values
            
            for i in range(len(building_data) - sequence_length):
                X.append(features[i:i+sequence_length])
                y_energy.append(energy[i+sequence_length])
                y_temp.append(temp[i+sequence_length])
        
        return np.array(X), np.array(y_energy), np.array(y_temp)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, model_name='energy'):
        """Train LSTM model"""
        print(f"\nTraining LSTM model for {model_name} prediction...")
        print(f"Training samples: {len(X_train)}")
        print(f"Input shape: {X_train.shape}")
        
        model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.history[model_name] = history.history
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, model_name='energy'):
        """Train XGBoost model"""
        print(f"\nTraining XGBoost model for {model_name} prediction...")
        print(f"Training samples: {len(X_train)}")
        
        # Flatten sequences if needed (XGBoost doesn't handle sequences)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        return model
    
    def train(self, data, sequence_length=24):
        """Train surrogate models"""
        print("="*80)
        print("TRAINING SURROGATE MODELS")
        print("="*80)
        
        if self.model_type == 'lstm':
            # Prepare sequences for LSTM
            X, y_energy, y_temp = self.prepare_sequences(data, sequence_length)
            
            # Split data
            X_train, X_temp, y_energy_train, y_energy_temp = train_test_split(
                X, y_energy, test_size=0.3, random_state=42
            )
            X_val, X_test, y_energy_val, y_energy_test = train_test_split(
                X_temp, y_energy_temp, test_size=0.5, random_state=42
            )
            
            _, _, y_temp_train, y_temp_temp = train_test_split(
                X, y_temp, test_size=0.3, random_state=42
            )
            _, _, y_temp_val, y_temp_test = train_test_split(
                X_temp, y_temp_temp, test_size=0.5, random_state=42
            )
            
            # Train energy consumption model
            self.model_energy = self.train_lstm(
                X_train, y_energy_train, X_val, y_energy_val, 'energy'
            )
            
            # Train indoor temperature model
            self.model_temp = self.train_lstm(
                X_train, y_temp_train, X_val, y_temp_val, 'temperature'
            )
            
        else:  # XGBoost
            # Use last observation as features (no sequence)
            X = data[self.input_features].values
            y_energy = data['energy_consumption'].values
            y_temp = data['indoor_temp'].values
            
            # Split data
            X_train, X_temp, y_energy_train, y_energy_temp = train_test_split(
                X, y_energy, test_size=0.3, random_state=42
            )
            X_val, X_test, y_energy_val, y_energy_test = train_test_split(
                X_temp, y_energy_temp, test_size=0.5, random_state=42
            )
            
            _, _, y_temp_train, y_temp_temp = train_test_split(
                X, y_temp, test_size=0.3, random_state=42
            )
            _, _, y_temp_val, y_temp_test = train_test_split(
                X_temp, y_temp_temp, test_size=0.5, random_state=42
            )
            
            # Train energy consumption model
            self.model_energy = self.train_xgboost(
                X_train, y_energy_train, X_val, y_energy_val, 'energy'
            )
            
            # Train indoor temperature model
            self.model_temp = self.train_xgboost(
                X_train, y_temp_train, X_val, y_temp_val, 'temperature'
            )
        
        # Evaluate models
        metrics = self.evaluate(X_test, y_energy_test, y_temp_test)
        
        return metrics
    
    def evaluate(self, X_test, y_energy_test, y_temp_test):
        """Evaluate model performance"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Flatten if needed for XGBoost
        X_test_flat = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) == 3 and self.model_type == 'xgboost' else X_test
        
        # Predictions
        y_energy_pred = self.model_energy.predict(X_test_flat if self.model_type == 'xgboost' else X_test)
        y_temp_pred = self.model_temp.predict(X_test_flat if self.model_type == 'xgboost' else X_test)
        
        # Flatten predictions if needed
        if len(y_energy_pred.shape) > 1:
            y_energy_pred = y_energy_pred.flatten()
        if len(y_temp_pred.shape) > 1:
            y_temp_pred = y_temp_pred.flatten()
        
        # Calculate metrics
        metrics = {
            'energy': {
                'mae': mean_absolute_error(y_energy_test, y_energy_pred),
                'rmse': np.sqrt(mean_squared_error(y_energy_test, y_energy_pred)),
                'r2': r2_score(y_energy_test, y_energy_pred),
                'mape': np.mean(np.abs((y_energy_test - y_energy_pred) / y_energy_test)) * 100
            },
            'temperature': {
                'mae': mean_absolute_error(y_temp_test, y_temp_pred),
                'rmse': np.sqrt(mean_squared_error(y_temp_test, y_temp_pred)),
                'r2': r2_score(y_temp_test, y_temp_pred),
                'mape': np.mean(np.abs((y_temp_test - y_temp_pred) / y_temp_test)) * 100
            }
        }
        
        # Print results
        print("\nEnergy Consumption Model:")
        print(f"  MAE:  {metrics['energy']['mae']:.4f} kWh")
        print(f"  RMSE: {metrics['energy']['rmse']:.4f} kWh")
        print(f"  R²:   {metrics['energy']['r2']:.4f}")
        print(f"  MAPE: {metrics['energy']['mape']:.2f}%")
        
        print("\nIndoor Temperature Model:")
        print(f"  MAE:  {metrics['temperature']['mae']:.4f} °C")
        print(f"  RMSE: {metrics['temperature']['rmse']:.4f} °C")
        print(f"  R²:   {metrics['temperature']['r2']:.4f}")
        print(f"  MAPE: {metrics['temperature']['mape']:.2f}%")
        
        return metrics
    
    def predict(self, X):
        """Predict energy consumption and indoor temperature"""
        # Flatten if needed for XGBoost
        X_input = X.reshape(X.shape[0], -1) if len(X.shape) == 3 and self.model_type == 'xgboost' else X
        
        energy_pred = self.model_energy.predict(X_input)
        temp_pred = self.model_temp.predict(X_input)
        
        # Flatten predictions
        if len(energy_pred.shape) > 1:
            energy_pred = energy_pred.flatten()
        if len(temp_pred.shape) > 1:
            temp_pred = temp_pred.flatten()
        
        return energy_pred, temp_pred
    
    def save(self, path='results/surrogate_model'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.model_type == 'lstm':
            self.model_energy.save(f'{path}/lstm_energy.h5')
            self.model_temp.save(f'{path}/lstm_temperature.h5')
        else:
            self.model_energy.save_model(f'{path}/xgb_energy.json')
            self.model_temp.save_model(f'{path}/xgb_temperature.json')
        
        # Save history and metadata
        with open(f'{path}/training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nModels saved to {path}/")
    
    def load(self, path='results/surrogate_model'):
        """Load trained models"""
        if self.model_type == 'lstm':
            self.model_energy = keras.models.load_model(f'{path}/lstm_energy.h5')
            self.model_temp = keras.models.load_model(f'{path}/lstm_temperature.h5')
        else:
            self.model_energy = xgb.XGBRegressor()
            self.model_energy.load_model(f'{path}/xgb_energy.json')
            self.model_temp = xgb.XGBRegressor()
            self.model_temp.load_model(f'{path}/xgb_temperature.json')
        
        print(f"Models loaded from {path}/")


def generate_table2():
    """Generate Table 2: Input Variables for the Prediction Model"""
    print("\n" + "="*80)
    print("GENERATING TABLE 2: INPUT VARIABLES")
    print("="*80)
    
    table2_data = {
        'Variable Category': [
            'Environmental',
            'Environmental',
            'Environmental',
            'Temporal',
            'Temporal',
            'Control'
        ],
        'Feature Name': [
            'Outdoor Air Temp',
            'Global Solar Radiation',
            'Relative Humidity',
            'Hour of Day',
            'Day of Week',
            'Cooling/Heating Setpoint'
        ],
        'Unit': [
            '°C',
            'W/m²',
            '%',
            '0-23',
            '1-7',
            '°C'
        ],
        'Source': [
            'BDG2 Weather',
            'BDG2 Weather',
            'BDG2 Weather',
            'Time Feature',
            'Time Feature',
            'Optimization Variable'
        ],
        'Relevance to Proposal': [
            'Climatic data analysis',
            'Impact on heating/cooling load',
            'Latent load calculation',
            "Residents' behavioral patterns",
            'Occupancy schedules',
            'System control parameters'
        ]
    }
    
    table2 = pd.DataFrame(table2_data)
    
    # Save table
    table2.to_csv('tables/table2_input_variables.csv', index=False)
    
    with open('tables/table2_input_variables.txt', 'w') as f:
        f.write("Table 2: Input Variables for the Prediction Model\n")
        f.write("="*120 + "\n\n")
        f.write(table2.to_string(index=False))
    
    print("\nTable 2 saved to tables/")
    print("\n" + table2.to_string(index=False))
    
    return table2


def main():
    """Execute Phase 2: Surrogate Model Development"""
    print("="*80)
    print("PHASE 2: SURROGATE MODEL DEVELOPMENT (DIGITAL TWIN)")
    print("="*80)
    
    # Load processed data
    print("\nLoading processed data...")
    data = pd.read_csv('data/processed_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    # Generate Table 2
    table2 = generate_table2()
    
    # Train LSTM model
    print("\n" + "="*80)
    print("Training LSTM-based Surrogate Model...")
    print("="*80)
    
    lstm_model = SurrogateModel(model_type='lstm')
    lstm_metrics = lstm_model.train(data, sequence_length=24)
    lstm_model.save('results/surrogate_model_lstm')
    
    # Train XGBoost model for comparison
    print("\n" + "="*80)
    print("Training XGBoost-based Surrogate Model...")
    print("="*80)
    
    xgb_model = SurrogateModel(model_type='xgboost')
    xgb_metrics = xgb_model.train(data)
    xgb_model.save('results/surrogate_model_xgboost')
    
    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Metric': ['MAE (kWh)', 'RMSE (kWh)', 'R² Score', 'MAPE (%)'],
        'LSTM': [
            f"{lstm_metrics['energy']['mae']:.4f}",
            f"{lstm_metrics['energy']['rmse']:.4f}",
            f"{lstm_metrics['energy']['r2']:.4f}",
            f"{lstm_metrics['energy']['mape']:.2f}"
        ],
        'XGBoost': [
            f"{xgb_metrics['energy']['mae']:.4f}",
            f"{xgb_metrics['energy']['rmse']:.4f}",
            f"{xgb_metrics['energy']['r2']:.4f}",
            f"{xgb_metrics['energy']['mape']:.2f}"
        ]
    })
    
    print("\nEnergy Consumption Prediction Comparison:")
    print(comparison.to_string(index=False))
    
    comparison.to_csv('results/model_comparison.csv', index=False)
    
    print("\n✓ Phase 2 completed successfully!")
    print(f"\nSurrogate models can predict building behavior instantly:")
    print(f"  - LSTM model R²: {lstm_metrics['energy']['r2']:.4f}")
    print(f"  - XGBoost model R²: {xgb_metrics['energy']['r2']:.4f}")


if __name__ == "__main__":
    main()
