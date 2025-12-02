"""
LSTM Model for Energy Prediction with Thermal Comfort Modeling
Implements multi-task learning: energy prediction + comfort prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMEnergyPredictor(nn.Module):
    """
    LSTM-based energy predictor with thermal comfort modeling.
    Multi-task learning: predicts energy consumption and thermal comfort (PMV).
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.2, num_features=None):
        super(LSTMEnergyPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Energy prediction head
        self.energy_fc1 = nn.Linear(hidden_size, 64)
        self.energy_fc2 = nn.Linear(64, 32)
        self.energy_output = nn.Linear(32, 1)
        
        # Thermal comfort prediction head (PMV - Predicted Mean Vote)
        self.comfort_fc1 = nn.Linear(hidden_size, 64)
        self.comfort_fc2 = nn.Linear(64, 32)
        self.comfort_output = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            energy_pred: Energy consumption prediction
            comfort_pred: Thermal comfort (PMV) prediction
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Energy prediction branch
        energy = F.relu(self.energy_fc1(last_hidden))
        energy = self.dropout(energy)
        energy = F.relu(self.energy_fc2(energy))
        energy_pred = self.energy_output(energy)
        
        # Comfort prediction branch
        comfort = F.relu(self.comfort_fc1(last_hidden))
        comfort = self.dropout(comfort)
        comfort = F.relu(self.comfort_fc2(comfort))
        comfort_pred = self.comfort_output(comfort)
        
        return energy_pred, comfort_pred
    
    def predict_energy(self, x):
        """Predict only energy consumption."""
        energy_pred, _ = self.forward(x)
        return energy_pred
    
    def predict_comfort(self, x):
        """Predict only thermal comfort."""
        _, comfort_pred = self.forward(x)
        return comfort_pred

class ComfortModel:
    """
    Thermal comfort model based on PMV (Predicted Mean Vote) calculation.
    PMV ranges from -3 (cold) to +3 (hot), with 0 being neutral.
    """
    
    @staticmethod
    def calculate_pmv(temperature, humidity, air_velocity=0.1, 
                     metabolic_rate=1.0, clothing_insulation=0.5):
        """
        Simplified PMV calculation.
        Full PMV requires more parameters, but this provides a reasonable approximation.
        
        Args:
            temperature: Air temperature (°C)
            humidity: Relative humidity (%)
            air_velocity: Air velocity (m/s)
            metabolic_rate: Metabolic rate (met)
            clothing_insulation: Clothing insulation (clo)
        """
        # Simplified PMV calculation
        # Optimal comfort temperature around 22-24°C
        temp_diff = temperature - 23.0
        
        # Humidity effect (optimal around 40-60%)
        rh_effect = np.clip((humidity - 50) / 50, -1, 1) * 0.3
        
        # Combined PMV approximation
        pmv = -temp_diff * 0.15 + rh_effect
        
        # Clamp to PMV range [-3, 3]
        pmv = np.clip(pmv, -3, 3)
        
        return pmv
    
    @staticmethod
    def pmv_to_comfort_level(pmv):
        """Convert PMV to comfort level."""
        if pmv < -2:
            return "Cold"
        elif pmv < -1:
            return "Cool"
        elif pmv < 0.5:
            return "Slightly Cool"
        elif pmv <= 0.5:
            return "Neutral"
        elif pmv < 1:
            return "Slightly Warm"
        elif pmv < 2:
            return "Warm"
        else:
            return "Hot"

def train_lstm_model(model, train_loader, val_loader, num_epochs=50, 
                     lr=0.001, device='cuda', save_path='models/deep_learning/lstm_model.pth'):
    """
    Training function for LSTM model.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    criterion_energy = nn.MSELoss()
    criterion_comfort = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y_energy, batch_y_comfort in train_loader:
            batch_x = batch_x.to(device)
            batch_y_energy = batch_y_energy.to(device)
            batch_y_comfort = batch_y_comfort.to(device)
            
            optimizer.zero_grad()
            pred_energy, pred_comfort = model(batch_x)
            
            loss_energy = criterion_energy(pred_energy, batch_y_energy)
            loss_comfort = criterion_comfort(pred_comfort, batch_y_comfort)
            
            # Combined loss (weighted)
            loss = loss_energy + 0.3 * loss_comfort
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y_energy, batch_y_comfort in val_loader:
                batch_x = batch_x.to(device)
                batch_y_energy = batch_y_energy.to(device)
                batch_y_comfort = batch_y_comfort.to(device)
                
                pred_energy, pred_comfort = model(batch_x)
                
                loss_energy = criterion_energy(pred_energy, batch_y_energy)
                loss_comfort = criterion_comfort(pred_comfort, batch_y_comfort)
                loss = loss_energy + 0.3 * loss_comfort
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved at epoch {epoch+1}')
    
    return train_losses, val_losses
