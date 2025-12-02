"""
Transformer Model for Energy Prediction with Thermal Comfort Modeling
Implements multi-head attention mechanism for time-series energy prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEnergyPredictor(nn.Module):
    """
    Transformer-based energy predictor with thermal comfort modeling.
    Uses multi-head self-attention for capturing long-range dependencies.
    """
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, max_seq_length=100):
        super(TransformerEnergyPredictor, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Energy prediction head
        self.energy_fc1 = nn.Linear(d_model, 128)
        self.energy_fc2 = nn.Linear(128, 64)
        self.energy_output = nn.Linear(64, 1)
        
        # Thermal comfort prediction head
        self.comfort_fc1 = nn.Linear(d_model, 128)
        self.comfort_fc2 = nn.Linear(128, 64)
        self.comfort_output = nn.Linear(64, 1)
        
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
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)
        
        # Use the last time step
        last_hidden = transformer_out[-1, :, :]  # (batch_size, d_model)
        
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

def train_transformer_model(model, train_loader, val_loader, num_epochs=50,
                           lr=0.0001, device='cuda', save_path='models/deep_learning/transformer_model.pth'):
    """
    Training function for Transformer model.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
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
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved at epoch {epoch+1}')
    
    return train_losses, val_losses
