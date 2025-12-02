"""
Deep Learning Models for Building Energy Prediction
- LSTM with attention mechanism
- Transformer encoder for sequence modeling
- Multi-task learning: energy prediction + comfort estimation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os


class EnergySequenceDataset(Dataset):
    """PyTorch dataset for energy time series."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMEnergyPredictor(nn.Module):
    """
    LSTM-based energy prediction model with attention mechanism.
    Includes comfort modeling as auxiliary task.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMEnergyPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output heads
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Comfort prediction head (auxiliary task)
        self.comfort_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Comfort index in [-1, 1]
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Predictions
        energy_pred = self.energy_head(context_vector)
        comfort_pred = self.comfort_head(context_vector)
        
        return energy_pred.squeeze(), comfort_pred.squeeze(), attention_weights


class TransformerEnergyPredictor(nn.Module):
    """
    Transformer-based energy prediction model.
    Uses self-attention to capture long-range dependencies.
    """
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1):
        super(TransformerEnergyPredictor, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output heads
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.comfort_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # Use last timestep for prediction (can also use mean pooling)
        last_hidden = transformer_out[:, -1, :]
        
        # Predictions
        energy_pred = self.energy_head(last_hidden)
        comfort_pred = self.comfort_head(last_hidden)
        
        return energy_pred.squeeze(), comfort_pred.squeeze(), transformer_out


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        return self.dropout(x)


class EnergyModelTrainer:
    """
    Trainer class for energy prediction models.
    Handles training, validation, and evaluation.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001, 
              patience=10, save_path='../models'):
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            save_path: Path to save best model
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion_energy = nn.MSELoss()
        criterion_comfort = nn.MSELoss()
        
        os.makedirs(save_path, exist_ok=True)
        patience_counter = 0
        
        print(f"\nTraining on device: {self.device}")
        print("="*80)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss_epoch = 0
            train_energy_loss = 0
            train_comfort_loss = 0
            
            for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                energy_pred, comfort_pred, _ = self.model(batch_X)
                
                # Compute losses
                loss_energy = criterion_energy(energy_pred, batch_y)
                
                # Use synthetic comfort labels (normally would be in data)
                comfort_target = torch.tanh(batch_y / 100)  # Normalize energy to comfort proxy
                loss_comfort = criterion_comfort(comfort_pred, comfort_target)
                
                # Combined loss (weighted multi-task learning)
                loss = loss_energy + 0.3 * loss_comfort
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_epoch += loss.item()
                train_energy_loss += loss_energy.item()
                train_comfort_loss += loss_comfort.item()
            
            avg_train_loss = train_loss_epoch / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Energy: {train_energy_loss/len(train_loader):.4f}, "
                  f"Comfort: {train_comfort_loss/len(train_loader):.4f}")
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                model_name = type(self.model).__name__
                save_file = os.path.join(save_path, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                }, save_file)
                print(f"  → Model saved: {save_file}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        print("="*80)
        print(f"Training complete. Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                energy_pred, comfort_pred, _ = self.model(batch_X)
                loss = criterion(energy_pred, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def predict(self, test_loader, scaler_y=None):
        """
        Make predictions on test set.
        
        Args:
            test_loader: Test data loader
            scaler_y: Target scaler for inverse transform
        
        Returns:
            predictions, actuals, attention_weights (if available)
        """
        self.model.eval()
        predictions = []
        actuals = []
        attention_weights_all = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                
                energy_pred, comfort_pred, attn = self.model(batch_X)
                
                predictions.extend(energy_pred.cpu().numpy())
                actuals.extend(batch_y.numpy())
                
                if attn is not None and len(attn.shape) > 1:
                    attention_weights_all.append(attn.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform if scaler provided
        if scaler_y is not None:
            predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        return predictions, actuals, attention_weights_all
    
    def plot_training_history(self, save_path='../figures'):
        """Plot training and validation loss curves."""
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        plt.title('Training History - Energy Prediction Model', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(save_path, 'training_history_dl.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {filename}")
        plt.close()


def evaluate_model(predictions, actuals, model_name='Model', save_path='../figures'):
    """
    Evaluate model performance with metrics and visualizations.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        model_name: Name of the model
        save_path: Path to save figures
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"  MAE:  {mae:.2f} Wh")
    print(f"  RMSE: {rmse:.2f} Wh")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Create visualization
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter plot
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=10)
    axes[0, 0].plot([actuals.min(), actuals.max()], 
                    [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Energy (Wh)', fontweight='bold')
    axes[0, 0].set_ylabel('Predicted Energy (Wh)', fontweight='bold')
    axes[0, 0].set_title('Predicted vs Actual', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = actuals - predictions
    axes[0, 1].scatter(predictions, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Energy (Wh)', fontweight='bold')
    axes[0, 1].set_ylabel('Residuals (Wh)', fontweight='bold')
    axes[0, 1].set_title('Residual Plot', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time series comparison (first 500 points)
    n_points = min(500, len(actuals))
    axes[1, 0].plot(actuals[:n_points], label='Actual', linewidth=1.5, alpha=0.7)
    axes[1, 0].plot(predictions[:n_points], label='Predicted', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_xlabel('Time Step', fontweight='bold')
    axes[1, 0].set_ylabel('Energy (Wh)', fontweight='bold')
    axes[1, 0].set_title('Time Series Comparison (Sample)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Residual (Wh)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Error Distribution', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{model_name} - Performance Evaluation\n' +
                 f'MAE: {mae:.2f} Wh, RMSE: {rmse:.2f} Wh, R²: {r2:.4f}',
                 fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    filename = os.path.join(save_path, f'{model_name.lower().replace(" ", "_")}_evaluation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Evaluation plot saved: {filename}")
    plt.close()
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


if __name__ == "__main__":
    print("="*80)
    print("Testing Deep Learning Models")
    print("="*80)
    
    # Create synthetic data for testing
    seq_length = 24
    n_features = 15
    n_samples = 1000
    
    X_train = np.random.randn(n_samples, seq_length, n_features)
    y_train = np.random.randn(n_samples)
    X_val = np.random.randn(200, seq_length, n_features)
    y_val = np.random.randn(200)
    
    # Create datasets
    train_dataset = EnergySequenceDataset(X_train, y_train)
    val_dataset = EnergySequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Test LSTM model
    print("\nTesting LSTM Model...")
    lstm_model = LSTMEnergyPredictor(input_size=n_features, hidden_size=64, num_layers=2)
    lstm_trainer = EnergyModelTrainer(lstm_model)
    
    print(f"Model architecture:\n{lstm_model}")
    print(f"Total parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Test Transformer model
    print("\nTesting Transformer Model...")
    transformer_model = TransformerEnergyPredictor(
        input_size=n_features, d_model=64, nhead=4, num_layers=2
    )
    
    print(f"Model architecture:\n{transformer_model}")
    print(f"Total parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    print("\n" + "="*80)
    print("Deep Learning Models Test Complete!")
    print("="*80)
