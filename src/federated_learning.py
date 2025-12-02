"""
Federated Learning for Building Energy Optimization
Privacy-preserving training across multiple buildings
Implements FedAvg algorithm with differential privacy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
from typing import List, Dict
import matplotlib.pyplot as plt
import os


class FederatedClient:
    """
    Federated learning client representing a single building.
    Trains local model on private data.
    """
    
    def __init__(self, client_id, model, train_loader, device='cpu'):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.local_epochs = 0
        
    def local_train(self, epochs=5, learning_rate=0.001, dp_epsilon=None):
        """
        Train model on local data.
        
        Args:
            epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            dp_epsilon: Differential privacy epsilon (None = no DP)
        
        Returns:
            Local model state dict and training metrics
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        local_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                energy_pred, comfort_pred, _ = self.model(batch_X)
                
                # Loss
                loss = criterion(energy_pred, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy noise to gradients if specified
                if dp_epsilon is not None:
                    self._add_dp_noise(dp_epsilon)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            local_losses.append(avg_loss)
            self.local_epochs += 1
        
        return self.model.state_dict(), local_losses
    
    def _add_dp_noise(self, epsilon, delta=1e-5):
        """
        Add differential privacy noise to gradients (Gaussian mechanism).
        
        Args:
            epsilon: Privacy parameter (smaller = more privacy)
            delta: Probability of privacy breach
        """
        sensitivity = 1.0  # L2 sensitivity (assuming gradient clipping to 1.0)
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * sigma
                param.grad += noise
    
    def update_model(self, global_state_dict):
        """Update local model with global model parameters."""
        self.model.load_state_dict(global_state_dict)


class FederatedServer:
    """
    Federated learning server coordinating multiple clients.
    Aggregates local models and manages global model.
    """
    
    def __init__(self, global_model, aggregation_method='fedavg'):
        self.global_model = global_model
        self.aggregation_method = aggregation_method
        self.round_metrics = []
        
    def aggregate_models(self, client_state_dicts, client_data_sizes=None):
        """
        Aggregate local models into global model.
        
        Args:
            client_state_dicts: List of client model state dicts
            client_data_sizes: List of client dataset sizes (for weighted averaging)
        
        Returns:
            Updated global model state dict
        """
        if self.aggregation_method == 'fedavg':
            return self._fedavg_aggregate(client_state_dicts, client_data_sizes)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _fedavg_aggregate(self, client_state_dicts, client_data_sizes=None):
        """
        Federated Averaging (FedAvg) aggregation.
        Weighted average of client models by dataset size.
        """
        # If no data sizes provided, use uniform weights
        if client_data_sizes is None:
            client_data_sizes = [1.0] * len(client_state_dicts)
        
        total_size = sum(client_data_sizes)
        weights = [size / total_size for size in client_data_sizes]
        
        # Initialize aggregated state dict
        global_state_dict = copy.deepcopy(client_state_dicts[0])
        
        # Weighted averaging
        for key in global_state_dict.keys():
            # Zero out
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])
            
            # Weighted sum
            for client_state, weight in zip(client_state_dicts, weights):
                global_state_dict[key] += client_state[key] * weight
        
        # Update global model
        self.global_model.load_state_dict(global_state_dict)
        
        return global_state_dict
    
    def get_global_model_state(self):
        """Return global model state dict."""
        return copy.deepcopy(self.global_model.state_dict())


class FederatedLearningCoordinator:
    """
    Coordinator for federated learning across multiple buildings.
    Manages training rounds, client selection, and evaluation.
    """
    
    def __init__(self, global_model, clients, test_loader=None, device='cpu'):
        self.server = FederatedServer(global_model)
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        
        # Metrics tracking
        self.round_losses = []
        self.test_losses = []
        self.client_participation = {client.client_id: 0 for client in clients}
        
    def train_round(self, local_epochs=5, learning_rate=0.001, 
                   client_fraction=1.0, dp_epsilon=None):
        """
        Execute one round of federated learning.
        
        Args:
            local_epochs: Number of epochs for local training
            learning_rate: Learning rate for local training
            client_fraction: Fraction of clients to select (random sampling)
            dp_epsilon: Differential privacy parameter
        
        Returns:
            Round statistics
        """
        # Select clients for this round
        n_selected = max(1, int(len(self.clients) * client_fraction))
        selected_indices = np.random.choice(len(self.clients), n_selected, replace=False)
        selected_clients = [self.clients[i] for i in selected_indices]
        
        # Distribute global model to selected clients
        global_state = self.server.get_global_model_state()
        for client in selected_clients:
            client.update_model(global_state)
        
        # Local training
        client_state_dicts = []
        client_data_sizes = []
        round_local_losses = []
        
        for client in selected_clients:
            local_state, local_losses = client.local_train(
                epochs=local_epochs,
                learning_rate=learning_rate,
                dp_epsilon=dp_epsilon
            )
            
            client_state_dicts.append(local_state)
            client_data_sizes.append(len(client.train_loader.dataset))
            round_local_losses.extend(local_losses)
            
            self.client_participation[client.client_id] += 1
        
        # Aggregate models
        self.server.aggregate_models(client_state_dicts, client_data_sizes)
        
        # Evaluate on test set if available
        if self.test_loader is not None:
            test_loss = self._evaluate_global_model()
            self.test_losses.append(test_loss)
        else:
            test_loss = None
        
        # Record metrics
        avg_local_loss = np.mean(round_local_losses)
        self.round_losses.append(avg_local_loss)
        
        return {
            'avg_local_loss': avg_local_loss,
            'test_loss': test_loss,
            'n_clients': n_selected
        }
    
    def train(self, n_rounds=50, local_epochs=5, learning_rate=0.001,
              client_fraction=1.0, dp_epsilon=None, verbose=True):
        """
        Train federated model for multiple rounds.
        
        Args:
            n_rounds: Number of federated rounds
            local_epochs: Epochs per local training
            learning_rate: Learning rate
            client_fraction: Fraction of clients per round
            dp_epsilon: Differential privacy parameter
            verbose: Print progress
        
        Returns:
            Training metrics
        """
        print("="*80)
        print("Federated Learning Training")
        print(f"Clients: {len(self.clients)}, Rounds: {n_rounds}")
        if dp_epsilon is not None:
            print(f"Differential Privacy: ε = {dp_epsilon}")
        print("="*80)
        
        for round_num in range(n_rounds):
            round_stats = self.train_round(
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                client_fraction=client_fraction,
                dp_epsilon=dp_epsilon
            )
            
            if verbose and (round_num + 1) % 5 == 0:
                print(f"Round {round_num+1}/{n_rounds} | "
                      f"Local Loss: {round_stats['avg_local_loss']:.4f} | "
                      f"Test Loss: {round_stats['test_loss']:.4f if round_stats['test_loss'] else 'N/A'} | "
                      f"Clients: {round_stats['n_clients']}")
        
        print("="*80)
        print("Federated Training Complete!")
        print("="*80)
        
        return {
            'round_losses': self.round_losses,
            'test_losses': self.test_losses,
            'client_participation': self.client_participation
        }
    
    def _evaluate_global_model(self):
        """Evaluate global model on test set."""
        self.server.global_model.eval()
        criterion = nn.MSELoss()
        test_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                energy_pred, _, _ = self.server.global_model(batch_X)
                loss = criterion(energy_pred, batch_y)
                test_loss += loss.item()
        
        return test_loss / len(self.test_loader)
    
    def save_global_model(self, path):
        """Save global model."""
        torch.save(self.server.global_model.state_dict(), path)
        print(f"Global model saved to {path}")
    
    def plot_training_progress(self, save_path='../figures'):
        """Plot federated learning training progress."""
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training losses
        axes[0].plot(self.round_losses, linewidth=2, color='#2E86AB', label='Local Loss')
        if self.test_losses:
            axes[0].plot(self.test_losses, linewidth=2, color='#A23B72', label='Test Loss')
        axes[0].set_xlabel('Federated Round', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontweight='bold', fontsize=12)
        axes[0].set_title('Federated Learning Progress', fontweight='bold', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Client participation
        client_ids = list(self.client_participation.keys())
        participation_counts = list(self.client_participation.values())
        
        axes[1].bar(range(len(client_ids)), participation_counts, color='#F18F01', alpha=0.7)
        axes[1].set_xlabel('Client ID', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Participation Count', fontweight='bold', fontsize=12)
        axes[1].set_title('Client Participation Distribution', fontweight='bold', fontsize=14)
        axes[1].set_xticks(range(len(client_ids)))
        axes[1].set_xticklabels(client_ids)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = os.path.join(save_path, 'federated_learning_progress.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved: {filename}")
        plt.close()


def compare_federated_vs_centralized(fl_model, centralized_model, test_loader, 
                                     save_path='../figures'):
    """
    Compare federated and centralized model performance.
    
    Args:
        fl_model: Trained federated model
        centralized_model: Trained centralized model
        test_loader: Test data loader
        save_path: Path to save comparison figure
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fl_model.eval()
    centralized_model.eval()
    
    fl_predictions = []
    cent_predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            
            # Federated predictions
            fl_pred, _, _ = fl_model(batch_X)
            fl_predictions.extend(fl_pred.cpu().numpy())
            
            # Centralized predictions
            cent_pred, _, _ = centralized_model(batch_X)
            cent_predictions.extend(cent_pred.cpu().numpy())
            
            actuals.extend(batch_y.numpy())
    
    fl_predictions = np.array(fl_predictions)
    cent_predictions = np.array(cent_predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    fl_mae = mean_absolute_error(actuals, fl_predictions)
    fl_rmse = np.sqrt(mean_squared_error(actuals, fl_predictions))
    fl_r2 = r2_score(actuals, fl_predictions)
    
    cent_mae = mean_absolute_error(actuals, cent_predictions)
    cent_rmse = np.sqrt(mean_squared_error(actuals, cent_predictions))
    cent_r2 = r2_score(actuals, cent_predictions)
    
    print("\nFederated vs Centralized Comparison:")
    print(f"  Federated     - MAE: {fl_mae:.4f}, RMSE: {fl_rmse:.4f}, R²: {fl_r2:.4f}")
    print(f"  Centralized   - MAE: {cent_mae:.4f}, RMSE: {cent_rmse:.4f}, R²: {cent_r2:.4f}")
    
    # Visualization
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plots
    axes[0].scatter(actuals, fl_predictions, alpha=0.5, s=10, label='Federated')
    axes[0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Predicted', fontweight='bold', fontsize=12)
    axes[0].set_title(f'Federated Learning\nMAE: {fl_mae:.4f}, R²: {fl_r2:.4f}', 
                     fontweight='bold', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(actuals, cent_predictions, alpha=0.5, s=10, 
                   color='orange', label='Centralized')
    axes[1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Predicted', fontweight='bold', fontsize=12)
    axes[1].set_title(f'Centralized Learning\nMAE: {cent_mae:.4f}, R²: {cent_r2:.4f}', 
                     fontweight='bold', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Federated vs Centralized Learning Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(save_path, 'federated_vs_centralized.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {filename}")
    plt.close()
    
    return {
        'federated': {'mae': fl_mae, 'rmse': fl_rmse, 'r2': fl_r2},
        'centralized': {'mae': cent_mae, 'rmse': cent_rmse, 'r2': cent_r2}
    }


if __name__ == "__main__":
    print("="*80)
    print("Testing Federated Learning Components")
    print("="*80)
    
    # Test with synthetic data
    from deep_learning_models import LSTMEnergyPredictor, EnergySequenceDataset
    from torch.utils.data import DataLoader
    
    # Create synthetic data for 3 clients
    seq_length = 24
    n_features = 15
    
    print("\nCreating synthetic federated datasets...")
    client_datasets = []
    for i in range(3):
        X = np.random.randn(500, seq_length, n_features)
        y = np.random.randn(500)
        dataset = EnergySequenceDataset(X, y)
        client_datasets.append(dataset)
        print(f"  Client {i+1}: {len(dataset)} samples")
    
    # Create test set
    X_test = np.random.randn(200, seq_length, n_features)
    y_test = np.random.randn(200)
    test_dataset = EnergySequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create global model
    global_model = LSTMEnergyPredictor(input_size=n_features, hidden_size=64, num_layers=2)
    
    # Create clients
    print("\nInitializing federated clients...")
    clients = []
    for i, dataset in enumerate(client_datasets):
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        client_model = copy.deepcopy(global_model)
        client = FederatedClient(f"building_{i+1}", client_model, loader)
        clients.append(client)
        print(f"  Client {i+1} initialized")
    
    # Create coordinator
    print("\nInitializing federated coordinator...")
    coordinator = FederatedLearningCoordinator(
        global_model=global_model,
        clients=clients,
        test_loader=test_loader
    )
    
    print(f"  {len(clients)} clients ready")
    print(f"  Global model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    
    print("\n" + "="*80)
    print("Federated Learning Components Test Complete!")
    print("="*80)
