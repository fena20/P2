"""
Federated Learning System for Privacy-Preserving Training
Simulates distributed training across multiple buildings without sharing raw data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import copy
from collections import defaultdict

class FederatedClient:
    """
    Represents a client (building) in federated learning system.
    Each client has local data and trains model locally.
    """
    
    def __init__(self, client_id: int, model: nn.Module, 
                 local_data: Tuple[np.ndarray, np.ndarray],
                 device='cuda'):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.local_data = local_data
        self.device = device
        self.local_updates = 0
        
    def train_local(self, num_epochs=5, batch_size=32, lr=0.001):
        """
        Train model on local data.
        
        Returns:
            model_state_dict: Updated model parameters
            num_samples: Number of training samples used
        """
        X_local, y_local = self.local_data
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_local).to(self.device)
        y_tensor = torch.FloatTensor(y_local).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self.model, 'predict_energy'):
                    pred = self.model.predict_energy(batch_X)
                else:
                    pred, _ = self.model(batch_X)
                
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            if epoch == num_epochs - 1:
                print(f"Client {self.client_id}: Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        self.local_updates += 1
        
        # Return model state dict and number of samples
        return self.model.state_dict(), len(X_local)
    
    def get_model_parameters(self):
        """Get current model parameters."""
        return self.model.state_dict()

class FederatedServer:
    """
    Central server for federated learning.
    Aggregates model updates from clients using Federated Averaging (FedAvg).
    """
    
    def __init__(self, global_model: nn.Module, device='cuda'):
        self.global_model = global_model.to(device)
        self.device = device
        self.round = 0
        
    def federated_averaging(self, client_updates: List[Dict], 
                           client_sample_sizes: List[int]):
        """
        Perform Federated Averaging (FedAvg) aggregation.
        
        Args:
            client_updates: List of model state dicts from clients
            client_sample_sizes: List of number of samples per client
        
        Returns:
            aggregated_state_dict: Aggregated model parameters
        """
        # Calculate total samples
        total_samples = sum(client_sample_sizes)
        
        # Initialize aggregated parameters
        aggregated_state = {}
        
        # Weighted average of parameters
        for key in client_updates[0].keys():
            aggregated_state[key] = torch.zeros_like(client_updates[0][key])
            
            for client_state, num_samples in zip(client_updates, client_sample_sizes):
                weight = num_samples / total_samples
                aggregated_state[key] += weight * client_state[key]
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        self.round += 1
        
        return aggregated_state
    
    def distribute_model(self):
        """Get global model for distribution to clients."""
        return copy.deepcopy(self.global_model.state_dict())

class FederatedLearningTrainer:
    """
    Main federated learning trainer.
    Coordinates training across multiple clients and server.
    """
    
    def __init__(self, global_model: nn.Module, num_clients: int = 5,
                 device='cuda'):
        self.global_model = global_model
        self.num_clients = num_clients
        self.device = device
        
        self.server = FederatedServer(global_model, device)
        self.clients = []
        
    def setup_clients(self, data_splits: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Initialize clients with their local data.
        
        Args:
            data_splits: List of (X, y) tuples for each client
        """
        if len(data_splits) != self.num_clients:
            raise ValueError(f"Expected {self.num_clients} data splits, "
                           f"got {len(data_splits)}")
        
        self.clients = []
        for i, (X_local, y_local) in enumerate(data_splits):
            client = FederatedClient(
                client_id=i,
                model=self.global_model,
                local_data=(X_local, y_local),
                device=self.device
            )
            self.clients.append(client)
        
        print(f"Initialized {len(self.clients)} federated clients")
    
    def split_data_federated(self, X: np.ndarray, y: np.ndarray, 
                            split_method='iid'):
        """
        Split data across clients.
        
        Args:
            X: Features
            y: Targets
            split_method: 'iid' (independent and identically distributed) 
                         or 'non_iid' (non-IID, e.g., by building)
        
        Returns:
            List of (X_client, y_client) tuples
        """
        n_samples = len(X)
        samples_per_client = n_samples // self.num_clients
        
        data_splits = []
        
        if split_method == 'iid':
            # Random shuffle and split
            indices = np.random.permutation(n_samples)
            for i in range(self.num_clients):
                start_idx = i * samples_per_client
                end_idx = (i + 1) * samples_per_client if i < self.num_clients - 1 else n_samples
                client_indices = indices[start_idx:end_idx]
                data_splits.append((X[client_indices], y[client_indices]))
        
        elif split_method == 'non_iid':
            # Non-IID: split by time periods (simulating different buildings)
            # Each client gets data from different time periods
            period_length = n_samples // self.num_clients
            for i in range(self.num_clients):
                start_idx = i * period_length
                end_idx = (i + 1) * period_length if i < self.num_clients - 1 else n_samples
                data_splits.append((X[start_idx:end_idx], y[start_idx:end_idx]))
        
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        return data_splits
    
    def train_federated(self, num_rounds=10, clients_per_round=None,
                       local_epochs=5, local_batch_size=32, local_lr=0.001):
        """
        Train model using federated learning.
        
        Args:
            num_rounds: Number of federated learning rounds
            clients_per_round: Number of clients to select per round (None = all)
            local_epochs: Number of local training epochs per client
            local_batch_size: Batch size for local training
            local_lr: Learning rate for local training
        """
        if clients_per_round is None:
            clients_per_round = self.num_clients
        
        training_history = {
            'round': [],
            'global_loss': [],
            'client_losses': []
        }
        
        print("="*80)
        print("FEDERATED LEARNING TRAINING")
        print("="*80)
        
        for round_num in range(num_rounds):
            print(f"\n--- Federated Round {round_num + 1}/{num_rounds} ---")
            
            # Distribute global model to selected clients
            global_state = self.server.distribute_model()
            selected_clients = np.random.choice(
                self.num_clients, 
                size=min(clients_per_round, self.num_clients),
                replace=False
            )
            
            # Local training on selected clients
            client_updates = []
            client_sample_sizes = []
            client_losses = []
            
            for client_id in selected_clients:
                client = self.clients[client_id]
                client.model.load_state_dict(global_state)
                
                # Local training
                client_state, num_samples = client.train_local(
                    num_epochs=local_epochs,
                    batch_size=local_batch_size,
                    lr=local_lr
                )
                
                client_updates.append(client_state)
                client_sample_sizes.append(num_samples)
                
                # Evaluate client model (optional)
                client.model.eval()
                with torch.no_grad():
                    X_test, y_test = client.local_data
                    X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                    y_test_tensor = torch.FloatTensor(y_test).to(self.device)
                    
                    if hasattr(client.model, 'predict_energy'):
                        pred = client.model.predict_energy(X_test_tensor)
                    else:
                        pred, _ = client.model(X_test_tensor)
                    
                    loss = nn.MSELoss()(pred, y_test_tensor)
                    client_losses.append(loss.item())
            
            # Aggregate updates
            aggregated_state = self.server.federated_averaging(
                client_updates, client_sample_sizes
            )
            
            # Evaluate global model (on a validation set if available)
            avg_client_loss = np.mean(client_losses)
            
            training_history['round'].append(round_num + 1)
            training_history['global_loss'].append(avg_client_loss)
            training_history['client_losses'].append(client_losses)
            
            print(f"Round {round_num + 1} complete. Average client loss: {avg_client_loss:.4f}")
            print(f"Selected clients: {selected_clients}")
        
        print("\n" + "="*80)
        print("FEDERATED LEARNING COMPLETE")
        print("="*80)
        
        return training_history
    
    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate global model on test data."""
        self.server.global_model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.server.global_model, 'predict_energy'):
                pred = self.server.global_model.predict_energy(X_test_tensor)
            else:
                pred, _ = self.server.global_model(X_test_tensor)
            
            loss = nn.MSELoss()(pred, y_test_tensor)
            mae = nn.L1Loss()(pred, y_test_tensor)
        
        return {
            'mse': loss.item(),
            'mae': mae.item(),
            'predictions': pred.cpu().numpy(),
            'targets': y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        }

def simulate_federated_learning(model_class, model_config, X_train, y_train,
                               X_test, y_test, num_clients=5, num_rounds=10,
                               device='cuda'):
    """
    Convenience function to simulate federated learning.
    
    Args:
        model_class: Model class (e.g., LSTMEnergyPredictor)
        model_config: Dictionary of model configuration parameters
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        num_clients: Number of federated clients
        num_rounds: Number of federated rounds
        device: Device to use
    
    Returns:
        trainer: Trained federated learning trainer
        history: Training history
    """
    # Initialize global model
    global_model = model_class(**model_config)
    
    # Create federated trainer
    trainer = FederatedLearningTrainer(
        global_model=global_model,
        num_clients=num_clients,
        device=device
    )
    
    # Split data across clients (non-IID to simulate different buildings)
    data_splits = trainer.split_data_federated(
        X_train, y_train, split_method='non_iid'
    )
    
    # Setup clients
    trainer.setup_clients(data_splits)
    
    # Train federated
    history = trainer.train_federated(num_rounds=num_rounds)
    
    # Evaluate
    results = trainer.evaluate_global_model(X_test, y_test)
    print(f"\nGlobal Model Test Results:")
    print(f"MSE: {results['mse']:.4f}, MAE: {results['mae']:.4f}")
    
    return trainer, history, results
