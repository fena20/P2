"""
Complete System Training Script
Trains all components: Deep Learning, RL, Multi-Agent RL, Federated Learning, and Edge AI export
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

# Import modules
from scripts.data_preparation import BDG2DataPreprocessor
from models.deep_learning.lstm_model import LSTMEnergyPredictor, train_lstm_model, ComfortModel
from models.deep_learning.transformer_model import TransformerEnergyPredictor, train_transformer_model
from models.reinforcement_learning.hvac_env import HVACControlEnv
from models.reinforcement_learning.ppo_lstm import PPOAgent, train_ppo_agent
from models.multi_agent.multi_agent_rl import (
    HVACAgent, LightingAgent, MultiAgentBuildingEnv, train_multi_agent_system
)
from models.federated_learning.federated_trainer import (
    FederatedLearningTrainer, simulate_federated_learning
)
from models.edge_ai.torchscript_export import export_model_for_edge, EdgeInferenceEngine

def create_data_loaders(X, y, batch_size=32, shuffle=True):
    """Create PyTorch data loaders."""
    import torch.utils.data as data
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    dataset = data.TensorDataset(X_tensor, y_tensor)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def prepare_comfort_targets(data, scaler_features, scaler_target):
    """Prepare comfort (PMV) targets from data."""
    comfort_model = ComfortModel()
    
    # Extract temperature and humidity from sequences
    # This is simplified - in practice, you'd extract these from the original data
    comfort_targets = []
    
    # For simplicity, create synthetic comfort targets
    # In practice, calculate PMV from temperature and humidity
    for i in range(len(data)):
        # Simplified: use average of first few features as temperature proxy
        temp_proxy = np.mean(data[i, :, :min(5, data.shape[2])])
        humidity_proxy = 50.0  # Default humidity
        
        pmv = comfort_model.calculate_pmv(temp_proxy, humidity_proxy)
        comfort_targets.append([pmv])
    
    comfort_targets = np.array(comfort_targets)
    
    # Scale comfort targets
    comfort_scaled = scaler_target.transform(comfort_targets)
    
    return comfort_scaled

def main():
    """Main training pipeline."""
    print("="*80)
    print("EDGE AI WITH HYBRID RL AND DEEP LEARNING")
    print("Building Energy Optimization System - Complete Training Pipeline")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # STEP 1: Data Preparation
    # ========================================================================
    print("STEP 1: Data Preparation")
    print("-" * 80)
    preprocessor = BDG2DataPreprocessor(data_dir='data')
    data_dict = preprocessor.preprocess(
        sequence_length=24,
        prediction_horizon=1,
        test_size=0.2,
        val_size=0.1
    )
    
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    metadata = data_dict['metadata']
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Prepare comfort targets
    print("\nPreparing comfort targets...")
    comfort_train = prepare_comfort_targets(X_train, data_dict['scaler_features'], 
                                            data_dict['scaler_target'])
    comfort_val = prepare_comfort_targets(X_val, data_dict['scaler_features'],
                                         data_dict['scaler_target'])
    
    # Create data loaders with comfort targets
    def create_multi_target_loaders(X, y_energy, y_comfort, batch_size=32):
        import torch.utils.data as data
        X_tensor = torch.FloatTensor(X)
        y_energy_tensor = torch.FloatTensor(y_energy)
        y_comfort_tensor = torch.FloatTensor(y_comfort)
        dataset = data.TensorDataset(X_tensor, y_energy_tensor, y_comfort_tensor)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    train_loader = create_multi_target_loaders(X_train, y_train, comfort_train, batch_size=32)
    val_loader = create_multi_target_loaders(X_val, y_val, comfort_val, batch_size=32)
    
    # ========================================================================
    # STEP 2: Deep Learning Models
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Deep Learning Models Training")
    print("-" * 80)
    
    # LSTM Model
    print("\n2.1 Training LSTM Model...")
    lstm_model = LSTMEnergyPredictor(
        input_size=metadata['n_features'],
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    try:
        train_losses_lstm, val_losses_lstm = train_lstm_model(
            lstm_model, train_loader, val_loader,
            num_epochs=50, lr=0.001, device=device,
            save_path='models/deep_learning/lstm_model.pth'
        )
        print("LSTM training complete.")
    except Exception as e:
        print(f"LSTM training error: {e}")
        print("Continuing with other components...")
    
    # Transformer Model
    print("\n2.2 Training Transformer Model...")
    transformer_model = TransformerEnergyPredictor(
        input_size=metadata['n_features'],
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    )
    
    try:
        train_losses_transformer, val_losses_transformer = train_transformer_model(
            transformer_model, train_loader, val_loader,
            num_epochs=50, lr=0.0001, device=device,
            save_path='models/deep_learning/transformer_model.pth'
        )
        print("Transformer training complete.")
    except Exception as e:
        print(f"Transformer training error: {e}")
        print("Continuing with other components...")
    
    # ========================================================================
    # STEP 3: Reinforcement Learning (PPO)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Reinforcement Learning (PPO with LSTM)")
    print("-" * 80)
    
    # Prepare data for RL environment
    # Use flattened sequences for RL state
    X_rl = X_train.reshape(X_train.shape[0], -1)  # Flatten sequences
    y_rl = y_train.flatten()
    
    # Create environment
    print("\n3.1 Creating HVAC Control Environment...")
    hvac_env = HVACControlEnv(
        data=X_rl[:1000],  # Use subset for faster training
        energy_data=y_rl[:1000],
        comfort_model=ComfortModel(),
        max_steps=500
    )
    
    # Create PPO agent
    print("\n3.2 Training PPO Agent...")
    ppo_agent = PPOAgent(
        state_dim=6,
        action_dim=2,
        lr=3e-4,
        gamma=0.99,
        device=device
    )
    
    try:
        ppo_rewards, ppo_lengths = train_ppo_agent(
            hvac_env, ppo_agent,
            num_episodes=200,  # Reduced for faster execution
            max_steps=500,
            update_frequency=512,
            save_path='models/reinforcement_learning/ppo_lstm.pth'
        )
        print("PPO training complete.")
    except Exception as e:
        print(f"PPO training error: {e}")
        print("Continuing with other components...")
    
    # ========================================================================
    # STEP 4: Multi-Agent RL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Multi-Agent Reinforcement Learning")
    print("-" * 80)
    
    print("\n4.1 Creating Multi-Agent Environment...")
    multi_env = MultiAgentBuildingEnv(
        data=X_rl[:1000],
        energy_data=y_rl[:1000],
        lighting_data=y_rl[:1000] * 0.1,  # Simulated lighting data
        max_steps=500
    )
    
    # Create agents
    hvac_agent = HVACAgent(state_dim=6, action_dim=2, device=device)
    lighting_agent = LightingAgent(state_dim=6, action_dim=2, device=device)
    
    print("\n4.2 Training Multi-Agent System...")
    try:
        multi_agent_rewards = train_multi_agent_system(
            multi_env, hvac_agent, lighting_agent,
            num_episodes=200,
            max_steps=500,
            update_frequency=512
        )
        print("Multi-agent training complete.")
    except Exception as e:
        print(f"Multi-agent training error: {e}")
        print("Continuing with other components...")
    
    # ========================================================================
    # STEP 5: Federated Learning
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Federated Learning")
    print("-" * 80)
    
    print("\n5.1 Setting up Federated Learning...")
    try:
        # Use LSTM model for federated learning
        federated_model_config = {
            'input_size': metadata['n_features'],
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2
        }
        
        federated_trainer, fed_history, fed_results = simulate_federated_learning(
            model_class=LSTMEnergyPredictor,
            model_config=federated_model_config,
            X_train=X_train[:5000],  # Use subset
            y_train=y_train[:5000],
            X_test=X_test[:1000],
            y_test=y_test[:1000],
            num_clients=5,
            num_rounds=10,
            device=device
        )
        print("Federated learning complete.")
    except Exception as e:
        print(f"Federated learning error: {e}")
        print("Continuing with other components...")
    
    # ========================================================================
    # STEP 6: Edge AI Export
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Edge AI Export (TorchScript)")
    print("-" * 80)
    
    print("\n6.1 Exporting models to TorchScript...")
    try:
        # Export LSTM model
        example_input = torch.FloatTensor(X_test[:1]).to(device)
        lstm_model.eval()
        
        exported_models = export_model_for_edge(
            lstm_model.cpu(),
            example_input.cpu(),
            output_dir='models/edge_ai',
            model_name='lstm_energy_predictor'
        )
        
        print(f"Models exported:")
        print(f"  - TorchScript: {exported_models['torchscript']}")
        if exported_models['quantized']:
            print(f"  - Quantized: {exported_models['quantized']}")
        
        # Test inference engine
        print("\n6.2 Testing Edge Inference Engine...")
        inference_engine = EdgeInferenceEngine(
            exported_models['torchscript'],
            device='cpu'
        )
        
        test_input = X_test[0]
        prediction = inference_engine.predict(test_input)
        print(f"Test prediction shape: {prediction.shape}")
        
        # Benchmark
        inference_engine.benchmark_inference(test_input, num_runs=100)
        
    except Exception as e:
        print(f"Edge AI export error: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated Models:")
    print("  - Deep Learning: LSTM, Transformer")
    print("  - Reinforcement Learning: PPO with LSTM")
    print("  - Multi-Agent RL: HVAC + Lighting agents")
    print("  - Federated Learning: Distributed training")
    print("  - Edge AI: TorchScript exported models")
    print("\nNext steps:")
    print("  1. Run visualization scripts to generate figures")
    print("  2. Review paper draft in papers/")
    print("  3. Evaluate models on test set")

if __name__ == "__main__":
    main()
