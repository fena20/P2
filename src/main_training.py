"""
Main Training Script for Edge AI Building Energy Optimization
Orchestrates all components: data preparation, DL models, RL agents, 
federated learning, and edge deployment
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import copy

# Import custom modules
from data_preparation import BuildingDataProcessor
from deep_learning_models import (
    LSTMEnergyPredictor, TransformerEnergyPredictor,
    EnergySequenceDataset, EnergyModelTrainer, evaluate_model
)
from rl_agents import (
    BuildingEnergyEnv, PPOAgent, MultiAgentSystem, train_ppo_agent
)
from federated_learning import (
    FederatedClient, FederatedLearningCoordinator, 
    compare_federated_vs_centralized
)
from edge_ai_deployment import (
    create_edge_deployment_package, generate_edge_deployment_report
)


class EdgeAIExperimentPipeline:
    """
    Complete pipeline for Edge AI building energy optimization experiments.
    """
    
    def __init__(self, config=None):
        """
        Initialize experiment pipeline.
        
        Args:
            config: Configuration dictionary
        """
        # Default configuration
        self.config = {
            'data': {
                'dataset_type': 'bdg2',
                'sequence_length': 24,
                'test_size': 0.2,
                'val_size': 0.1
            },
            'dl_models': {
                'lstm': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'epochs': 50,
                    'batch_size': 64,
                    'learning_rate': 0.001
                },
                'transformer': {
                    'd_model': 128,
                    'nhead': 8,
                    'num_layers': 3,
                    'epochs': 50,
                    'batch_size': 64,
                    'learning_rate': 0.001
                }
            },
            'rl': {
                'ppo': {
                    'hidden_size': 128,
                    'learning_rate': 3e-4,
                    'n_episodes': 500,
                    'max_steps': 1000
                },
                'multiagent': {
                    'hvac_hidden_size': 64,
                    'lighting_hidden_size': 32,
                    'n_episodes': 500,
                    'max_steps': 1000
                }
            },
            'federated': {
                'n_clients': 3,
                'n_rounds': 50,
                'local_epochs': 5,
                'client_fraction': 1.0,
                'dp_epsilon': None  # Set to enable differential privacy
            },
            'paths': {
                'data_dir': '../data',
                'models_dir': '../models',
                'results_dir': '../results',
                'figures_dir': '../figures',
                'logs_dir': '../logs'
            }
        }
        
        # Update with custom config
        if config:
            self._update_config(self.config, config)
        
        # Create directories
        for path in self.config['paths'].values():
            os.makedirs(path, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'data_stats': {},
            'dl_models': {},
            'rl_agents': {},
            'federated': {},
            'edge_deployment': {},
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        print("="*80)
        print("Edge AI Building Energy Optimization - Experiment Pipeline")
        print("="*80)
        print(f"Experiment ID: {self.results['timestamp']}")
    
    def _update_config(self, base, update):
        """Recursively update configuration."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def run_full_pipeline(self):
        """Execute complete experiment pipeline."""
        print("\n" + "="*80)
        print("PHASE 1: Data Preparation")
        print("="*80)
        self.prepare_data()
        
        print("\n" + "="*80)
        print("PHASE 2: Deep Learning Models")
        print("="*80)
        self.train_dl_models()
        
        print("\n" + "="*80)
        print("PHASE 3: Reinforcement Learning Agents")
        print("="*80)
        self.train_rl_agents()
        
        print("\n" + "="*80)
        print("PHASE 4: Federated Learning")
        print("="*80)
        self.train_federated_learning()
        
        print("\n" + "="*80)
        print("PHASE 5: Edge AI Deployment")
        print("="*80)
        self.prepare_edge_deployment()
        
        print("\n" + "="*80)
        print("PHASE 6: Results Compilation")
        print("="*80)
        self.compile_results()
        
        print("\n" + "="*80)
        print("EXPERIMENT PIPELINE COMPLETE!")
        print("="*80)
        
        return self.results
    
    def prepare_data(self):
        """Prepare and preprocess datasets."""
        print("\nInitializing data processor...")
        
        self.data_processor = BuildingDataProcessor(
            dataset_type=self.config['data']['dataset_type'],
            data_dir=self.config['paths']['data_dir']
        )
        
        # Load and engineer features
        data = self.data_processor.load_data()
        data = self.data_processor.engineer_features()
        
        # Prepare DL dataset
        self.dl_data = self.data_processor.prepare_dl_dataset(
            sequence_length=self.config['data']['sequence_length'],
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size']
        )
        
        # Prepare RL data
        self.rl_data = self.data_processor.prepare_rl_environment_data()
        
        # Prepare federated data
        self.fl_data = self.data_processor.split_federated_data(
            n_clients=self.config['federated']['n_clients']
        )
        
        # Store statistics
        self.results['data_stats'] = {
            'total_samples': len(data),
            'train_samples': self.dl_data['X_train'].shape[0],
            'val_samples': self.dl_data['X_val'].shape[0],
            'test_samples': self.dl_data['X_test'].shape[0],
            'n_features': self.dl_data['X_train'].shape[2],
            'sequence_length': self.config['data']['sequence_length'],
            'n_federated_clients': len(self.fl_data)
        }
        
        print(f"\nData preparation complete:")
        print(f"  Total samples: {self.results['data_stats']['total_samples']}")
        print(f"  Training: {self.results['data_stats']['train_samples']}")
        print(f"  Validation: {self.results['data_stats']['val_samples']}")
        print(f"  Test: {self.results['data_stats']['test_samples']}")
    
    def train_dl_models(self):
        """Train deep learning models (LSTM and Transformer)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nTraining device: {device}")
        
        # Create data loaders
        train_dataset = EnergySequenceDataset(
            self.dl_data['X_train'], 
            self.dl_data['y_train']
        )
        val_dataset = EnergySequenceDataset(
            self.dl_data['X_val'], 
            self.dl_data['y_val']
        )
        test_dataset = EnergySequenceDataset(
            self.dl_data['X_test'], 
            self.dl_data['y_test']
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['dl_models']['lstm']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['dl_models']['lstm']['batch_size'],
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['dl_models']['lstm']['batch_size'],
            shuffle=False
        )
        
        n_features = self.dl_data['X_train'].shape[2]
        
        # Train LSTM
        print("\n" + "-"*80)
        print("Training LSTM Model")
        print("-"*80)
        
        lstm_model = LSTMEnergyPredictor(
            input_size=n_features,
            hidden_size=self.config['dl_models']['lstm']['hidden_size'],
            num_layers=self.config['dl_models']['lstm']['num_layers'],
            dropout=self.config['dl_models']['lstm']['dropout']
        )
        
        lstm_trainer = EnergyModelTrainer(lstm_model, device=device)
        lstm_trainer.train(
            train_loader, val_loader,
            epochs=self.config['dl_models']['lstm']['epochs'],
            learning_rate=self.config['dl_models']['lstm']['learning_rate'],
            save_path=self.config['paths']['models_dir']
        )
        lstm_trainer.plot_training_history(self.config['paths']['figures_dir'])
        
        # Evaluate LSTM
        lstm_predictions, lstm_actuals, _ = lstm_trainer.predict(
            test_loader, 
            scaler_y=self.dl_data['scaler_y']
        )
        lstm_metrics = evaluate_model(
            lstm_predictions, lstm_actuals, 
            model_name='LSTM Energy Predictor',
            save_path=self.config['paths']['figures_dir']
        )
        
        self.lstm_model = lstm_model
        self.results['dl_models']['lstm'] = lstm_metrics
        
        # Train Transformer
        print("\n" + "-"*80)
        print("Training Transformer Model")
        print("-"*80)
        
        transformer_model = TransformerEnergyPredictor(
            input_size=n_features,
            d_model=self.config['dl_models']['transformer']['d_model'],
            nhead=self.config['dl_models']['transformer']['nhead'],
            num_layers=self.config['dl_models']['transformer']['num_layers']
        )
        
        transformer_trainer = EnergyModelTrainer(transformer_model, device=device)
        transformer_trainer.train(
            train_loader, val_loader,
            epochs=self.config['dl_models']['transformer']['epochs'],
            learning_rate=self.config['dl_models']['transformer']['learning_rate'],
            save_path=self.config['paths']['models_dir']
        )
        
        # Evaluate Transformer
        transformer_predictions, transformer_actuals, _ = transformer_trainer.predict(
            test_loader,
            scaler_y=self.dl_data['scaler_y']
        )
        transformer_metrics = evaluate_model(
            transformer_predictions, transformer_actuals,
            model_name='Transformer Energy Predictor',
            save_path=self.config['paths']['figures_dir']
        )
        
        self.transformer_model = transformer_model
        self.results['dl_models']['transformer'] = transformer_metrics
        
        print("\nDeep Learning Models Summary:")
        print(f"  LSTM     - MAE: {lstm_metrics['mae']:.2f}, RMSE: {lstm_metrics['rmse']:.2f}, R²: {lstm_metrics['r2']:.4f}")
        print(f"  Transformer - MAE: {transformer_metrics['mae']:.2f}, RMSE: {transformer_metrics['rmse']:.2f}, R²: {transformer_metrics['r2']:.4f}")
    
    def train_rl_agents(self):
        """Train reinforcement learning agents."""
        # Create environment
        env = BuildingEnergyEnv(
            data_processor=self.data_processor,
            max_steps=self.config['rl']['ppo']['max_steps']
        )
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Train single PPO agent
        print("\n" + "-"*80)
        print("Training PPO Agent")
        print("-"*80)
        
        ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self.config['rl']['ppo']['hidden_size'],
            lr=self.config['rl']['ppo']['learning_rate']
        )
        
        ppo_rewards, ppo_energies, ppo_comforts = train_ppo_agent(
            env, ppo_agent,
            n_episodes=self.config['rl']['ppo']['n_episodes'],
            max_steps=self.config['rl']['ppo']['max_steps'],
            save_path=self.config['paths']['models_dir']
        )
        
        self.ppo_agent = ppo_agent
        self.results['rl_agents']['ppo'] = {
            'final_reward': np.mean(ppo_rewards[-10:]),
            'final_energy': np.mean(ppo_energies[-10:]),
            'final_comfort': np.mean(ppo_comforts[-10:])
        }
        
        # Train multi-agent system
        print("\n" + "-"*80)
        print("Training Multi-Agent System")
        print("-"*80)
        
        multi_agent = MultiAgentSystem(
            state_dim=state_dim,
            hvac_action_dim=1,
            lighting_action_dim=1
        )
        
        ma_rewards, ma_energies, ma_comforts = train_ppo_agent(
            env, multi_agent,
            n_episodes=self.config['rl']['multiagent']['n_episodes'],
            max_steps=self.config['rl']['multiagent']['max_steps'],
            save_path=self.config['paths']['models_dir']
        )
        
        self.multi_agent = multi_agent
        self.results['rl_agents']['multiagent'] = {
            'final_reward': np.mean(ma_rewards[-10:]),
            'final_energy': np.mean(ma_energies[-10:]),
            'final_comfort': np.mean(ma_comforts[-10:])
        }
        
        # Plot RL training curves
        self._plot_rl_training(ppo_rewards, ma_rewards, ppo_energies, ma_energies)
        
        print("\nRL Agents Summary:")
        print(f"  PPO         - Reward: {self.results['rl_agents']['ppo']['final_reward']:.2f}, Energy: {self.results['rl_agents']['ppo']['final_energy']:.3f} kWh")
        print(f"  Multi-Agent - Reward: {self.results['rl_agents']['multiagent']['final_reward']:.2f}, Energy: {self.results['rl_agents']['multiagent']['final_energy']:.3f} kWh")
    
    def _plot_rl_training(self, ppo_rewards, ma_rewards, ppo_energies, ma_energies):
        """Plot RL training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Rewards
        axes[0].plot(self._smooth(ppo_rewards, 10), label='PPO', linewidth=2)
        axes[0].plot(self._smooth(ma_rewards, 10), label='Multi-Agent', linewidth=2)
        axes[0].set_xlabel('Episode', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Episode Reward', fontweight='bold', fontsize=12)
        axes[0].set_title('RL Training Progress - Rewards', fontweight='bold', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Energy consumption
        axes[1].plot(self._smooth(ppo_energies, 10), label='PPO', linewidth=2)
        axes[1].plot(self._smooth(ma_energies, 10), label='Multi-Agent', linewidth=2)
        axes[1].set_xlabel('Episode', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Average Energy (kWh)', fontweight='bold', fontsize=12)
        axes[1].set_title('RL Training Progress - Energy Consumption', fontweight='bold', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.config['paths']['figures_dir'], 'rl_training_progress.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"RL training plot saved: {filename}")
        plt.close()
    
    def _smooth(self, data, window=10):
        """Smooth data with moving average."""
        if len(data) < window:
            return data
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        return smoothed
    
    def train_federated_learning(self):
        """Train federated learning model."""
        # Prepare federated datasets
        from torch.utils.data import TensorDataset, DataLoader
        
        clients = []
        n_features = self.dl_data['X_train'].shape[2]
        
        # Create clients with their data
        for i, client_data in enumerate(self.fl_data):
            # Process client data
            client_processor = BuildingDataProcessor(
                dataset_type=self.config['data']['dataset_type'],
                data_dir=self.config['paths']['data_dir']
            )
            client_processor.data = client_data
            client_processor.engineer_features()
            
            client_dl_data = client_processor.prepare_dl_dataset(
                sequence_length=self.config['data']['sequence_length']
            )
            
            # Create dataset and loader
            client_dataset = EnergySequenceDataset(
                client_dl_data['X_train'],
                client_dl_data['y_train']
            )
            client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
            
            # Create client model
            client_model = LSTMEnergyPredictor(
                input_size=n_features,
                hidden_size=64,
                num_layers=2
            )
            
            # Create client
            from federated_learning import FederatedClient
            client = FederatedClient(f"building_{i+1}", client_model, client_loader)
            clients.append(client)
        
        # Create global model
        global_model = LSTMEnergyPredictor(
            input_size=n_features,
            hidden_size=64,
            num_layers=2
        )
        
        # Create test loader
        test_dataset = EnergySequenceDataset(
            self.dl_data['X_test'],
            self.dl_data['y_test']
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train federated model
        print("\n" + "-"*80)
        print("Training Federated Learning Model")
        print("-"*80)
        
        from federated_learning import FederatedLearningCoordinator
        
        fl_coordinator = FederatedLearningCoordinator(
            global_model=global_model,
            clients=clients,
            test_loader=test_loader
        )
        
        fl_metrics = fl_coordinator.train(
            n_rounds=self.config['federated']['n_rounds'],
            local_epochs=self.config['federated']['local_epochs'],
            client_fraction=self.config['federated']['client_fraction'],
            dp_epsilon=self.config['federated']['dp_epsilon']
        )
        
        # Plot federated learning progress
        fl_coordinator.plot_training_progress(self.config['paths']['figures_dir'])
        
        # Save federated model
        fl_coordinator.save_global_model(
            os.path.join(self.config['paths']['models_dir'], 'federated_global_model.pth')
        )
        
        self.federated_model = global_model
        self.results['federated'] = {
            'final_train_loss': fl_metrics['round_losses'][-1],
            'final_test_loss': fl_metrics['test_losses'][-1],
            'n_rounds': self.config['federated']['n_rounds'],
            'n_clients': len(clients)
        }
        
        # Compare with centralized
        centralized_model = copy.deepcopy(self.lstm_model)
        comparison = compare_federated_vs_centralized(
            global_model, centralized_model, test_loader,
            save_path=self.config['paths']['figures_dir']
        )
        
        self.results['federated']['comparison'] = comparison
        
        print("\nFederated Learning Summary:")
        print(f"  Final test loss: {self.results['federated']['final_test_loss']:.4f}")
        print(f"  Federated MAE: {comparison['federated']['mae']:.4f}")
        print(f"  Centralized MAE: {comparison['centralized']['mae']:.4f}")
    
    def prepare_edge_deployment(self):
        """Prepare models for edge deployment."""
        # Prepare example inputs
        seq_length = self.config['data']['sequence_length']
        n_features = self.dl_data['X_train'].shape[2]
        
        example_input_dl = torch.randn(1, seq_length, n_features)
        example_input_rl = torch.randn(1, 9)  # State dimension for RL
        
        # Package models for deployment
        models_dict = {
            'lstm_predictor': self.lstm_model,
            'transformer_predictor': self.transformer_model,
            'federated_predictor': self.federated_model
        }
        
        example_inputs_dict = {
            'lstm_predictor': example_input_dl,
            'transformer_predictor': example_input_dl,
            'federated_predictor': example_input_dl
        }
        
        # Create deployment package
        deployment_info = create_edge_deployment_package(
            models_dict,
            example_inputs_dict,
            output_dir=os.path.join(self.config['paths']['models_dir'], 'edge_deployment')
        )
        
        # Generate deployment report
        generate_edge_deployment_report(
            deployment_info,
            save_path=self.config['paths']['figures_dir']
        )
        
        self.results['edge_deployment'] = deployment_info
        
        print("\nEdge Deployment Summary:")
        for model_name, info in deployment_info.items():
            print(f"  {model_name}: {info['torchscript_latency_ms']:.2f} ms, {info['model_size_mb']:.2f} MB")
    
    def compile_results(self):
        """Compile and save all results."""
        # Save results to JSON
        results_file = os.path.join(
            self.config['paths']['results_dir'],
            f'experiment_results_{self.results["timestamp"]}.json'
        )
        
        # Convert numpy types to native Python types
        results_serializable = self._make_serializable(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Create summary figure
        self._create_summary_figure()
    
    def _make_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _create_summary_figure(self):
        """Create comprehensive summary figure."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # DL Model Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        models = ['LSTM', 'Transformer']
        maes = [
            self.results['dl_models']['lstm']['mae'],
            self.results['dl_models']['transformer']['mae']
        ]
        r2s = [
            self.results['dl_models']['lstm']['r2'],
            self.results['dl_models']['transformer']['r2']
        ]
        
        x = np.arange(len(models))
        width = 0.35
        ax1.bar(x - width/2, maes, width, label='MAE (Wh)', color='#2E86AB', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.bar(x + width/2, r2s, width, label='R²', color='#F18F01', alpha=0.7)
        
        ax1.set_ylabel('MAE (Wh)', fontweight='bold')
        ax1_twin.set_ylabel('R² Score', fontweight='bold')
        ax1.set_title('Deep Learning Model Comparison', fontweight='bold', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # RL Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        rl_models = ['PPO', 'Multi-Agent']
        rl_energies = [
            self.results['rl_agents']['ppo']['final_energy'],
            self.results['rl_agents']['multiagent']['final_energy']
        ]
        ax2.bar(rl_models, rl_energies, color='#A23B72', alpha=0.7)
        ax2.set_ylabel('Energy (kWh)', fontweight='bold')
        ax2.set_title('RL Agent Comparison', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Federated Learning
        ax3 = fig.add_subplot(gs[1, :])
        ax3.text(0.5, 0.5, 
                f"Federated Learning\n\n"
                f"Clients: {self.results['federated']['n_clients']}\n"
                f"Rounds: {self.results['federated']['n_rounds']}\n"
                f"Final Test Loss: {self.results['federated']['final_test_loss']:.4f}\n"
                f"Federated MAE: {self.results['federated']['comparison']['federated']['mae']:.4f}\n"
                f"Centralized MAE: {self.results['federated']['comparison']['centralized']['mae']:.4f}",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.axis('off')
        
        # Edge Deployment Summary
        ax4 = fig.add_subplot(gs[2, :])
        summary_text = "Edge Deployment Summary\n" + "="*60 + "\n\n"
        for model_name, info in self.results['edge_deployment'].items():
            summary_text += f"{model_name}:\n"
            summary_text += f"  Latency: {info['torchscript_latency_ms']:.2f} ms"
            summary_text += f"  | Size: {info['model_size_mb']:.2f} MB"
            summary_text += f"  | Speedup: {info['speedup']:.2f}x\n"
        
        ax4.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax4.axis('off')
        
        plt.suptitle('Edge AI Building Energy Optimization - Comprehensive Results',
                    fontsize=16, fontweight='bold')
        
        filename = os.path.join(self.config['paths']['figures_dir'], 'comprehensive_summary.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Summary figure saved: {filename}")
        plt.close()


if __name__ == "__main__":
    print("="*80)
    print("Edge AI Building Energy Optimization")
    print("Main Training Pipeline")
    print("="*80)
    
    # Create pipeline
    pipeline = EdgeAIExperimentPipeline()
    
    # Run complete pipeline
    results = pipeline.run_full_pipeline()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {pipeline.config['paths']['results_dir']}")
    print(f"Figures saved in: {pipeline.config['paths']['figures_dir']}")
    print(f"Models saved in: {pipeline.config['paths']['models_dir']}")
