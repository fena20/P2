"""
PPO Agent Training for Physics-Informed Smart Home Control
Applied Energy Journal Submission

This module implements:
1. PPO agent initialization with optimized hyperparameters
2. Custom callbacks for model checkpointing
3. Training loop with evaluation
4. Baseline comparison generation
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from pi_drl_environment import SmartHomeEnv, BaselineThermostat, load_ampds2_mock_data


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to track and log training metrics
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.metrics_history = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones'):
            for i, done in enumerate(self.locals['dones']):
                if done:
                    # Get episode info
                    info = self.locals['infos'][i]
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        # Log every 10 episodes
                        if len(self.episode_rewards) % 10 == 0:
                            mean_reward = np.mean(self.episode_rewards[-10:])
                            if self.verbose > 0:
                                print(f"Episode {len(self.episode_rewards)}: "
                                      f"Mean Reward = {mean_reward:.2f}")
        
        return True
    
    def get_metrics(self) -> Dict:
        """Return training metrics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }


class PI_DRL_Trainer:
    """
    Main trainer class for Physics-Informed DRL
    """
    
    def __init__(
        self,
        env_params: Optional[Dict] = None,
        ppo_params: Optional[Dict] = None,
        output_dir: str = "./models",
        device: str = "auto"
    ):
        """
        Initialize trainer
        
        Args:
            env_params: Environment configuration
            ppo_params: PPO hyperparameters
            output_dir: Directory for saving models
            device: 'cpu', 'cuda', or 'auto'
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default environment parameters
        self.env_params = env_params or {
            'R': 10.0,
            'C': 20.0,
            'hvac_power': 3.0,
            'dt': 1/60,
            'comfort_range': (20.0, 24.0),
            'w_cost': 1.0,
            'w_comfort': 10.0,
            'w_cycling': 5.0,
            'min_cycle_time': 15,
            'episode_length': 1440
        }
        
        # Default PPO parameters (tuned for energy management)
        self.ppo_params = ppo_params or {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=torch.nn.ReLU
            ),
            'device': device,
            'verbose': 1
        }
        
        # Load data
        print("Loading AMPds2 data...")
        self.data = load_ampds2_mock_data()
        
        # Create environments
        self.train_env = None
        self.eval_env = None
        self.model = None
        
    def create_env(self, monitor_file: Optional[str] = None):
        """Create training environment"""
        env = SmartHomeEnv(data=self.data, **self.env_params)
        
        if monitor_file:
            env = Monitor(env, monitor_file)
        
        return env
    
    def initialize_agent(self):
        """Initialize PPO agent"""
        print("Initializing PPO agent...")
        
        # Create vectorized environment
        self.train_env = DummyVecEnv([lambda: self.create_env(
            os.path.join(self.output_dir, "monitor_train.csv")
        )])
        
        self.eval_env = DummyVecEnv([lambda: self.create_env(
            os.path.join(self.output_dir, "monitor_eval.csv")
        )])
        
        # Initialize PPO
        self.model = PPO(
            "MlpPolicy",
            self.train_env,
            **self.ppo_params
        )
        
        print(f"Agent initialized with {self.model.policy}")
        
    def train(
        self,
        total_timesteps: int = 500000,
        checkpoint_freq: int = 50000,
        eval_freq: int = 10000
    ):
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Total training steps
            checkpoint_freq: Frequency for saving checkpoints
            eval_freq: Frequency for evaluation
        """
        if self.model is None:
            self.initialize_agent()
        
        print(f"\nStarting training for {total_timesteps} timesteps...")
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(self.output_dir, "checkpoints"),
            name_prefix="pi_drl_model"
        )
        
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.output_dir,
            log_path=self.output_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        metrics_callback = TrainingMetricsCallback(verbose=1)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, metrics_callback],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "pi_drl_final_model")
        self.model.save(final_model_path)
        print(f"\nTraining complete! Model saved to {final_model_path}")
        
        return metrics_callback.get_metrics()
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)
        print("Model loaded successfully!")
    
    def evaluate_agent(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate trained agent
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        print(f"\nEvaluating agent over {n_episodes} episodes...")
        
        # Create fresh evaluation environment
        eval_env = self.create_env()
        
        episode_rewards = []
        episode_costs = []
        episode_discomforts = []
        episode_switches = []
        all_histories = []
        
        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            truncated = False
            ep_reward = 0
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = eval_env.step(action)
                ep_reward += reward
            
            # Get episode history
            history = eval_env.get_episode_history()
            all_histories.append(history)
            
            # Calculate episode metrics
            episode_rewards.append(ep_reward)
            episode_costs.append(history['cost'].sum())
            episode_discomforts.append(history['discomfort'].sum())
            episode_switches.append(history['action'].diff().abs().sum())
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_cost': np.mean(episode_costs),
            'mean_discomfort': np.mean(episode_discomforts),
            'mean_switches': np.mean(episode_switches),
            'histories': all_histories
        }
        
        print(f"Evaluation Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Cost: ${results['mean_cost']:.2f}")
        print(f"  Mean Discomfort: {results['mean_discomfort']:.2f} degree-hours")
        print(f"  Mean Switches: {results['mean_switches']:.0f}")
        
        return results
    
    def evaluate_baseline(self, n_episodes: int = 10) -> Dict:
        """
        Evaluate baseline thermostat controller
        
        Returns:
            Dictionary with baseline metrics
        """
        print(f"\nEvaluating baseline thermostat over {n_episodes} episodes...")
        
        # Create environment
        eval_env = self.create_env()
        baseline_controller = BaselineThermostat(T_setpoint=22.0, deadband=0.5)
        
        episode_rewards = []
        episode_costs = []
        episode_discomforts = []
        episode_switches = []
        all_histories = []
        
        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            truncated = False
            ep_reward = 0
            
            while not (done or truncated):
                # Baseline uses only indoor temperature
                action = baseline_controller.get_action(info['indoor_temp'])
                obs, reward, done, truncated, info = eval_env.step(action)
                ep_reward += reward
            
            # Get episode history
            history = eval_env.get_episode_history()
            all_histories.append(history)
            
            # Calculate episode metrics
            episode_rewards.append(ep_reward)
            episode_costs.append(history['cost'].sum())
            episode_discomforts.append(history['discomfort'].sum())
            episode_switches.append(history['action'].diff().abs().sum())
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_cost': np.mean(episode_costs),
            'mean_discomfort': np.mean(episode_discomforts),
            'mean_switches': np.mean(episode_switches),
            'histories': all_histories
        }
        
        print(f"Baseline Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Cost: ${results['mean_cost']:.2f}")
        print(f"  Mean Discomfort: {results['mean_discomfort']:.2f} degree-hours")
        print(f"  Mean Switches: {results['mean_switches']:.0f}")
        
        return results
    
    def run_ablation_study(self, n_episodes: int = 10) -> Dict:
        """
        Ablation study: Train agent WITHOUT cycling penalty
        
        Returns:
            Dictionary comparing different configurations
        """
        print("\n" + "="*60)
        print("ABLATION STUDY: Impact of Cycling Penalty")
        print("="*60)
        
        results = {}
        
        # 1. Full model (with cycling penalty)
        print("\n1. Training FULL MODEL (with cycling penalty)...")
        results['full_model'] = self.evaluate_agent(n_episodes=n_episodes)
        
        # 2. No cycling penalty
        print("\n2. Training WITHOUT cycling penalty...")
        no_cycle_params = self.env_params.copy()
        no_cycle_params['w_cycling'] = 0.0  # Remove cycling penalty
        
        # Create new trainer
        ablation_trainer = PI_DRL_Trainer(
            env_params=no_cycle_params,
            ppo_params=self.ppo_params,
            output_dir=os.path.join(self.output_dir, "ablation"),
            device=self.ppo_params['device']
        )
        
        ablation_trainer.initialize_agent()
        ablation_trainer.train(total_timesteps=100000, checkpoint_freq=50000, eval_freq=10000)
        results['no_cycling'] = ablation_trainer.evaluate_agent(n_episodes=n_episodes)
        
        # 3. Baseline
        print("\n3. Evaluating BASELINE...")
        results['baseline'] = self.evaluate_baseline(n_episodes=n_episodes)
        
        # Print comparison
        print("\n" + "="*60)
        print("ABLATION STUDY RESULTS")
        print("="*60)
        print(f"{'Method':<20} {'Cost ($)':<12} {'Discomfort':<15} {'Switches':<12}")
        print("-"*60)
        
        for method, res in results.items():
            print(f"{method:<20} {res['mean_cost']:<12.2f} "
                  f"{res['mean_discomfort']:<15.2f} {res['mean_switches']:<12.0f}")
        
        return results


def run_complete_training_pipeline(
    output_dir: str = "./models",
    total_timesteps: int = 500000,
    n_eval_episodes: int = 20
):
    """
    Complete training pipeline for publication
    
    Args:
        output_dir: Output directory for models
        total_timesteps: Training timesteps
        n_eval_episodes: Number of evaluation episodes
    """
    # Initialize trainer
    trainer = PI_DRL_Trainer(output_dir=output_dir)
    
    # Train
    print("\n" + "="*80)
    print("PHYSICS-INFORMED DEEP REINFORCEMENT LEARNING FOR SMART HOME ENERGY MANAGEMENT")
    print("="*80)
    
    trainer.initialize_agent()
    training_metrics = trainer.train(total_timesteps=total_timesteps)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    agent_results = trainer.evaluate_agent(n_episodes=n_eval_episodes)
    baseline_results = trainer.evaluate_baseline(n_episodes=n_eval_episodes)
    
    # Ablation study
    ablation_results = trainer.run_ablation_study(n_episodes=n_eval_episodes)
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.pkl")
    import pickle
    with open(results_path, 'wb') as f:
        pickle.dump({
            'training_metrics': training_metrics,
            'agent_results': agent_results,
            'baseline_results': baseline_results,
            'ablation_results': ablation_results
        }, f)
    
    print(f"\nResults saved to {results_path}")
    
    return {
        'agent_results': agent_results,
        'baseline_results': baseline_results,
        'ablation_results': ablation_results
    }


if __name__ == "__main__":
    # Run complete pipeline
    results = run_complete_training_pipeline(
        output_dir="./models_pi_drl",
        total_timesteps=500000,
        n_eval_episodes=20
    )
