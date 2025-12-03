"""
PPO Training Script for Physics-Informed Deep Reinforcement Learning
Uses stable-baselines3 with callbacks for model checkpointing
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

from pi_drl_environment import SmartHomeEnv


class BestModelCallback(BaseCallback):
    """
    Callback to save the best model based on evaluation reward.
    """
    def __init__(self, eval_env, best_model_save_path: str, 
                 log_path: Optional[str] = None, verbose: int = 1):
        super(BestModelCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        
    def _on_step(self) -> bool:
        # Evaluate every 10000 steps
        if self.n_calls % 10000 == 0:
            # Evaluate the agent
            obs = self.eval_env.reset()
            episode_rewards = []
            episode_lengths = []
            
            for _ in range(10):  # Run 10 evaluation episodes
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        obs = self.eval_env.reset()
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
            
            mean_reward = np.mean(episode_rewards)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f} - Saving model")
                self.model.save(self.best_model_save_path)
        
        return True


def train_ppo_agent(
    env: SmartHomeEnv,
    total_timesteps: int = 500000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    policy_kwargs: Optional[Dict] = None,
    save_dir: str = "./models/pi_drl",
    model_name: str = "ppo_pi_drl",
    verbose: int = 1,
    seed: Optional[int] = None
) -> PPO:
    """
    Train a PPO agent on the SmartHomeEnv environment.
    
    Args:
        env: SmartHomeEnv instance
        total_timesteps: Total number of training timesteps
        learning_rate: Learning rate
        n_steps: Number of steps per update
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        policy_kwargs: Additional policy network arguments
        save_dir: Directory to save models
        model_name: Base name for saved models
        verbose: Verbosity level
        seed: Random seed
        
    Returns:
        Trained PPO model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Wrap environment
    env = Monitor(env, filename=os.path.join(save_dir, "monitor.csv"))
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = SmartHomeEnv(
        max_steps=env.envs[0].max_steps,
        comfort_setpoint=env.envs[0].comfort_setpoint,
        min_cycle_time=env.envs[0].min_cycle_time,
        R=env.envs[0].R,
        C=env.envs[0].C,
        hvac_power=env.envs[0].hvac_power,
        w1=env.envs[0].w1,
        w2=env.envs[0].w2,
        w3=env.envs[0].w3
    )
    eval_env = Monitor(eval_env, filename=os.path.join(save_dir, "eval_monitor.csv"))
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Default policy network (MLP with 2 hidden layers)
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        tensorboard_log=os.path.join(save_dir, "tensorboard")
    )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix=model_name
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    best_model_callback = BestModelCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(save_dir, f"{model_name}_best"),
        verbose=verbose
    )
    
    # Train the model
    print("="*80)
    print("Training PPO Agent for Physics-Informed DRL")
    print("="*80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor (γ): {gamma}")
    print(f"Min cycle time: {env.envs[0].min_cycle_time} minutes")
    print("="*80)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, best_model_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{model_name}_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    return model


def evaluate_agent(
    model: PPO,
    env: SmartHomeEnv,
    n_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained PPO agent.
    
    Args:
        model: Trained PPO model
        env: Environment instance
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_costs = []
    episode_discomforts = []
    episode_cycles = []
    episode_lengths = []
    
    all_actions = []
    all_indoor_temps = []
    all_outdoor_temps = []
    all_prices = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        actions_ep = []
        temps_ep = []
        outdoor_temps_ep = []
        prices_ep = []
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            actions_ep.append(int(action))
            temps_ep.append(info['indoor_temp'])
            outdoor_temps_ep.append(info['outdoor_temp'])
            prices_ep.append(info.get('cost', 0))
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_costs.append(info.get('total_cost', 0))
        episode_discomforts.append(info.get('total_discomfort', 0))
        episode_cycles.append(info.get('total_cycles', 0))
        episode_lengths.append(episode_length)
        
        all_actions.extend(actions_ep)
        all_indoor_temps.extend(temps_ep)
        all_outdoor_temps.extend(outdoor_temps_ep)
        all_prices.extend(prices_ep)
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'episode_discomforts': episode_discomforts,
        'episode_cycles': episode_cycles,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_discomfort': np.mean(episode_discomforts),
        'std_discomfort': np.std(episode_discomforts),
        'mean_cycles': np.mean(episode_cycles),
        'std_cycles': np.std(episode_cycles),
        'all_actions': all_actions,
        'all_indoor_temps': all_indoor_temps,
        'all_outdoor_temps': all_outdoor_temps,
        'all_prices': all_prices
    }
    
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Cost: ${results['mean_cost']:.2f} ± ${results['std_cost']:.2f}")
    print(f"Mean Discomfort: {results['mean_discomfort']:.2f} ± {results['std_discomfort']:.2f} degree-hours")
    print(f"Mean Cycles: {results['mean_cycles']:.1f} ± {results['std_cycles']:.1f}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Example usage
    env = SmartHomeEnv(
        max_steps=1440,  # 24 hours
        min_cycle_time=15,  # 15 minutes minimum
        seed=42
    )
    
    model = train_ppo_agent(
        env=env,
        total_timesteps=200000,
        save_dir="./models/pi_drl",
        model_name="ppo_pi_drl",
        verbose=1
    )
    
    # Evaluate
    eval_results = evaluate_agent(model, env, n_episodes=10)
