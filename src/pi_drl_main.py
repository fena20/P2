"""
Main Execution Script for Physics-Informed Deep Reinforcement Learning Framework
Orchestrates: Environment, Training, Evaluation, Visualization, and Table Generation
"""

import os
import numpy as np
from stable_baselines3 import PPO
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from pi_drl_environment import SmartHomeEnv
from pi_drl_training import train_ppo_agent, evaluate_agent
from pi_drl_visualization import ResultVisualizer
from pi_drl_tables import TableGenerator


class BaselineController:
    """
    Simple baseline thermostat controller for comparison.
    Implements a rule-based controller with frequent switching (no cycling protection).
    """
    
    def __init__(self, comfort_setpoint: float = 22.0, deadband: float = 1.0):
        """
        Initialize baseline controller.
        
        Args:
            comfort_setpoint: Desired temperature (°C)
            deadband: Temperature deadband (°C)
        """
        self.comfort_setpoint = comfort_setpoint
        self.deadband = deadband
        self.current_action = 0
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict action based on observation.
        
        Args:
            observation: State vector [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
            
        Returns:
            Action: 0 (OFF) or 1 (ON)
        """
        # Extract indoor temperature (denormalize)
        indoor_temp = observation[0] * 30.0  # Denormalize from [0,1] to [15,30]
        
        # Simple rule: turn ON if below setpoint - deadband, OFF if above setpoint + deadband
        if indoor_temp < self.comfort_setpoint - self.deadband:
            self.current_action = 1  # ON
        elif indoor_temp > self.comfort_setpoint + self.deadband:
            self.current_action = 0  # OFF
        # Else keep current action (hysteresis)
        
        return self.current_action


def run_baseline_controller(
    env: SmartHomeEnv,
    n_episodes: int = 10
) -> Dict[str, Any]:
    """
    Run baseline thermostat controller for comparison.
    
    Args:
        env: Environment instance
        n_episodes: Number of episodes to run
        
    Returns:
        Dictionary with results
    """
    controller = BaselineController(
        comfort_setpoint=env.comfort_setpoint,
        deadband=1.0
    )
    
    episode_rewards = []
    episode_costs = []
    episode_discomforts = []
    episode_cycles = []
    
    all_actions = []
    all_indoor_temps = []
    all_outdoor_temps = []
    all_prices = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        last_action = 0
        cycles = 0
        
        actions_ep = []
        temps_ep = []
        outdoor_temps_ep = []
        prices_ep = []
        
        done = False
        while not done:
            action = controller.predict(obs)
            
            # Count cycles
            if action != last_action:
                cycles += 1
            last_action = action
            
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
        episode_cycles.append(cycles)
        
        all_actions.extend(actions_ep)
        all_indoor_temps.extend(temps_ep)
        all_outdoor_temps.extend(outdoor_temps_ep)
        all_prices.extend(prices_ep)
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'episode_discomforts': episode_discomforts,
        'episode_cycles': episode_cycles,
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
    print("Baseline Controller Results")
    print("="*80)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Cost: ${results['mean_cost']:.2f} ± ${results['std_cost']:.2f}")
    print(f"Mean Discomfort: {results['mean_discomfort']:.2f} ± {results['std_discomfort']:.2f} degree-hours")
    print(f"Mean Cycles: {results['mean_cycles']:.1f} ± {results['std_cycles']:.1f}")
    print("="*80)
    
    return results


def run_ablation_study(
    env_params: Dict[str, Any],
    training_params: Dict[str, Any],
    save_dir: str = "./models/pi_drl_ablation"
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study: Compare PI-DRL with and without cycling penalty.
    
    Args:
        env_params: Environment parameters
        training_params: Training parameters
        save_dir: Directory to save models
        
    Returns:
        Dictionary with ablation results
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: Cycling Penalty Impact")
    print("="*80)
    
    # 1. Train with cycling penalty (default)
    print("\n[1/2] Training PI-DRL WITH cycling penalty...")
    env_with = SmartHomeEnv(**env_params)
    model_with = train_ppo_agent(
        env=env_with,
        save_dir=os.path.join(save_dir, "with_cycling"),
        model_name="ppo_with_cycling",
        **training_params
    )
    results_with = evaluate_agent(model_with, env_with, n_episodes=5)
    
    # 2. Train without cycling penalty (w3 = 0)
    print("\n[2/2] Training PI-DRL WITHOUT cycling penalty...")
    env_params_no_cycling = env_params.copy()
    env_params_no_cycling['w3'] = 0.0  # Remove cycling penalty
    env_without = SmartHomeEnv(**env_params_no_cycling)
    model_without = train_ppo_agent(
        env=env_without,
        save_dir=os.path.join(save_dir, "without_cycling"),
        model_name="ppo_without_cycling",
        **training_params
    )
    results_without = evaluate_agent(model_without, env_without, n_episodes=5)
    
    ablation_results = {
        'with_cycling': {
            'cost': results_with['mean_cost'],
            'discomfort': results_with['mean_discomfort'],
            'cycles': results_with['mean_cycles']
        },
        'without_cycling': {
            'cost': results_without['mean_cost'],
            'discomfort': results_without['mean_discomfort'],
            'cycles': results_without['mean_cycles']
        }
    }
    
    print("\n" + "="*80)
    print("Ablation Study Results")
    print("="*80)
    print(f"WITH Cycling Penalty:")
    print(f"  Cost: ${ablation_results['with_cycling']['cost']:.2f}")
    print(f"  Cycles: {ablation_results['with_cycling']['cycles']:.0f}")
    print(f"\nWITHOUT Cycling Penalty:")
    print(f"  Cost: ${ablation_results['without_cycling']['cost']:.2f}")
    print(f"  Cycles: {ablation_results['without_cycling']['cycles']:.0f}")
    print(f"\n⚠️  Warning: Without cycling penalty, cycles increased by "
          f"{((ablation_results['without_cycling']['cycles'] - ablation_results['with_cycling']['cycles']) / ablation_results['with_cycling']['cycles'] * 100):.0f}%")
    print("="*80)
    
    return ablation_results


def main(
    data_path: Optional[str] = None,
    train_agent: bool = True,
    run_ablation: bool = True,
    total_timesteps: int = 200000,
    n_eval_episodes: int = 10,
    save_dir: str = "./output"
):
    """
    Main execution function.
    
    Args:
        data_path: Path to AMPds2 data CSV (None for synthetic)
        train_agent: Whether to train the PPO agent
        run_ablation: Whether to run ablation study
        total_timesteps: Total training timesteps
        n_eval_episodes: Number of evaluation episodes
        save_dir: Base directory for saving results
    """
    print("="*80)
    print("Physics-Informed Deep Reinforcement Learning Framework")
    print("For Residential Building Energy Management (Applied Energy Journal)")
    print("="*80)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "tables"), exist_ok=True)
    
    # Environment parameters
    env_params = {
        'data_path': data_path,
        'max_steps': 1440,  # 24 hours at 1-min resolution
        'comfort_setpoint': 22.0,
        'comfort_tolerance': 2.0,
        'min_cycle_time': 15,  # 15 minutes (critical!)
        'R': 0.05,
        'C': 0.5,
        'hvac_power': 3.0,
        'w1': 1.0,
        'w2': 10.0,
        'w3': 5.0,  # Cycling penalty weight
        'seed': 42
    }
    
    # Training parameters
    training_params = {
        'total_timesteps': total_timesteps,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'verbose': 1,
        'seed': 42
    }
    
    # Create environment
    env = SmartHomeEnv(**env_params)
    
    # Train PPO agent
    if train_agent:
        print("\n" + "="*80)
        print("PHASE 1: Training PPO Agent")
        print("="*80)
        model = train_ppo_agent(
            env=env,
            save_dir=os.path.join(save_dir, "models"),
            model_name="ppo_pi_drl",
            **training_params
        )
    else:
        # Load pre-trained model
        model_path = os.path.join(save_dir, "models", "ppo_pi_drl_best")
        print(f"\nLoading pre-trained model from: {model_path}")
        model = PPO.load(model_path)
    
    # Evaluate PI-DRL agent
    print("\n" + "="*80)
    print("PHASE 2: Evaluating PI-DRL Agent")
    print("="*80)
    pi_drl_results = evaluate_agent(model, env, n_episodes=n_eval_episodes)
    
    # Run baseline controller
    print("\n" + "="*80)
    print("PHASE 3: Evaluating Baseline Controller")
    print("="*80)
    baseline_results = run_baseline_controller(env, n_episodes=n_eval_episodes)
    
    # Run ablation study (optional)
    ablation_results = None
    if run_ablation:
        ablation_results = run_ablation_study(
            env_params=env_params,
            training_params=training_params,
            save_dir=os.path.join(save_dir, "models", "ablation")
        )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("PHASE 4: Generating Publication Figures")
    print("="*80)
    visualizer = ResultVisualizer(save_dir=os.path.join(save_dir, "figures"))
    visualizer.generate_all_figures(
        model=model,
        env=env,
        pi_drl_results=pi_drl_results,
        baseline_results=baseline_results
    )
    
    # Generate tables
    print("\n" + "="*80)
    print("PHASE 5: Generating Publication Tables")
    print("="*80)
    table_generator = TableGenerator(save_dir=os.path.join(save_dir, "tables"))
    
    # Prepare results for tables
    baseline_table_results = {
        'cost': baseline_results['mean_cost'],
        'discomfort': baseline_results['mean_discomfort'],
        'cycles': baseline_results['mean_cycles'],
        'peak_load': np.max([3.0 if a == 1 else 0.0 for a in baseline_results['all_actions']]) if len(baseline_results['all_actions']) > 0 else None
    }
    
    pi_drl_table_results = {
        'cost': pi_drl_results['mean_cost'],
        'discomfort': pi_drl_results['mean_discomfort'],
        'cycles': pi_drl_results['mean_cycles'],
        'peak_load': np.max([3.0 if a == 1 else 0.0 for a in pi_drl_results['all_actions']]) if len(pi_drl_results['all_actions']) > 0 else None
    }
    
    table_generator.generate_all_tables(
        env_params=env_params,
        training_params=training_params,
        baseline_results=baseline_table_results,
        pi_drl_results=pi_drl_table_results,
        ablation_results=ablation_results
    )
    
    print("\n" + "="*80)
    print("COMPLETE! All results saved to:", save_dir)
    print("="*80)
    print("\nGenerated outputs:")
    print(f"  - Models: {os.path.join(save_dir, 'models')}")
    print(f"  - Figures: {os.path.join(save_dir, 'figures')}")
    print(f"  - Tables: {os.path.join(save_dir, 'tables')}")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PI-DRL Framework for Building Energy Management')
    parser.add_argument('--data_path', type=str, default=None, help='Path to AMPds2 CSV data')
    parser.add_argument('--no_train', action='store_true', help='Skip training (load existing model)')
    parser.add_argument('--no_ablation', action='store_true', help='Skip ablation study')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total training timesteps')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default='./results/pi_drl', help='Save directory')
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        train_agent=not args.no_train,
        run_ablation=not args.no_ablation,
        total_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        save_dir=args.save_dir
    )
