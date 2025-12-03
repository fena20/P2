"""
Test Suite for Physics-Informed Deep Reinforcement Learning (PI-DRL) Framework

This script validates all components of the PI-DRL framework:
1. AMPds2 Data Loader
2. SmartHomeEnv (Physics-Informed Environment)
3. PPO Agent Training
4. ResultVisualizer (Publication-Quality Figures)
5. Golden Tables Generation

Target Journal: Applied Energy (Q1)
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings('ignore')


def test_data_loader():
    """Test AMPds2 data loader."""
    print("\n" + "=" * 70)
    print("TEST 1: AMPds2 Data Loader")
    print("=" * 70)
    
    from pi_drl_framework import AMPds2DataLoader
    
    # Create loader
    loader = AMPds2DataLoader(n_days=7, resolution_minutes=1)
    data = loader.load_data()
    
    # Validate structure
    expected_columns = ['timestamp', 'WHE', 'HPE', 'FRE', 
                       'Outdoor_Temp', 'Solar_Radiation', 'Electricity_Price']
    
    assert all(col in data.columns for col in expected_columns), \
        f"Missing columns. Expected: {expected_columns}, Got: {list(data.columns)}"
    
    # Validate data ranges
    assert data['Outdoor_Temp'].min() >= -15, "Outdoor temp too low"
    assert data['Outdoor_Temp'].max() <= 45, "Outdoor temp too high"
    assert data['Solar_Radiation'].min() >= 0, "Negative solar radiation"
    assert data['Electricity_Price'].min() > 0, "Price must be positive"
    
    # Validate data length
    expected_samples = 7 * 24 * 60  # 7 days * 24 hours * 60 minutes
    assert len(data) == expected_samples, \
        f"Expected {expected_samples} samples, got {len(data)}"
    
    print(f"✓ Data shape: {data.shape}")
    print(f"✓ Columns: {list(data.columns)}")
    print(f"✓ Temperature range: [{data['Outdoor_Temp'].min():.1f}, {data['Outdoor_Temp'].max():.1f}] °C")
    print(f"✓ Price range: [{data['Electricity_Price'].min():.3f}, {data['Electricity_Price'].max():.3f}] $/kWh")
    print("✓ Data Loader: PASSED")
    
    return data


def test_environment(data):
    """Test SmartHomeEnv physics-informed environment."""
    print("\n" + "=" * 70)
    print("TEST 2: SmartHomeEnv (Physics-Informed Environment)")
    print("=" * 70)
    
    from pi_drl_framework import SmartHomeEnv
    
    # Create environment
    env = SmartHomeEnv(
        data=data,
        R=2.0,
        C=10.0,
        Q_hvac=3.5,
        episode_length=100
    )
    
    # Test reset
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (6,), f"Expected obs shape (6,), got {obs.shape}"
    assert env.observation_space.contains(obs), "Observation outside bounds"
    
    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space: {env.action_space}")
    print(f"✓ Initial observation: {obs}")
    
    # Test step with both actions
    for action in [0, 1]:
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        assert env.observation_space.contains(obs_next), "Next obs outside bounds"
        assert isinstance(reward, (int, float)), "Reward must be numeric"
        assert 'cost' in info, "Info must contain 'cost'"
        assert 'discomfort' in info, "Info must contain 'discomfort'"
        assert 'cycling_penalty' in info, "Info must contain 'cycling_penalty'"
    
    print(f"✓ Step returns valid observation, reward, and info")
    
    # Test physics model (temperature should change)
    env.reset(seed=42)
    initial_temp = env.T_indoor
    
    for _ in range(10):
        env.step(1)  # HVAC ON
    
    temp_after_heating = env.T_indoor
    assert temp_after_heating != initial_temp, "Temperature should change with HVAC"
    
    print(f"✓ Physics model: Temperature changed from {initial_temp:.2f} to {temp_after_heating:.2f} °C")
    
    # Test cycling penalty
    env.reset(seed=42)
    env.step(0)  # OFF
    env.step(1)  # ON - switch
    _, _, _, _, info = env.step(0)  # OFF - rapid switch
    
    assert info['cycling_penalty'] > 0, "Cycling penalty should be triggered"
    print(f"✓ Cycling penalty triggered: {info['cycling_penalty']:.4f}")
    
    # Test episode completion
    env.reset(seed=42)
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    
    assert steps == env.episode_length, f"Episode should be {env.episode_length} steps"
    print(f"✓ Episode completed in {steps} steps")
    
    # Test hyperparameters export
    params = env.get_hyperparameters()
    assert len(params) > 0, "Hyperparameters should not be empty"
    print(f"✓ Hyperparameters exported: {len(params)} parameters")
    
    print("✓ SmartHomeEnv: PASSED")
    
    return env


def test_baseline_controller(env):
    """Test baseline thermostat controller."""
    print("\n" + "=" * 70)
    print("TEST 3: Baseline Thermostat Controller")
    print("=" * 70)
    
    from pi_drl_framework import BaselineThermostat, SmartHomeEnv
    
    thermostat = BaselineThermostat(setpoint=21.0, deadband=0.5)
    
    # Create fresh environment
    env = SmartHomeEnv(episode_length=100)
    obs, _ = env.reset(seed=42)
    
    # Run episode
    total_cost = 0
    total_switches = 0
    last_action = 0
    
    done = False
    while not done:
        T_indoor = obs[0]
        T_outdoor = obs[1]
        action = thermostat.get_action(T_indoor, T_outdoor)
        
        if action != last_action:
            total_switches += 1
        last_action = action
        
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_cost += info['cost']
    
    print(f"✓ Baseline completed with:")
    print(f"  - Total cost: ${total_cost:.4f}")
    print(f"  - Switching count: {total_switches}")
    print(f"  - Average switches per 100 steps: {total_switches}")
    
    print("✓ Baseline Controller: PASSED")


def test_ppo_agent(env):
    """Test PPO agent creation and training."""
    print("\n" + "=" * 70)
    print("TEST 4: PPO Agent (Stable-Baselines3)")
    print("=" * 70)
    
    from pi_drl_framework import create_ppo_agent, SmartHomeEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create environment
    env = SmartHomeEnv(episode_length=100)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    # Create agent
    agent = create_ppo_agent(
        env=vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        verbose=0
    )
    
    print(f"✓ PPO agent created")
    print(f"  - Policy: {agent.policy}")
    print(f"  - Learning rate: {agent.learning_rate}")
    
    # Short training test
    print("  - Training for 500 steps...")
    agent.learn(total_timesteps=500, progress_bar=False)
    
    # Test prediction
    test_env = SmartHomeEnv(episode_length=10)
    obs, _ = test_env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    
    assert action in [0, 1], f"Invalid action: {action}"
    print(f"✓ Agent prediction: action = {action}")
    
    # Test saving/loading
    save_path = './models/test_agent'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    
    from stable_baselines3 import PPO
    loaded_agent = PPO.load(save_path)
    loaded_action, _ = loaded_agent.predict(obs, deterministic=True)
    
    assert loaded_action == action, "Loaded agent should give same action"
    print(f"✓ Agent saved and loaded successfully")
    
    print("✓ PPO Agent: PASSED")
    
    return agent


def test_visualizer():
    """Test ResultVisualizer for publication-quality figures."""
    print("\n" + "=" * 70)
    print("TEST 5: ResultVisualizer (Publication Figures)")
    print("=" * 70)
    
    from pi_drl_framework import ResultVisualizer
    
    # Create visualizer
    save_dir = './test_figures'
    os.makedirs(save_dir, exist_ok=True)
    visualizer = ResultVisualizer(save_dir=save_dir)
    
    # Test Figure 1: System Heartbeat
    print("  - Testing Figure 1: System Heartbeat...")
    baseline_data = {
        'action': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 12,  # Frequent switching
        'indoor_temp': np.linspace(20, 22, 120).tolist()
    }
    pidrl_data = {
        'action': [1] * 40 + [0] * 40 + [1] * 40,  # Stable runs
        'indoor_temp': np.linspace(20, 22, 120).tolist()
    }
    
    path1 = visualizer.figure1_system_heartbeat(baseline_data, pidrl_data)
    assert os.path.exists(path1), f"Figure 1 not saved: {path1}"
    print(f"  ✓ Figure 1 saved: {path1}")
    
    # Test Figure 2: Policy Heatmap
    print("  - Testing Figure 2: Policy Heatmap...")
    policy_data = np.random.rand(40, 24)
    # Simulate demand response (low probability during peak hours)
    policy_data[:, 17:20] *= 0.3
    
    path2 = visualizer.figure2_control_policy_heatmap(policy_data)
    assert os.path.exists(path2), f"Figure 2 not saved: {path2}"
    print(f"  ✓ Figure 2 saved: {path2}")
    
    # Test Figure 3: Radar Chart
    print("  - Testing Figure 3: Radar Chart...")
    baseline_metrics = {
        'energy_cost': 100,
        'comfort_violation': 100,
        'equipment_cycles': 100,
        'peak_load': 100,
        'carbon_emissions': 100
    }
    pidrl_metrics = {
        'energy_cost': 75,
        'comfort_violation': 80,
        'equipment_cycles': 35,
        'peak_load': 70,
        'carbon_emissions': 75
    }
    
    path3 = visualizer.figure3_radar_chart(baseline_metrics, pidrl_metrics)
    assert os.path.exists(path3), f"Figure 3 not saved: {path3}"
    print(f"  ✓ Figure 3 saved: {path3}")
    
    # Test Figure 4: Energy Carpet
    print("  - Testing Figure 4: Energy Carpet Plot...")
    baseline_carpet = np.random.rand(30, 24) * 2
    optimized_carpet = baseline_carpet.copy()
    optimized_carpet[:, 17:20] *= 0.3  # Reduce peak hours
    
    path4 = visualizer.figure4_energy_carpet(baseline_carpet, optimized_carpet)
    assert os.path.exists(path4), f"Figure 4 not saved: {path4}"
    print(f"  ✓ Figure 4 saved: {path4}")
    
    print("✓ ResultVisualizer: PASSED")
    
    return visualizer


def test_tables(visualizer):
    """Test golden tables generation."""
    print("\n" + "=" * 70)
    print("TEST 6: Golden Tables Generation")
    print("=" * 70)
    
    from pi_drl_framework import get_ppo_hyperparameters
    
    # Test Table 1: Hyperparameters
    print("  - Testing Table 1: Hyperparameters...")
    env_params = {
        'R': '2.0 °C/kW',
        'C': '10.0 kWh/°C',
        'Q_hvac': '3.5 kW',
        'w1': '1.0',
        'w2': '2.0',
        'w3': '0.5'
    }
    ppo_params = get_ppo_hyperparameters()
    
    path1 = visualizer.table1_hyperparameters(env_params, ppo_params)
    assert os.path.exists(path1), f"Table 1 not saved: {path1}"
    print(f"  ✓ Table 1 saved: {path1}")
    
    # Test Table 2: Performance Comparison
    print("  - Testing Table 2: Performance Comparison...")
    baseline_results = {
        'total_cost': 15.5,
        'discomfort': 12.3,
        'switching_count': 48,
        'peak_load': 1.17
    }
    pidrl_results = {
        'total_cost': 10.2,
        'discomfort': 8.5,
        'switching_count': 12,
        'peak_load': 1.17
    }
    
    path2 = visualizer.table2_performance_comparison(baseline_results, pidrl_results)
    assert os.path.exists(path2), f"Table 2 not saved: {path2}"
    print(f"  ✓ Table 2 saved: {path2}")
    
    # Test Table 3: Ablation Study
    print("  - Testing Table 3: Ablation Study...")
    full_model = pidrl_results
    no_cycling = {
        'total_cost': 9.8,
        'discomfort': 8.2,
        'switching_count': 85  # High switching without penalty
    }
    no_discomfort = {
        'total_cost': 8.5,
        'discomfort': 25.0,  # High discomfort
        'switching_count': 10
    }
    
    path3 = visualizer.table3_ablation_study(full_model, no_cycling, no_discomfort)
    assert os.path.exists(path3), f"Table 3 not saved: {path3}"
    print(f"  ✓ Table 3 saved: {path3}")
    
    print("✓ Golden Tables: PASSED")


def test_evaluation_functions():
    """Test evaluation and helper functions."""
    print("\n" + "=" * 70)
    print("TEST 7: Evaluation Functions")
    print("=" * 70)
    
    from pi_drl_framework import (
        evaluate_agent, evaluate_baseline, 
        generate_policy_heatmap_data, generate_carpet_data,
        SmartHomeEnv, BaselineThermostat
    )
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create and train quick agent
    env = SmartHomeEnv(episode_length=50)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    agent = PPO('MlpPolicy', vec_env, verbose=0)
    agent.learn(total_timesteps=200, progress_bar=False)
    
    # Test evaluate_agent
    print("  - Testing evaluate_agent...")
    eval_env = SmartHomeEnv(episode_length=50)
    results = evaluate_agent(eval_env, agent, n_episodes=2)
    
    assert 'mean_reward' in results, "Results should contain mean_reward"
    assert 'total_cost' in results, "Results should contain total_cost"
    assert 'switching_count' in results, "Results should contain switching_count"
    print(f"  ✓ evaluate_agent: reward={results['mean_reward']:.2f}")
    
    # Test evaluate_baseline
    print("  - Testing evaluate_baseline...")
    thermostat = BaselineThermostat()
    baseline_env = SmartHomeEnv(episode_length=50)
    baseline_results = evaluate_baseline(baseline_env, thermostat, n_episodes=2)
    
    assert 'total_cost' in baseline_results
    print(f"  ✓ evaluate_baseline: cost=${baseline_results['total_cost']:.2f}")
    
    # Test generate_carpet_data
    print("  - Testing generate_carpet_data...")
    actions = [0, 1] * 500
    carpet = generate_carpet_data(actions, n_days=5)
    
    assert carpet.shape[0] == 5, f"Expected 5 days, got {carpet.shape[0]}"
    assert carpet.shape[1] == 24, f"Expected 24 hours, got {carpet.shape[1]}"
    print(f"  ✓ generate_carpet_data: shape={carpet.shape}")
    
    print("✓ Evaluation Functions: PASSED")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "PI-DRL FRAMEWORK TEST SUITE" + " " * 21 + "#")
    print("#" * 70)
    
    results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    tests = [
        ('Data Loader', test_data_loader),
        ('Environment', lambda: test_environment(test_data_loader())),
        ('Baseline Controller', lambda: test_baseline_controller(SmartHomeEnv(episode_length=100))),
        ('PPO Agent', lambda: test_ppo_agent(SmartHomeEnv(episode_length=100))),
        ('Visualizer', test_visualizer),
        ('Tables', lambda: test_tables(ResultVisualizer('./test_figures'))),
        ('Evaluation Functions', test_evaluation_functions),
    ]
    
    # Import here to avoid issues
    from pi_drl_framework import SmartHomeEnv, ResultVisualizer
    
    for name, test_func in tests:
        try:
            test_func()
            results['passed'] += 1
        except Exception as e:
            results['failed'] += 1
            results['errors'].append((name, str(e)))
            print(f"\n✗ {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ Passed: {results['passed']}")
    print(f"✗ Failed: {results['failed']}")
    
    if results['errors']:
        print("\nErrors:")
        for name, error in results['errors']:
            print(f"  - {name}: {error}")
    
    print("=" * 70)
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! Framework ready for publication.")
    else:
        print(f"\n⚠️  {results['failed']} test(s) failed. Please review errors above.")
    
    return results['failed'] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
