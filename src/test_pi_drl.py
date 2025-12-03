"""
Quick Test Script for PI-DRL Framework
Verifies that all components work correctly
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_environment():
    """Test the SmartHomeEnv environment."""
    print("="*80)
    print("Testing SmartHomeEnv Environment")
    print("="*80)
    
    from pi_drl_environment import SmartHomeEnv
    
    env = SmartHomeEnv(max_steps=100, seed=42)
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation: {obs}")
    
    # Test step
    action = 1  # ON
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful")
    print(f"  Reward: {reward:.4f}")
    print(f"  Indoor Temp: {info['indoor_temp']:.2f}°C")
    print(f"  Cost: ${info['cost']:.4f}")
    
    # Test episode
    obs, _ = env.reset()
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"✓ Episode simulation successful")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps: {i+1}")
    print("="*80)
    return True


def test_baseline_controller():
    """Test the baseline controller."""
    print("\n" + "="*80)
    print("Testing Baseline Controller")
    print("="*80)
    
    from pi_drl_environment import SmartHomeEnv
    from pi_drl_main import BaselineController
    
    env = SmartHomeEnv(max_steps=100, seed=42)
    controller = BaselineController(comfort_setpoint=22.0, deadband=1.0)
    
    obs, _ = env.reset()
    actions = []
    for i in range(20):
        action = controller.predict(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"✓ Baseline controller test successful")
    print(f"  Actions taken: {actions[:10]}...")
    print(f"  Action changes: {sum([1 for i in range(1, len(actions)) if actions[i] != actions[i-1]])}")
    print("="*80)
    return True


def test_visualization():
    """Test visualization module (without actual model)."""
    print("\n" + "="*80)
    print("Testing Visualization Module")
    print("="*80)
    
    from pi_drl_visualization import ResultVisualizer
    
    visualizer = ResultVisualizer(save_dir="./test_figures")
    
    # Create synthetic data
    n_samples = 200
    pi_drl_actions = np.random.randint(0, 2, n_samples)
    pi_drl_temps = 22.0 + np.random.normal(0, 1, n_samples)
    baseline_actions = np.random.randint(0, 2, n_samples)
    baseline_temps = 22.0 + np.random.normal(0, 1.5, n_samples)
    time_hours = np.arange(n_samples) / 60.0
    
    # Test Figure 1
    try:
        visualizer.figure1_system_heartbeat(
            pi_drl_actions, pi_drl_temps,
            baseline_actions, baseline_temps,
            time_hours
        )
        print("✓ Figure 1 (System Heartbeat) generated successfully")
    except Exception as e:
        print(f"✗ Figure 1 failed: {e}")
        return False
    
    # Test Figure 3 (Radar Chart)
    try:
        baseline_metrics = {
            'energy_cost': 100,
            'comfort_violation': 100,
            'equipment_cycles': 100,
            'peak_load': 100,
            'carbon': 100
        }
        pi_drl_metrics = {
            'energy_cost': 80,
            'comfort_violation': 75,
            'equipment_cycles': 60,
            'peak_load': 70,
            'carbon': 85
        }
        visualizer.figure3_multi_objective_radar(baseline_metrics, pi_drl_metrics)
        print("✓ Figure 3 (Radar Chart) generated successfully")
    except Exception as e:
        print(f"✗ Figure 3 failed: {e}")
        return False
    
    print("="*80)
    return True


def test_table_generation():
    """Test table generation module."""
    print("\n" + "="*80)
    print("Testing Table Generation Module")
    print("="*80)
    
    from pi_drl_tables import TableGenerator
    
    generator = TableGenerator(save_dir="./test_tables")
    
    # Test Table 1
    try:
        df1 = generator.table1_simulation_hyperparameters()
        print("✓ Table 1 (Hyperparameters) generated successfully")
        print(f"  Shape: {df1.shape}")
    except Exception as e:
        print(f"✗ Table 1 failed: {e}")
        return False
    
    # Test Table 2
    try:
        df2 = generator.table2_performance_comparison(
            baseline_cost=120.50,
            pi_drl_cost=85.30,
            baseline_discomfort=45.2,
            pi_drl_discomfort=28.5,
            baseline_cycles=150,
            pi_drl_cycles=60
        )
        print("✓ Table 2 (Performance Comparison) generated successfully")
        print(f"  Shape: {df2.shape}")
    except Exception as e:
        print(f"✗ Table 2 failed: {e}")
        return False
    
    # Test Table 3
    try:
        df3 = generator.table3_ablation_study(
            pi_drl_with_cycling={'cost': 85.30, 'discomfort': 28.5, 'cycles': 60},
            pi_drl_without_cycling={'cost': 75.20, 'discomfort': 25.0, 'cycles': 450},
            baseline={'cost': 120.50, 'discomfort': 45.2, 'cycles': 150}
        )
        print("✓ Table 3 (Ablation Study) generated successfully")
        print(f"  Shape: {df3.shape}")
    except Exception as e:
        print(f"✗ Table 3 failed: {e}")
        return False
    
    print("="*80)
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PI-DRL Framework - Component Tests")
    print("="*80)
    
    tests = [
        ("Environment", test_environment),
        ("Baseline Controller", test_baseline_controller),
        ("Visualization", test_visualization),
        ("Table Generation", test_table_generation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("="*80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    main()
