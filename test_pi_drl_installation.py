"""
Installation Test Script for PI-DRL Framework

Run this after installing dependencies to verify everything works.

Usage:
    python3 test_pi_drl_installation.py
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("TESTING MODULE IMPORTS")
    print("="*60 + "\n")
    
    try:
        print("Testing environment module...", end=" ")
        from pi_drl_environment import SmartHomeEnv, BaselineThermostat, load_ampds2_mock_data
        print("✓")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    try:
        print("Testing training module...", end=" ")
        from pi_drl_training import PI_DRL_Trainer
        print("✓")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    try:
        print("Testing visualizer module...", end=" ")
        from publication_visualizer import ResultVisualizer
        print("✓")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    try:
        print("Testing table generator module...", end=" ")
        from publication_tables import PublicationTableGenerator
        print("✓")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n✓ All modules imported successfully!\n")
    return True


def test_environment():
    """Test environment creation and basic functionality"""
    print("="*60)
    print("TESTING ENVIRONMENT")
    print("="*60 + "\n")
    
    try:
        from pi_drl_environment import SmartHomeEnv, load_ampds2_mock_data
        
        print("Creating mock data...", end=" ")
        data = load_ampds2_mock_data(num_samples=1440)  # 24 hours
        print(f"✓ ({len(data)} samples)")
        
        print("Initializing environment...", end=" ")
        env = SmartHomeEnv(data=data)
        print("✓")
        
        print("Testing reset...", end=" ")
        obs, info = env.reset()
        print(f"✓ (state shape: {obs.shape})")
        
        print("Testing step...", end=" ")
        obs, reward, done, truncated, info = env.step(1)
        print(f"✓ (reward: {reward:.2f})")
        
        print("Testing episode...", end=" ")
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        print("✓")
        
        print("\n✓ Environment tests passed!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Environment test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_init():
    """Test trainer initialization (without full training)"""
    print("="*60)
    print("TESTING TRAINER INITIALIZATION")
    print("="*60 + "\n")
    
    try:
        from pi_drl_training import PI_DRL_Trainer
        
        print("Initializing trainer...", end=" ")
        trainer = PI_DRL_Trainer(output_dir="./test_output")
        print("✓")
        
        print("Creating environment...", end=" ")
        env = trainer.create_env()
        print("✓")
        
        print("Initializing PPO agent...", end=" ")
        trainer.initialize_agent()
        print("✓")
        
        print("\n✓ Trainer initialization tests passed!\n")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_output"):
            shutil.rmtree("./test_output")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Trainer test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_visualizer():
    """Test visualizer initialization"""
    print("="*60)
    print("TESTING VISUALIZER")
    print("="*60 + "\n")
    
    try:
        from publication_visualizer import ResultVisualizer
        
        print("Initializing visualizer...", end=" ")
        viz = ResultVisualizer(output_dir="./test_figures")
        print("✓")
        
        print("\n✓ Visualizer initialization tests passed!\n")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_figures"):
            shutil.rmtree("./test_figures")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Visualizer test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_table_generator():
    """Test table generator initialization"""
    print("="*60)
    print("TESTING TABLE GENERATOR")
    print("="*60 + "\n")
    
    try:
        from publication_tables import PublicationTableGenerator
        
        print("Initializing table generator...", end=" ")
        table_gen = PublicationTableGenerator(output_dir="./test_tables")
        print("✓")
        
        print("\n✓ Table generator initialization tests passed!\n")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_tables"):
            shutil.rmtree("./test_tables")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Table generator test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check that all required packages are installed"""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60 + "\n")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('gymnasium', 'gymnasium'),
        ('stable_baselines3', 'stable-baselines3'),
        ('torch', 'torch'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]
    
    missing = []
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (install with: pip install {pip_name})")
            missing.append(pip_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies installed!\n")
        return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PI-DRL FRAMEWORK INSTALLATION TEST")
    print("="*60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n✗ Please install missing dependencies:")
        print("   pip install -r requirements.txt\n")
        return
    
    # Run tests
    results = []
    results.append(("Module Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("Trainer", test_trainer_init()))
    results.append(("Visualizer", test_visualizer()))
    results.append(("Table Generator", test_table_generator()))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready to run the PI-DRL framework!")
        print("\nNext steps:")
        print("  1. Quick validation (5 min):")
        print("     python3 src/main_pi_drl.py --mode full --timesteps 10000")
        print("\n  2. Full training (3-5 hours):")
        print("     python3 src/main_pi_drl.py --mode full --timesteps 500000")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease review the errors above.")
    
    print()


if __name__ == "__main__":
    main()
