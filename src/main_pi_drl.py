"""
Main Execution Script for Physics-Informed Deep Reinforcement Learning
Smart Home Energy Management System

Applied Energy Journal Submission

This script orchestrates the complete pipeline:
1. Environment creation and validation
2. PPO agent training
3. Performance evaluation
4. Ablation study
5. Publication-quality visualization generation
6. Table generation for manuscript

Usage:
    python main_pi_drl.py --mode [train|evaluate|ablation|full]
    
    Modes:
        train     - Train new PI-DRL agent
        evaluate  - Evaluate trained agent and generate figures
        ablation  - Run ablation study
        full      - Complete pipeline (train + evaluate + ablation + visualize)
"""

import os
import sys
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pi_drl_environment import SmartHomeEnv, load_ampds2_mock_data
from pi_drl_training import PI_DRL_Trainer
from publication_visualizer import ResultVisualizer
from publication_tables import PublicationTableGenerator


def setup_directories(base_output_dir: str = "./outputs_pi_drl"):
    """Create output directory structure"""
    directories = {
        'base': base_output_dir,
        'models': os.path.join(base_output_dir, 'models'),
        'figures': os.path.join(base_output_dir, 'figures'),
        'tables': os.path.join(base_output_dir, 'tables'),
        'results': os.path.join(base_output_dir, 'results')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def mode_train(dirs: dict, args):
    """Training mode: Train PI-DRL agent"""
    print_header("MODE: TRAINING")
    
    # Initialize trainer
    trainer = PI_DRL_Trainer(
        output_dir=dirs['models'],
        device=args.device
    )
    
    # Train
    trainer.initialize_agent()
    training_metrics = trainer.train(
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq
    )
    
    # Save training metrics
    results_path = os.path.join(dirs['results'], 'training_metrics.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(training_metrics, f)
    
    print(f"\nTraining complete! Model saved to: {dirs['models']}")
    print(f"Training metrics saved to: {results_path}")
    
    return trainer


def mode_evaluate(dirs: dict, args):
    """Evaluation mode: Evaluate trained agent and generate outputs"""
    print_header("MODE: EVALUATION")
    
    # Load or train model
    trainer = PI_DRL_Trainer(output_dir=dirs['models'], device=args.device)
    
    model_path = os.path.join(dirs['models'], 'best_model.zip')
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        trainer.initialize_agent()
        trainer.model = trainer.model.load(model_path)
    else:
        print("No trained model found. Training new model...")
        trainer.initialize_agent()
        trainer.train(total_timesteps=args.timesteps)
    
    # Evaluate agent
    print("\n" + "-"*60)
    print("Evaluating PI-DRL Agent...")
    print("-"*60)
    agent_results = trainer.evaluate_agent(n_episodes=args.n_eval_episodes)
    
    # Evaluate baseline
    print("\n" + "-"*60)
    print("Evaluating Baseline Thermostat...")
    print("-"*60)
    baseline_results = trainer.evaluate_baseline(n_episodes=args.n_eval_episodes)
    
    # Save results
    results = {
        'agent': agent_results,
        'baseline': baseline_results
    }
    
    results_path = os.path.join(dirs['results'], 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualizations
    print_header("GENERATING PUBLICATION FIGURES")
    
    visualizer = ResultVisualizer(output_dir=dirs['figures'], dpi=300)
    visualizer.generate_all_figures(
        agent_results=agent_results,
        baseline_results=baseline_results,
        model=trainer.model,
        env=trainer.create_env()
    )
    
    # Generate tables
    print_header("GENERATING PUBLICATION TABLES")
    
    table_gen = PublicationTableGenerator(output_dir=dirs['tables'])
    tables = table_gen.generate_all_tables(
        env_params=trainer.env_params,
        ppo_params=trainer.ppo_params,
        baseline_results=baseline_results,
        agent_results=agent_results
    )
    
    return results, tables


def mode_ablation(dirs: dict, args):
    """Ablation study mode"""
    print_header("MODE: ABLATION STUDY")
    
    # Initialize trainer with full model
    trainer = PI_DRL_Trainer(output_dir=dirs['models'], device=args.device)
    
    # Check if model exists
    model_path = os.path.join(dirs['models'], 'best_model.zip')
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        trainer.initialize_agent()
        trainer.model = trainer.model.load(model_path)
    else:
        print("No trained model found. Training new model...")
        trainer.initialize_agent()
        trainer.train(total_timesteps=args.timesteps)
    
    # Run ablation study
    ablation_results = trainer.run_ablation_study(n_episodes=args.n_eval_episodes)
    
    # Save results
    results_path = os.path.join(dirs['results'], 'ablation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(ablation_results, f)
    print(f"\nAblation results saved to: {results_path}")
    
    # Generate Table 3
    print_header("GENERATING ABLATION TABLE")
    
    table_gen = PublicationTableGenerator(output_dir=dirs['tables'])
    table3 = table_gen.table3_ablation_study(ablation_results)
    
    return ablation_results


def mode_full(dirs: dict, args):
    """Full pipeline: Train + Evaluate + Ablation + Visualize"""
    print_header("MODE: FULL PIPELINE")
    print("This will execute the complete publication pipeline:")
    print("  1. Train PI-DRL agent")
    print("  2. Evaluate performance vs baseline")
    print("  3. Run ablation study")
    print("  4. Generate all figures and tables")
    print()
    
    # Step 1: Train
    print_header("STEP 1/4: TRAINING")
    trainer = mode_train(dirs, args)
    
    # Step 2: Evaluate
    print_header("STEP 2/4: EVALUATION")
    eval_results, tables = mode_evaluate(dirs, args)
    
    # Step 3: Ablation
    print_header("STEP 3/4: ABLATION STUDY")
    ablation_results = mode_ablation(dirs, args)
    
    # Step 4: Generate comprehensive report
    print_header("STEP 4/4: FINAL REPORT GENERATION")
    
    # Regenerate tables with ablation results
    table_gen = PublicationTableGenerator(output_dir=dirs['tables'])
    all_tables = table_gen.generate_all_tables(
        env_params=trainer.env_params,
        ppo_params=trainer.ppo_params,
        baseline_results=eval_results['baseline'],
        agent_results=eval_results['agent'],
        ablation_results=ablation_results
    )
    
    # Generate summary report
    generate_summary_report(dirs, eval_results, ablation_results)
    
    print_header("PIPELINE COMPLETE")
    print(f"All outputs saved to: {dirs['base']}")
    print(f"\nDirectory structure:")
    print(f"  📂 {dirs['base']}/")
    print(f"    ├── 📂 models/       (Trained PPO models)")
    print(f"    ├── 📂 figures/      (Publication-quality figures)")
    print(f"    ├── 📂 tables/       (LaTeX and CSV tables)")
    print(f"    ├── 📂 results/      (Pickled results)")
    print(f"    └── 📄 SUMMARY_REPORT.txt")
    print()


def generate_summary_report(dirs: dict, eval_results: dict, ablation_results: dict):
    """Generate comprehensive summary report"""
    report_path = os.path.join(dirs['base'], 'SUMMARY_REPORT.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHYSICS-INFORMED DEEP REINFORCEMENT LEARNING\n")
        f.write("Smart Home Energy Management System\n")
        f.write("Applied Energy Journal Submission\n")
        f.write("="*80 + "\n\n")
        
        # Performance Summary
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n\n")
        
        baseline = eval_results['baseline']
        agent = eval_results['agent']
        
        f.write("Baseline Thermostat:\n")
        f.write(f"  Energy Cost:      ${baseline['mean_cost']:.2f}\n")
        f.write(f"  Discomfort:       {baseline['mean_discomfort']:.2f} °C·h\n")
        f.write(f"  Equipment Cycles: {int(baseline['mean_switches'])}\n\n")
        
        f.write("Proposed PI-DRL:\n")
        f.write(f"  Energy Cost:      ${agent['mean_cost']:.2f}\n")
        f.write(f"  Discomfort:       {agent['mean_discomfort']:.2f} °C·h\n")
        f.write(f"  Equipment Cycles: {int(agent['mean_switches'])}\n\n")
        
        # Calculate improvements
        cost_reduction = (baseline['mean_cost'] - agent['mean_cost']) / baseline['mean_cost'] * 100
        cycle_reduction = (baseline['mean_switches'] - agent['mean_switches']) / baseline['mean_switches'] * 100
        
        f.write("IMPROVEMENTS:\n")
        f.write(f"  Cost Reduction:   {cost_reduction:.1f}%\n")
        f.write(f"  Cycle Reduction:  {cycle_reduction:.1f}%\n\n")
        
        # Ablation Study Summary
        if ablation_results:
            f.write("\nABLATION STUDY\n")
            f.write("-"*80 + "\n\n")
            
            if 'no_cycling' in ablation_results:
                no_cycle = ablation_results['no_cycling']
                full = ablation_results['full_model']
                
                f.write("Impact of Cycling Penalty:\n")
                f.write(f"  Without penalty: {int(no_cycle['mean_switches'])} cycles/day\n")
                f.write(f"  With penalty:    {int(full['mean_switches'])} cycles/day\n")
                f.write(f"  Reduction:       {(1 - full['mean_switches']/no_cycle['mean_switches'])*100:.1f}%\n\n")
        
        # Key Contributions
        f.write("\nKEY CONTRIBUTIONS FOR MANUSCRIPT\n")
        f.write("-"*80 + "\n\n")
        f.write("1. PHYSICS-INFORMED REWARD:\n")
        f.write("   Novel cycling penalty prevents short-cycling (hardware protection)\n\n")
        f.write("2. MULTI-OBJECTIVE OPTIMIZATION:\n")
        f.write("   Balances cost, comfort, and equipment longevity\n\n")
        f.write("3. DEMAND RESPONSE CAPABILITY:\n")
        f.write("   Agent learns to avoid peak pricing periods\n\n")
        f.write("4. HIGH-RESOLUTION CONTROL:\n")
        f.write("   Leverages 1-minute resolution of AMPds2 dataset\n\n")
        
        # Files Generated
        f.write("\nGENERATED FILES\n")
        f.write("-"*80 + "\n\n")
        f.write("Figures (publication-quality, 300 DPI):\n")
        f.write("  • fig1_system_heartbeat.png\n")
        f.write("  • fig2_control_policy_heatmap.png\n")
        f.write("  • fig3_multiobjective_radar.png\n")
        f.write("  • fig4_energy_carpet_plot.png\n\n")
        
        f.write("Tables (CSV + LaTeX):\n")
        f.write("  • table1_simulation_parameters\n")
        f.write("  • table2_performance_comparison\n")
        f.write("  • table3_ablation_study\n\n")
        
        f.write("="*80 + "\n")
        f.write("Report generated successfully!\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Physics-Informed DRL for Smart Home Energy Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new agent
  python main_pi_drl.py --mode train --timesteps 500000
  
  # Evaluate existing agent
  python main_pi_drl.py --mode evaluate --n-eval-episodes 20
  
  # Run ablation study
  python main_pi_drl.py --mode ablation
  
  # Complete pipeline
  python main_pi_drl.py --mode full --timesteps 500000
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['train', 'evaluate', 'ablation', 'full'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs_pi_drl',
        help='Base output directory'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=500000,
        help='Training timesteps'
    )
    
    parser.add_argument(
        '--n-eval-episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes'
    )
    
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=50000,
        help='Checkpoint save frequency'
    )
    
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=10000,
        help='Evaluation frequency during training'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for training'
    )
    
    args = parser.parse_args()
    
    # Setup directories
    dirs = setup_directories(args.output_dir)
    
    # Print configuration
    print_header("CONFIGURATION")
    print(f"Mode:             {args.mode}")
    print(f"Output Directory: {dirs['base']}")
    print(f"Training Steps:   {args.timesteps:,}")
    print(f"Eval Episodes:    {args.n_eval_episodes}")
    print(f"Device:           {args.device}")
    
    # Execute mode
    if args.mode == 'train':
        mode_train(dirs, args)
    elif args.mode == 'evaluate':
        mode_evaluate(dirs, args)
    elif args.mode == 'ablation':
        mode_ablation(dirs, args)
    elif args.mode == 'full':
        mode_full(dirs, args)
    
    print("\n✓ Execution complete!\n")


if __name__ == "__main__":
    main()
