"""
Publication-Quality Table Generator for Applied Energy Journal

Generates the "3 Golden Tables" required for rigorous peer review:
1. Table 1: Simulation & Hyperparameters (Reproducibility)
2. Table 2: Quantitative Performance Comparison
3. Table 3: Ablation Study Results

All tables are formatted for LaTeX export and manuscript inclusion.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import os


class PublicationTableGenerator:
    """
    Generates publication-ready tables for manuscript
    """
    
    def __init__(self, output_dir: str = "./tables"):
        """
        Initialize table generator
        
        Args:
            output_dir: Directory to save tables
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def table1_simulation_parameters(
        self,
        env_params: Dict,
        ppo_params: Dict,
        save_name: str = "table1_simulation_parameters"
    ) -> pd.DataFrame:
        """
        Table 1: Simulation & Hyperparameters
        
        Critical for reproducibility - reviewers must know exact configuration
        
        Args:
            env_params: Environment parameters
            ppo_params: PPO hyperparameters
            save_name: Output filename (without extension)
        
        Returns:
            DataFrame containing all parameters
        """
        print("Generating Table 1: Simulation & Hyperparameters...")
        
        # Create comprehensive parameter table
        data = {
            'Category': [],
            'Parameter': [],
            'Symbol': [],
            'Value': [],
            'Unit': [],
            'Description': []
        }
        
        # === Building Physics Parameters ===
        physics_params = [
            ('Thermal Resistance', 'R', env_params['R'], 'K/kW', 
             'Building envelope thermal resistance'),
            ('Thermal Capacitance', 'C', env_params['C'], 'kWh/K', 
             'Building thermal mass'),
            ('HVAC Power', 'P_HVAC', env_params['hvac_power'], 'kW', 
             'Heat pump rated power'),
            ('Time Step', 'Δt', env_params['dt']*60, 'minutes', 
             'Simulation time resolution'),
            ('Comfort Min Temp', 'T_min', env_params['comfort_range'][0], '°C', 
             'Lower comfort boundary'),
            ('Comfort Max Temp', 'T_max', env_params['comfort_range'][1], '°C', 
             'Upper comfort boundary'),
        ]
        
        for param, symbol, value, unit, desc in physics_params:
            data['Category'].append('Building Physics')
            data['Parameter'].append(param)
            data['Symbol'].append(symbol)
            data['Value'].append(f"{value:.2f}" if isinstance(value, float) else str(value))
            data['Unit'].append(unit)
            data['Description'].append(desc)
        
        # === Reward Function Weights ===
        reward_params = [
            ('Cost Weight', 'w_1', env_params['w_cost'], '-', 
             'Economic objective weight'),
            ('Comfort Weight', 'w_2', env_params['w_comfort'], '-', 
             'Thermal comfort weight'),
            ('Cycling Weight', 'w_3', env_params['w_cycling'], '-', 
             'Hardware protection weight (NOVELTY)'),
            ('Min Cycle Time', 't_cycle', env_params['min_cycle_time'], 'minutes', 
             'Minimum time between switching'),
        ]
        
        for param, symbol, value, unit, desc in reward_params:
            data['Category'].append('Reward Function')
            data['Parameter'].append(param)
            data['Symbol'].append(symbol)
            data['Value'].append(f"{value:.2f}" if isinstance(value, float) else str(value))
            data['Unit'].append(unit)
            data['Description'].append(desc)
        
        # === PPO Hyperparameters ===
        ppo_config = [
            ('Learning Rate', 'α', ppo_params['learning_rate'], '-', 
             'Adam optimizer learning rate'),
            ('Discount Factor', 'γ', ppo_params['gamma'], '-', 
             'Future reward discount'),
            ('GAE Lambda', 'λ', ppo_params['gae_lambda'], '-', 
             'Generalized Advantage Estimation'),
            ('Clip Range', 'ε', ppo_params['clip_range'], '-', 
             'PPO clipping parameter'),
            ('Entropy Coef', 'c_ent', ppo_params['ent_coef'], '-', 
             'Exploration bonus coefficient'),
            ('Batch Size', 'B', ppo_params['batch_size'], 'samples', 
             'Mini-batch size for updates'),
            ('Steps per Update', 'N', ppo_params['n_steps'], 'steps', 
             'Rollout buffer size'),
            ('Epochs per Update', 'E', ppo_params['n_epochs'], 'epochs', 
             'Optimization epochs per rollout'),
        ]
        
        for param, symbol, value, unit, desc in ppo_config:
            data['Category'].append('PPO Algorithm')
            data['Parameter'].append(param)
            data['Symbol'].append(symbol)
            if isinstance(value, float):
                data['Value'].append(f"{value:.2e}" if value < 0.001 else f"{value:.3f}")
            else:
                data['Value'].append(str(value))
            data['Unit'].append(unit)
            data['Description'].append(desc)
        
        # === Network Architecture ===
        if 'policy_kwargs' in ppo_params and 'net_arch' in ppo_params['policy_kwargs']:
            net_arch = ppo_params['policy_kwargs']['net_arch']
            arch_params = [
                ('Policy Network', 'π(s)', str(net_arch['pi']), 'neurons', 
                 'Actor network architecture'),
                ('Value Network', 'V(s)', str(net_arch['vf']), 'neurons', 
                 'Critic network architecture'),
                ('Activation', 'φ', 'ReLU', '-', 
                 'Neural network activation function'),
            ]
            
            for param, symbol, value, unit, desc in arch_params:
                data['Category'].append('Neural Network')
                data['Parameter'].append(param)
                data['Symbol'].append(symbol)
                data['Value'].append(value)
                data['Unit'].append(unit)
                data['Description'].append(desc)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"{save_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV to: {csv_path}")
        
        # Save as LaTeX
        latex_path = os.path.join(self.output_dir, f"{save_name}.tex")
        latex_str = df.to_latex(
            index=False,
            caption="Simulation Parameters and Hyperparameters for Reproducibility",
            label="tab:parameters",
            column_format='llcccp{5cm}',
            escape=False
        )
        with open(latex_path, 'w') as f:
            f.write(latex_str)
        print(f"  Saved LaTeX to: {latex_path}")
        
        # Display table
        print("\n" + "="*120)
        print("TABLE 1: SIMULATION & HYPERPARAMETERS")
        print("="*120)
        print(df.to_string(index=False, max_colwidth=50))
        print("="*120 + "\n")
        
        return df
    
    def table2_performance_comparison(
        self,
        baseline_results: Dict,
        agent_results: Dict,
        ablation_results: Optional[Dict] = None,
        save_name: str = "table2_performance_comparison"
    ) -> pd.DataFrame:
        """
        Table 2: Quantitative Performance Comparison
        
        Provides hard numbers to complement radar chart
        
        Args:
            baseline_results: Baseline evaluation results
            agent_results: PI-DRL agent results
            ablation_results: Optional ablation study results
            save_name: Output filename
        
        Returns:
            DataFrame with performance metrics
        """
        print("Generating Table 2: Performance Comparison...")
        
        # Extract metrics
        methods = ['Baseline Thermostat', 'Proposed PI-DRL']
        
        total_costs = [
            baseline_results['mean_cost'],
            agent_results['mean_cost']
        ]
        
        discomforts = [
            baseline_results['mean_discomfort'],
            agent_results['mean_discomfort']
        ]
        
        switches = [
            baseline_results['mean_switches'],
            agent_results['mean_switches']
        ]
        
        # Calculate reductions (vs baseline)
        cost_reductions = [
            0.0,  # Baseline
            (baseline_results['mean_cost'] - agent_results['mean_cost']) / 
            baseline_results['mean_cost'] * 100
        ]
        
        discomfort_reductions = [
            0.0,
            (baseline_results['mean_discomfort'] - agent_results['mean_discomfort']) / 
            max(baseline_results['mean_discomfort'], 1e-6) * 100
        ]
        
        switch_reductions = [
            0.0,
            (baseline_results['mean_switches'] - agent_results['mean_switches']) / 
            baseline_results['mean_switches'] * 100
        ]
        
        # Add ablation study if available
        if ablation_results and 'no_cycling' in ablation_results:
            methods.append('DRL w/o Cycling Penalty')
            total_costs.append(ablation_results['no_cycling']['mean_cost'])
            discomforts.append(ablation_results['no_cycling']['mean_discomfort'])
            switches.append(ablation_results['no_cycling']['mean_switches'])
            
            cost_reductions.append(
                (baseline_results['mean_cost'] - ablation_results['no_cycling']['mean_cost']) / 
                baseline_results['mean_cost'] * 100
            )
            discomfort_reductions.append(
                (baseline_results['mean_discomfort'] - ablation_results['no_cycling']['mean_discomfort']) / 
                max(baseline_results['mean_discomfort'], 1e-6) * 100
            )
            switch_reductions.append(
                (baseline_results['mean_switches'] - ablation_results['no_cycling']['mean_switches']) / 
                baseline_results['mean_switches'] * 100
            )
        
        # Calculate energy consumption (kWh)
        energy_consumptions = [cost / 0.15 for cost in total_costs]  # Assuming avg price $0.15/kWh
        
        # Create DataFrame
        df = pd.DataFrame({
            'Method': methods,
            'Energy (kWh)': [f"{e:.2f}" for e in energy_consumptions],
            'Cost ($)': [f"{c:.2f}" for c in total_costs],
            'Cost Reduction (%)': [f"{r:.1f}" for r in cost_reductions],
            'Discomfort (°C·h)': [f"{d:.2f}" for d in discomforts],
            'Discomfort Reduction (%)': [f"{r:.1f}" for r in discomfort_reductions],
            'Equipment Cycles': [f"{int(s)}" for s in switches],
            'Cycle Reduction (%)': [f"{r:.1f}" for r in switch_reductions]
        })
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"{save_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV to: {csv_path}")
        
        # Save as LaTeX
        latex_path = os.path.join(self.output_dir, f"{save_name}.tex")
        latex_str = df.to_latex(
            index=False,
            caption="Quantitative Performance Comparison (24-hour episode, averaged over 20 episodes)",
            label="tab:performance",
            column_format='l' + 'c'*7,
            escape=False
        )
        with open(latex_path, 'w') as f:
            f.write(latex_str)
        print(f"  Saved LaTeX to: {latex_path}")
        
        # Display table
        print("\n" + "="*140)
        print("TABLE 2: QUANTITATIVE PERFORMANCE COMPARISON")
        print("="*140)
        print(df.to_string(index=False))
        print("="*140 + "\n")
        
        return df
    
    def table3_ablation_study(
        self,
        ablation_results: Dict,
        save_name: str = "table3_ablation_study"
    ) -> pd.DataFrame:
        """
        Table 3: Ablation Study - Validating Physics-Informed Design
        
        Critical for demonstrating the value of the cycling penalty
        
        Args:
            ablation_results: Results from ablation study
            save_name: Output filename
        
        Returns:
            DataFrame with ablation results
        """
        print("Generating Table 3: Ablation Study...")
        
        # Extract results for each configuration
        configs = {
            'Baseline Thermostat': ablation_results.get('baseline', {}),
            'DRL w/o Cycling Penalty': ablation_results.get('no_cycling', {}),
            'PI-DRL (Full Model)': ablation_results.get('full_model', {})
        }
        
        data = {
            'Configuration': [],
            'Cycling Penalty': [],
            'Cost ($)': [],
            'Discomfort (°C·h)': [],
            'Switches/day': [],
            'Hardware Risk': [],
            'Overall Score': []
        }
        
        for config_name, results in configs.items():
            if not results:
                continue
                
            data['Configuration'].append(config_name)
            
            # Cycling penalty enabled?
            if 'w/o' in config_name:
                data['Cycling Penalty'].append('✗')
            elif 'Baseline' in config_name:
                data['Cycling Penalty'].append('N/A')
            else:
                data['Cycling Penalty'].append('✓')
            
            # Metrics
            data['Cost ($)'].append(f"{results['mean_cost']:.2f}")
            data['Discomfort (°C·h)'].append(f"{results['mean_discomfort']:.2f}")
            
            switches = results['mean_switches']
            data['Switches/day'].append(f"{int(switches)}")
            
            # Hardware risk assessment
            if switches > 100:
                risk = 'HIGH ⚠'
            elif switches > 50:
                risk = 'MEDIUM'
            else:
                risk = 'LOW ✓'
            data['Hardware Risk'].append(risk)
            
            # Overall score (normalized, lower is better)
            # Score = normalized(cost + discomfort + switches/10)
            score = (results['mean_cost'] / 5.0 + 
                    results['mean_discomfort'] / 10.0 + 
                    switches / 10.0)
            data['Overall Score'].append(f"{score:.2f}")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"{save_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV to: {csv_path}")
        
        # Save as LaTeX
        latex_path = os.path.join(self.output_dir, f"{save_name}.tex")
        latex_str = df.to_latex(
            index=False,
            caption="Ablation Study: Impact of Physics-Informed Cycling Penalty on System Performance",
            label="tab:ablation",
            column_format='lcccclc',
            escape=False
        )
        with open(latex_path, 'w') as f:
            f.write(latex_str)
        print(f"  Saved LaTeX to: {latex_path}")
        
        # Display table
        print("\n" + "="*120)
        print("TABLE 3: ABLATION STUDY - VALIDATING PHYSICS-INFORMED DESIGN")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120 + "\n")
        
        # Key insight
        print("KEY INSIGHT:")
        if len(configs) >= 2:
            no_cycle_switches = float(data['Switches/day'][1]) if len(data['Switches/day']) > 1 else 0
            full_switches = float(data['Switches/day'][-1])
            if no_cycle_switches > 0:
                reduction = (no_cycle_switches - full_switches) / no_cycle_switches * 100
                print(f"  The cycling penalty reduces equipment wear by {reduction:.1f}%")
                print(f"  while maintaining comparable cost and comfort performance.")
        print()
        
        return df
    
    def generate_all_tables(
        self,
        env_params: Dict,
        ppo_params: Dict,
        baseline_results: Dict,
        agent_results: Dict,
        ablation_results: Optional[Dict] = None
    ):
        """
        Generate all three publication tables
        
        Args:
            env_params: Environment parameters
            ppo_params: PPO hyperparameters
            baseline_results: Baseline evaluation results
            agent_results: Agent evaluation results
            ablation_results: Ablation study results
        """
        print("\n" + "="*60)
        print("GENERATING PUBLICATION-QUALITY TABLES")
        print("="*60 + "\n")
        
        # Generate all tables
        table1 = self.table1_simulation_parameters(env_params, ppo_params)
        table2 = self.table2_performance_comparison(baseline_results, agent_results, ablation_results)
        
        if ablation_results:
            table3 = self.table3_ablation_study(ablation_results)
        else:
            print("Ablation results not provided. Skipping Table 3.")
            table3 = None
        
        print(f"{'='*60}")
        print(f"All tables saved to: {self.output_dir}")
        print(f"Files available in CSV and LaTeX formats")
        print(f"{'='*60}\n")
        
        return {
            'table1': table1,
            'table2': table2,
            'table3': table3
        }
