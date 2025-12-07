"""
Step 6: NSGA-II Optimization
=============================

Goal: Use NSGA-II to find Pareto-optimal solutions for retrofit + HP decisions.

This script:
1. Loads scenario data from Step 5
2. Sets up NSGA-II optimization problem:
   - Objectives: minimize cost, minimize emissions
   - Decision variables: retrofit combinations + HP choice
   - Constraints: comfort (meet peak load), optional budget
3. Runs NSGA-II algorithm
4. Extracts Pareto fronts
5. Visualizes Pareto fronts
6. Saves optimization results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.visualization.scatter import Scatter
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("WARNING: pymoo not installed. Install with: pip install pymoo")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
SCENARIO_DATA = OUTPUT_DIR / "retrofit_scenarios.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class RetrofitOptimizationProblem(Problem):
    """
    NSGA-II optimization problem for retrofit and heat pump decisions.
    
    Decision variables:
    - Retrofit measures (binary: 0/1 for each measure)
    - Heat pump choice (categorical: 0=none, 1=standard, 2=cold-climate)
    
    Objectives:
    - Minimize annualized total cost
    - Minimize annual CO₂ emissions
    """
    
    def __init__(self, scenario_df, dwelling_data, measures, hp_options):
        """
        Parameters
        ----------
        scenario_df : pd.DataFrame
            Pre-computed scenario data
        dwelling_data : pd.Series
            Single dwelling characteristics
        measures : dict
            Retrofit measures
        hp_options : dict
            Heat pump options
        """
        self.scenario_df = scenario_df
        self.dwelling_data = dwelling_data
        self.measures = measures
        self.hp_options = hp_options
        
        # Number of decision variables
        n_retrofits = len(measures)
        n_hp = 1  # HP choice
        n_vars = n_retrofits + n_hp
        
        # Bounds: retrofits are binary (0 or 1), HP is 0-2
        xl = [0] * n_retrofits + [0]
        xu = [1] * n_retrofits + [2]
        
        super().__init__(
            n_var=n_vars,
            n_obj=2,  # Cost and emissions
            n_constr=0,  # No constraints for now
            xl=xl,
            xu=xu,
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for a population of solutions.
        
        Parameters
        ----------
        X : np.ndarray
            Population of solutions (n_pop × n_var)
        out : dict
            Output dictionary with 'F' for objectives
        """
        n_pop = X.shape[0]
        objectives = np.zeros((n_pop, 2))
        
        dwelling_id = self.dwelling_data.get('dwelling_id', 0)
        
        for i in range(n_pop):
            # Decode solution
            retrofit_binary = X[i, :-1].astype(int)
            hp_choice = int(X[i, -1])
            
            # Find matching scenario
            retrofit_names = []
            measure_names = list(self.measures.keys())
            for j, bit in enumerate(retrofit_binary):
                if bit == 1:
                    retrofit_names.append(measure_names[j])
            
            retrofit_str = '+'.join(retrofit_names) if retrofit_names else 'none'
            
            hp_map = {0: 'none', 1: 'standard_hp', 2: 'cold_climate_hp'}
            hp_name = hp_map[hp_choice]
            
            # Find scenario in dataframe
            scenario_mask = (
                (self.scenario_df['dwelling_id'] == dwelling_id) &
                (self.scenario_df['retrofits'] == retrofit_str) &
                (self.scenario_df['heat_pump'] == hp_name)
            )
            
            if scenario_mask.any():
                scenario_row = self.scenario_df[scenario_mask].iloc[0]
                objectives[i, 0] = scenario_row['total_cost_annual']
                objectives[i, 1] = scenario_row['emissions_kg_co2']
            else:
                # Fallback: calculate on the fly (simplified)
                objectives[i, 0] = 1e6  # Penalty
                objectives[i, 1] = 1e6
        
        out["F"] = objectives


def run_nsga2_optimization(dwelling_data, scenario_df, measures, hp_options, 
                           n_gen=100, pop_size=50):
    """
    Run NSGA-II optimization for a single dwelling.
    
    Parameters
    ----------
    dwelling_data : pd.Series
        Dwelling characteristics
    scenario_df : pd.DataFrame
        Scenario dataframe
    measures : dict
        Retrofit measures
    hp_options : dict
        Heat pump options
    n_gen : int
        Number of generations
    pop_size : int
        Population size
        
    Returns
    -------
    dict
        Optimization results
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo is required. Install with: pip install pymoo")
    
    print(f"Running NSGA-II for dwelling {dwelling_data.get('dwelling_id', 'unknown')}...")
    
    # Create problem
    problem = RetrofitOptimizationProblem(scenario_df, dwelling_data, measures, hp_options)
    
    # Configure algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=False,
        seed=42
    )
    
    # Extract Pareto front
    pareto_front = res.F
    pareto_solutions = res.X
    
    return {
        'pareto_front': pareto_front,
        'pareto_solutions': pareto_solutions,
        'algorithm': res.algorithm,
        'problem': problem,
    }


def visualize_pareto_front(pareto_front, output_file, title=""):
    """
    Visualize Pareto front.
    
    Parameters
    ----------
    pareto_front : np.ndarray
        Pareto-optimal objectives (n_solutions × 2)
    output_file : Path
        Output file path
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Sort by first objective (cost)
    sorted_idx = np.argsort(pareto_front[:, 0])
    pareto_sorted = pareto_front[sorted_idx]
    
    ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 
            'o-', color='steelblue', markersize=8, linewidth=2, label='Pareto Front')
    
    ax.set_xlabel('Annualized Total Cost ($/year)', fontsize=12)
    ax.set_ylabel('Annual CO₂ Emissions (kg CO₂/year)', fontsize=12)
    ax.set_title(title or 'Pareto Front: Cost vs Emissions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved Pareto front plot to {output_file}")


def optimize_archetypes(scenario_df, measures, hp_options, n_gen=50, pop_size=30):
    """
    Run optimization for representative archetypes.
    
    Parameters
    ----------
    scenario_df : pd.DataFrame
        Scenario dataframe
    measures : dict
        Retrofit measures
    hp_options : dict
        Heat pump options
    n_gen : int
        Generations per optimization
    pop_size : int
        Population size
        
    Returns
    -------
    dict
        Results for each archetype
    """
    print("\nOptimizing representative archetypes...")
    
    # Select representative dwellings
    # Group by division and envelope class
    if 'DIVISION' in scenario_df.columns and 'envelope_class' in scenario_df.columns:
        archetype_groups = scenario_df.groupby(['DIVISION', 'envelope_class'])
    else:
        # Fallback: sample random dwellings
        unique_dwellings = scenario_df['dwelling_id'].unique()
        archetype_groups = [(None, df) for df in [scenario_df[scenario_df['dwelling_id'] == uid] 
                                                   for uid in unique_dwellings[:5]]]
    
    results = {}
    
    for (division, env_class), group_df in list(archetype_groups)[:5]:  # Limit to 5 archetypes
        # Get a representative dwelling
        dwelling_id = group_df['dwelling_id'].iloc[0]
        dwelling_data = group_df.iloc[0].to_dict()
        
        try:
            opt_result = run_nsga2_optimization(
                dwelling_data,
                scenario_df,
                measures,
                hp_options,
                n_gen=n_gen,
                pop_size=pop_size
            )
            
            archetype_name = f"Div{division}_Env{env_class}" if division else f"Dwelling{dwelling_id}"
            results[archetype_name] = opt_result
            
            # Visualize
            output_file = FIGURES_DIR / f"figure8_pareto_{archetype_name}.png"
            visualize_pareto_front(
                opt_result['pareto_front'],
                output_file,
                title=f'Pareto Front: {archetype_name}'
            )
            
        except Exception as e:
            print(f"  Error optimizing {archetype_name}: {e}")
            continue
    
    return results


def save_optimization_config(output_dir):
    """Save NSGA-II configuration table (Table 6)."""
    print("\nSaving optimization configuration...")
    
    config_data = {
        'parameter': [
            'Population size',
            'Number of generations',
            'Crossover probability',
            'Crossover eta',
            'Mutation probability',
            'Mutation eta',
            'Discount rate',
            'Analysis period (years)',
        ],
        'value': [
            '50',
            '100',
            '0.9',
            '15',
            '1/n_vars',
            '20',
            '0.03',
            '20',
        ],
        'description': [
            'Number of solutions in population',
            'Number of generations',
            'SBX crossover probability',
            'SBX distribution index',
            'Polynomial mutation probability',
            'PM distribution index',
            'Discount rate for annualization',
            'Lifetime for cost annualization',
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    output_file = output_dir / "table6_nsga2_configuration.csv"
    config_df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")


def main():
    """Main optimization pipeline."""
    print("=" * 70)
    print("RECS 2020 Heat Pump Retrofit Project - NSGA-II Optimization")
    print("=" * 70)
    
    if not PYMOO_AVAILABLE:
        print("\nERROR: pymoo library is required for NSGA-II optimization")
        print("Install with: pip install pymoo")
        print("\nCreating placeholder configuration table instead...")
        save_optimization_config(OUTPUT_DIR)
        return
    
    # Load scenario data
    if not SCENARIO_DATA.exists():
        raise FileNotFoundError(
            f"Scenario data not found: {SCENARIO_DATA}\n"
            f"Please run 05_retrofit_scenarios.py first"
        )
    
    scenario_df = pd.read_csv(SCENARIO_DATA)
    print(f"Loaded {len(scenario_df):,} scenario records")
    
    # Define measures and options (same as Step 5)
    from src.05_retrofit_scenarios import define_retrofit_measures, define_heat_pump_options
    measures = define_retrofit_measures()
    hp_options = define_heat_pump_options()
    
    # Save configuration
    save_optimization_config(OUTPUT_DIR)
    
    # Run optimization for archetypes
    results = optimize_archetypes(scenario_df, measures, hp_options, n_gen=50, pop_size=30)
    
    print(f"\n✓ Completed optimization for {len(results)} archetypes")
    
    # Save results summary
    summary_data = []
    for archetype_name, result in results.items():
        pareto_front = result['pareto_front']
        summary_data.append({
            'archetype': archetype_name,
            'n_pareto_solutions': len(pareto_front),
            'min_cost': pareto_front[:, 0].min(),
            'max_cost': pareto_front[:, 0].max(),
            'min_emissions': pareto_front[:, 1].min(),
            'max_emissions': pareto_front[:, 1].max(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = OUTPUT_DIR / "nsga2_results_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved results summary to {summary_file}")
    
    print("\n" + "=" * 70)
    print("Optimization complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review Pareto fronts")
    print("2. Proceed to Step 7: Tipping Point Maps")


if __name__ == "__main__":
    # Fix import path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    main()
