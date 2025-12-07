"""
06_nsga2_optimization.py
=========================
NSGA-II Multi-Objective Optimization for Heat Pump Retrofits

This module implements NSGA-II to find Pareto-optimal solutions
that minimize both annualized cost and CO2 emissions.

Author: Fafa (GitHub: Fateme9977)
Institution: K. N. Toosi University of Technology
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
from copy import deepcopy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# Import scenario definitions from previous module
from src.retrofit_scenarios import (
    RETROFIT_MEASURES, HEAT_PUMP_OPTIONS,
    EnergyPrices, GridEmissionFactor, GRID_EMISSIONS,
    DEFAULT_PRICES, REGIONAL_PRICES,
    evaluate_scenario
)


@dataclass
class Individual:
    """
    Represents a solution (individual) in the optimization.
    
    Chromosome encoding:
    - retrofit_idx: Index of retrofit measure (0-N)
    - hp_idx: Index of heat pump option (0-M)
    """
    retrofit_idx: int
    hp_idx: int
    objectives: Tuple[float, float] = None  # (cost, emissions)
    rank: int = None
    crowding_distance: float = 0.0


@dataclass
class NSGA2Config:
    """Configuration for NSGA-II algorithm."""
    population_size: int = 100
    n_generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    tournament_size: int = 2
    random_seed: int = 42


class NSGA2Optimizer:
    """
    NSGA-II Multi-Objective Optimizer for heat pump retrofit decisions.
    """
    
    def __init__(
        self,
        baseline_intensity: float,
        heated_sqft: float,
        hdd: float,
        envelope_class: str,
        prices: EnergyPrices = DEFAULT_PRICES,
        grid_emissions: GridEmissionFactor = GRID_EMISSIONS['national'],
        config: NSGA2Config = None
    ):
        self.baseline_intensity = baseline_intensity
        self.heated_sqft = heated_sqft
        self.hdd = hdd
        self.envelope_class = envelope_class
        self.prices = prices
        self.grid_emissions = grid_emissions
        self.config = config or NSGA2Config()
        
        # Build decision options
        self._build_decision_options()
        
        # Set random seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # Storage for results
        self.history = []
        self.pareto_front = []
    
    def _build_decision_options(self):
        """Build lists of applicable retrofit and HP options."""
        # Filter retrofits by envelope class
        self.retrofit_options = []
        self.retrofit_keys = []
        for key, measure in RETROFIT_MEASURES.items():
            if self.envelope_class in measure.applicable_envelope_classes:
                self.retrofit_options.append(measure)
                self.retrofit_keys.append(key)
        
        # Heat pump options (all applicable)
        self.hp_options = []
        self.hp_keys = []
        for key, hp in HEAT_PUMP_OPTIONS.items():
            self.hp_options.append(hp)
            self.hp_keys.append(key)
        
        self.n_retrofits = len(self.retrofit_options)
        self.n_hps = len(self.hp_options)
        
        logger.info(f"Decision space: {self.n_retrofits} retrofits × {self.n_hps} HP options")
    
    def evaluate(self, individual: Individual) -> Tuple[float, float]:
        """
        Evaluate an individual's objectives.
        
        Returns (annualized_cost, annual_emissions)
        """
        retrofit = self.retrofit_options[individual.retrofit_idx]
        hp = self.hp_options[individual.hp_idx]
        
        result = evaluate_scenario(
            self.baseline_intensity,
            self.heated_sqft,
            self.hdd,
            retrofit,
            hp,
            self.prices,
            self.grid_emissions
        )
        
        return (result['total_annual_cost'], result['annual_emissions_kg'])
    
    def create_random_individual(self) -> Individual:
        """Create a random individual."""
        retrofit_idx = random.randint(0, self.n_retrofits - 1)
        hp_idx = random.randint(0, self.n_hps - 1)
        return Individual(retrofit_idx, hp_idx)
    
    def initialize_population(self) -> List[Individual]:
        """Initialize population with random individuals."""
        population = []
        for _ in range(self.config.population_size):
            ind = self.create_random_individual()
            ind.objectives = self.evaluate(ind)
            population.append(ind)
        return population
    
    def non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Perform non-dominated sorting (fast non-dominated sort).
        
        Returns list of fronts, where front[0] is the Pareto front.
        """
        n = len(population)
        domination_count = [0] * n  # Number of solutions that dominate this one
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by this one
        
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])
        
        # Build subsequent fronts
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for ind in fronts[front_idx]:
                i = population.index(ind)
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = front_idx + 1
                        next_front.append(population[j])
            front_idx += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2 (both objectives minimized)."""
        better_in_any = False
        for obj1, obj2 in zip(ind1.objectives, ind2.objectives):
            if obj1 > obj2:  # ind1 is worse in this objective
                return False
            if obj1 < obj2:
                better_in_any = True
        return better_in_any
    
    def calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for individuals in a front."""
        n = len(front)
        if n == 0:
            return
        
        # Initialize distances
        for ind in front:
            ind.crowding_distance = 0
        
        # For each objective
        n_objectives = len(front[0].objectives)
        for obj_idx in range(n_objectives):
            # Sort by this objective
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            # Boundary points get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range == 0:
                continue
            
            # Calculate distances for intermediate points
            for i in range(1, n - 1):
                distance = (front[i + 1].objectives[obj_idx] - front[i - 1].objectives[obj_idx]) / obj_range
                front[i].crowding_distance += distance
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Binary tournament selection based on rank and crowding distance."""
        candidates = random.sample(population, self.config.tournament_size)
        
        # Compare by rank first, then crowding distance
        best = candidates[0]
        for candidate in candidates[1:]:
            if candidate.rank < best.rank:
                best = candidate
            elif candidate.rank == best.rank and candidate.crowding_distance > best.crowding_distance:
                best = candidate
        
        return deepcopy(best)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        if random.random() > self.config.crossover_prob:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Swap retrofit or HP selection
        child1 = Individual(parent1.retrofit_idx, parent2.hp_idx)
        child2 = Individual(parent2.retrofit_idx, parent1.hp_idx)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """Random mutation."""
        if random.random() < self.config.mutation_prob:
            # Mutate retrofit
            individual.retrofit_idx = random.randint(0, self.n_retrofits - 1)
        
        if random.random() < self.config.mutation_prob:
            # Mutate HP
            individual.hp_idx = random.randint(0, self.n_hps - 1)
        
        return individual
    
    def create_offspring(self, population: List[Individual]) -> List[Individual]:
        """Create offspring population through selection, crossover, mutation."""
        offspring = []
        
        while len(offspring) < self.config.population_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Evaluate
            child1.objectives = self.evaluate(child1)
            child2.objectives = self.evaluate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.config.population_size]
    
    def select_next_generation(self, population: List[Individual], 
                               offspring: List[Individual]) -> List[Individual]:
        """Select next generation from combined population."""
        combined = population + offspring
        
        # Non-dominated sorting
        fronts = self.non_dominated_sort(combined)
        
        # Select individuals to fill next generation
        next_gen = []
        front_idx = 0
        
        while len(next_gen) + len(fronts[front_idx]) <= self.config.population_size:
            # Add entire front
            for ind in fronts[front_idx]:
                next_gen.append(ind)
            front_idx += 1
            if front_idx >= len(fronts):
                break
        
        # Fill remaining spots from next front using crowding distance
        if len(next_gen) < self.config.population_size and front_idx < len(fronts):
            remaining = self.config.population_size - len(next_gen)
            self.calculate_crowding_distance(fronts[front_idx])
            fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            next_gen.extend(fronts[front_idx][:remaining])
        
        return next_gen
    
    def run(self) -> List[Individual]:
        """Run the NSGA-II optimization."""
        logger.info("Starting NSGA-II optimization...")
        
        # Initialize
        population = self.initialize_population()
        
        for gen in range(self.config.n_generations):
            # Create offspring
            offspring = self.create_offspring(population)
            
            # Select next generation
            population = self.select_next_generation(population, offspring)
            
            # Record Pareto front
            fronts = self.non_dominated_sort(population)
            self.pareto_front = fronts[0]
            
            # Log progress
            if (gen + 1) % 10 == 0:
                logger.info(f"Generation {gen + 1}/{self.config.n_generations}, "
                           f"Pareto front size: {len(self.pareto_front)}")
            
            # Store history
            self.history.append({
                'generation': gen + 1,
                'pareto_front_size': len(self.pareto_front),
                'best_cost': min(ind.objectives[0] for ind in self.pareto_front),
                'best_emissions': min(ind.objectives[1] for ind in self.pareto_front),
            })
        
        logger.info(f"Optimization complete. Final Pareto front: {len(self.pareto_front)} solutions")
        
        return self.pareto_front
    
    def get_pareto_front_details(self) -> pd.DataFrame:
        """Get detailed information about Pareto-optimal solutions."""
        details = []
        
        for ind in self.pareto_front:
            retrofit = self.retrofit_options[ind.retrofit_idx]
            hp = self.hp_options[ind.hp_idx]
            
            details.append({
                'retrofit': retrofit.name,
                'heat_pump': hp.name if hp else 'Gas Furnace',
                'annual_cost': ind.objectives[0],
                'annual_emissions_kg': ind.objectives[1],
                'crowding_distance': ind.crowding_distance,
            })
        
        return pd.DataFrame(details)


def generate_table6_nsga2_config(config: NSGA2Config) -> pd.DataFrame:
    """Generate Table 6: NSGA-II configuration and settings."""
    logger.info("Generating Table 6: NSGA-II configuration")
    
    config_data = [
        ('Population Size', config.population_size, '-'),
        ('Number of Generations', config.n_generations, '-'),
        ('Crossover Probability', config.crossover_prob, '-'),
        ('Mutation Probability', config.mutation_prob, '-'),
        ('Tournament Size', config.tournament_size, '-'),
        ('Random Seed', config.random_seed, '-'),
        ('Objectives', 'Minimize Cost, Minimize CO2', '-'),
        ('Decision Variables', 'Retrofit Measure, Heat Pump Type', '-'),
    ]
    
    config_df = pd.DataFrame(config_data, columns=['Parameter', 'Value', 'Unit'])
    config_df.to_csv(TABLES_DIR / "table6_nsga2_config.csv", index=False)
    
    return config_df


def generate_figure8_pareto_fronts(
    results: Dict[str, List[Individual]],
    save_path: Path = None
):
    """Generate Figure 8: Example Pareto fronts for different archetypes."""
    logger.info("Generating Figure 8: Pareto fronts")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for ax_idx, (title, climate_results) in enumerate(results.items()):
        ax = axes[ax_idx]
        
        for i, (label, pareto_front) in enumerate(climate_results.items()):
            costs = [ind.objectives[0] for ind in pareto_front]
            emissions = [ind.objectives[1] for ind in pareto_front]
            
            ax.scatter(
                costs, emissions,
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                s=100, alpha=0.7,
                label=label
            )
            
            # Connect Pareto front points
            sorted_idx = np.argsort(costs)
            ax.plot(
                [costs[i] for i in sorted_idx],
                [emissions[i] for i in sorted_idx],
                c=colors[i % len(colors)], alpha=0.5, linestyle='--'
            )
        
        ax.set_xlabel('Annual Cost ($)', fontsize=12)
        ax.set_ylabel('Annual CO₂ Emissions (kg)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    
    plt.close()


def run_optimization_for_archetypes(df: pd.DataFrame, n_samples: int = 5) -> Dict:
    """
    Run NSGA-II optimization for representative archetypes.
    """
    logger.info("Running optimization for archetypes...")
    
    results = {}
    
    # Define archetypes by climate and envelope class
    archetypes = [
        ('cold', 'poor'),
        ('cold', 'medium'),
        ('mixed', 'poor'),
        ('mixed', 'medium'),
        ('mild', 'poor'),
    ]
    
    config = NSGA2Config(
        population_size=50,
        n_generations=50,
        random_seed=42
    )
    
    for climate, envelope in archetypes:
        mask = (df['climate_zone'] == climate) & (df['envelope_class'] == envelope)
        subset = df[mask]
        
        if len(subset) < n_samples:
            logger.warning(f"Not enough samples for {climate}/{envelope}")
            continue
        
        # Sample representative households
        sample = subset.sample(n=min(n_samples, len(subset)), random_state=42)
        
        archetype_results = []
        
        for _, row in sample.iterrows():
            optimizer = NSGA2Optimizer(
                baseline_intensity=row['Thermal_Intensity_I'],
                heated_sqft=row['A_heated'],
                hdd=row['HDD65'],
                envelope_class=envelope,
                config=config
            )
            
            pareto_front = optimizer.run()
            archetype_results.append({
                'household': row.get('DOEID', 'unknown'),
                'pareto_front': pareto_front,
                'details': optimizer.get_pareto_front_details()
            })
        
        results[f"{climate}_{envelope}"] = archetype_results
    
    return results


def run_nsga2_pipeline() -> Dict:
    """Main function to run the NSGA-II optimization pipeline."""
    logger.info("=" * 60)
    logger.info("NSGA-II Multi-Objective Optimization")
    logger.info("=" * 60)
    
    # Load data
    data_path = OUTPUT_DIR / "03_gas_heated_clean.csv"
    df = pd.read_csv(data_path)
    
    # Generate Table 6
    config = NSGA2Config(population_size=100, n_generations=100)
    generate_table6_nsga2_config(config)
    
    # Run optimization for archetypes
    archetype_results = run_optimization_for_archetypes(df, n_samples=3)
    
    # Save detailed results
    all_pareto = []
    for archetype, results_list in archetype_results.items():
        for result in results_list:
            details = result['details']
            details['archetype'] = archetype
            details['household'] = result['household']
            all_pareto.append(details)
    
    if all_pareto:
        combined_df = pd.concat(all_pareto, ignore_index=True)
        combined_df.to_csv(RESULTS_DIR / "nsga2_pareto_results.csv", index=False)
    
    # Generate Figure 8 - Example for cold and mild climates
    if archetype_results:
        # Organize for plotting
        plot_results = {
            '(a) Cold Climate Division': {},
            '(b) Mild Climate Division': {},
        }
        
        for archetype, results_list in archetype_results.items():
            climate, envelope = archetype.split('_')
            
            if climate == 'cold' and len(results_list) > 0:
                plot_results['(a) Cold Climate Division'][envelope] = results_list[0]['pareto_front']
            elif climate == 'mild' and len(results_list) > 0:
                plot_results['(b) Mild Climate Division'][envelope] = results_list[0]['pareto_front']
        
        generate_figure8_pareto_fronts(
            plot_results,
            FIGURES_DIR / "figure8_pareto_fronts.png"
        )
    
    logger.info("=" * 60)
    logger.info("NSGA-II optimization complete!")
    logger.info("=" * 60)
    
    return {
        'archetype_results': archetype_results,
        'config': config,
    }


if __name__ == "__main__":
    # Handle import path
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    results = run_nsga2_pipeline()
    
    print("\n" + "=" * 60)
    print("NSGA-II OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    for archetype, results_list in results['archetype_results'].items():
        print(f"\n{archetype}:")
        for result in results_list:
            print(f"  Pareto front size: {len(result['pareto_front'])}")
