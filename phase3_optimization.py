"""
Phase 3: Optimization Framework (MPC + Genetic Algorithm)
Objective: Minimize cost and maximize comfort using GA with surrogate model
"""

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random
from typing import Callable, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class BuildingOptimizer:
    """Genetic Algorithm optimizer for HVAC control"""
    
    def __init__(self, surrogate_model, weather_forecast: pd.DataFrame,
                 energy_price: np.ndarray = None, comfort_weight: float = 0.5):
        """
        Args:
            surrogate_model: Trained surrogate model for predictions
            weather_forecast: DataFrame with weather data for optimization horizon
            energy_price: Hourly energy prices (if None, uses flat rate)
            comfort_weight: Weight for comfort in objective function (0-1)
        """
        self.surrogate_model = surrogate_model
        self.weather_forecast = weather_forecast.copy()
        self.comfort_weight = comfort_weight
        
        # Energy pricing (time-of-use or flat)
        if energy_price is None:
            # Default: peak hours (14-18) cost more
            self.energy_price = np.ones(len(weather_forecast)) * 0.15  # $/kWh base
            peak_hours = (weather_forecast['hour_of_day'] >= 14) & \
                        (weather_forecast['hour_of_day'] <= 18)
            self.energy_price[peak_hours] = 0.25  # Peak pricing
        else:
            self.energy_price = energy_price
        
        # GA parameters
        self.population_size = 50
        self.generations = 100
        self.crossover_prob = 0.7
        self.mutation_prob = 0.3
        
        # HVAC constraints
        self.min_setpoint = 19.0  # °C
        self.max_setpoint = 26.0  # °C
        self.comfort_min = 20.0    # °C
        self.comfort_max = 25.0    # °C
        
        # Initialize DEAP
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP framework for genetic algorithm"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Individual: 24-hour schedule of HVAC setpoints
        self.toolbox.register("attr_setpoint", 
                             lambda: random.uniform(self.min_setpoint, self.max_setpoint))
        self.toolbox.register("individual", 
                             tools.initRepeat, creator.Individual,
                             self.toolbox.attr_setpoint, n=24)
        self.toolbox.register("population", 
                             tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_schedule)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, 
                             mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _evaluate_schedule(self, individual: List[float]) -> Tuple[float]:
        """
        Evaluate a 24-hour HVAC schedule
        
        Objective: Minimize J = C_energy + w * D_comfort
        """
        # Clip setpoints to valid range
        schedule = np.clip(individual, self.min_setpoint, self.max_setpoint)
        
        # Prepare features for surrogate model
        features = []
        for hour in range(24):
            if hour < len(self.weather_forecast):
                row = self.weather_forecast.iloc[hour]
                feature_vector = np.array([
                    row['outdoor_temp'],
                    row['solar_radiation'],
                    row['humidity'],
                    row['hour_of_day'],
                    row['day_of_week'],
                    schedule[hour]  # HVAC setpoint
                ]).reshape(1, -1)
                features.append(feature_vector)
        
        features = np.vstack(features)
        
        # Predict energy and temperature
        predictions = self.surrogate_model.predict(features)
        energy_consumption = predictions[:, 0]  # kWh
        indoor_temp = predictions[:, 1]  # °C
        
        # Calculate energy cost
        energy_cost = np.sum(energy_consumption * self.energy_price[:len(energy_consumption)])
        
        # Calculate comfort violation (PMV approximation)
        # PMV ≈ 0 when temp is in comfort zone (20-25°C)
        comfort_violation = 0.0
        for temp in indoor_temp:
            if temp < self.comfort_min:
                comfort_violation += (self.comfort_min - temp) ** 2
            elif temp > self.comfort_max:
                comfort_violation += (temp - self.comfort_max) ** 2
        
        # Normalize comfort violation (scale to similar magnitude as cost)
        comfort_violation = comfort_violation / 100.0  # Normalization factor
        
        # Objective function
        objective = energy_cost + self.comfort_weight * comfort_violation
        
        return (objective,)
    
    def optimize(self, verbose: bool = True) -> Dict:
        """Run genetic algorithm optimization"""
        random.seed(42)
        np.random.seed(42)
        
        # Initialize population
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Track best solution
        best_fitness = float('inf')
        best_individual = None
        fitness_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Select and clone next generation
            offspring = list(map(self.toolbox.clone, 
                               self.toolbox.select(population, len(population))))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    # Clip to valid range
                    mutant[:] = np.clip(mutant, self.min_setpoint, self.max_setpoint)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Track best
            current_best = tools.selBest(population, 1)[0]
            current_fitness = current_best.fitness.values[0]
            
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_individual = current_best[:]
            
            fitness_history.append(best_fitness)
            
            if verbose and (generation % 10 == 0 or generation == self.generations - 1):
                print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Final evaluation of best solution
        best_schedule = np.clip(best_individual, self.min_setpoint, self.max_setpoint)
        
        # Get detailed results
        features = []
        for hour in range(24):
            if hour < len(self.weather_forecast):
                row = self.weather_forecast.iloc[hour]
                feature_vector = np.array([
                    row['outdoor_temp'],
                    row['solar_radiation'],
                    row['humidity'],
                    row['hour_of_day'],
                    row['day_of_week'],
                    best_schedule[hour]
                ]).reshape(1, -1)
                features.append(feature_vector)
        
        features = np.vstack(features)
        predictions = self.surrogate_model.predict(features)
        energy_consumption = predictions[:, 0]
        indoor_temp = predictions[:, 1]
        
        total_energy = np.sum(energy_consumption)
        total_cost = np.sum(energy_consumption * self.energy_price[:len(energy_consumption)])
        
        # Calculate comfort violations
        comfort_violations = []
        for temp in indoor_temp:
            if temp < self.comfort_min or temp > self.comfort_max:
                comfort_violations.append(1)
            else:
                comfort_violations.append(0)
        
        comfort_violation_hours = np.sum(comfort_violations)
        
        results = {
            'best_schedule': best_schedule,
            'best_fitness': best_fitness,
            'total_energy': total_energy,
            'total_cost': total_cost,
            'energy_consumption': energy_consumption,
            'indoor_temp': indoor_temp,
            'comfort_violation_hours': comfort_violation_hours,
            'fitness_history': fitness_history
        }
        
        return results
    
    def baseline_control(self, fixed_setpoint: float = 23.0) -> Dict:
        """Baseline control: fixed setpoint"""
        schedule = np.full(24, fixed_setpoint)
        
        features = []
        for hour in range(24):
            if hour < len(self.weather_forecast):
                row = self.weather_forecast.iloc[hour]
                feature_vector = np.array([
                    row['outdoor_temp'],
                    row['solar_radiation'],
                    row['humidity'],
                    row['hour_of_day'],
                    row['day_of_week'],
                    schedule[hour]
                ]).reshape(1, -1)
                features.append(feature_vector)
        
        features = np.vstack(features)
        predictions = self.surrogate_model.predict(features)
        energy_consumption = predictions[:, 0]
        indoor_temp = predictions[:, 1]
        
        total_energy = np.sum(energy_consumption)
        total_cost = np.sum(energy_consumption * self.energy_price[:len(energy_consumption)])
        
        comfort_violations = []
        for temp in indoor_temp:
            if temp < self.comfort_min or temp > self.comfort_max:
                comfort_violations.append(1)
            else:
                comfort_violations.append(0)
        
        comfort_violation_hours = np.sum(comfort_violations)
        
        return {
            'schedule': schedule,
            'total_energy': total_energy,
            'total_cost': total_cost,
            'energy_consumption': energy_consumption,
            'indoor_temp': indoor_temp,
            'comfort_violation_hours': comfort_violation_hours
        }


def generate_optimization_constraints_table() -> pd.DataFrame:
    """Generate Table 3: Objective Function & Optimization Constraints"""
    table_data = {
        'Parameter': [
            'Objective Function',
            'Decision Variable',
            'PMV (Predicted Mean Vote)',
            'Algorithm',
            'Time Horizon'
        ],
        'Description': [
            'Cost vs. Comfort Trade-off',
            'HVAC Setpoint (T_set)',
            'Comfort Range',
            'Genetic Algorithm (GA)',
            'Prediction Window'
        ],
        'Value / Constraint': [
            'Minimize: J = C_energy + w · D_comfort',
            '19°C ≤ T_set ≤ 26°C',
            '-0.5 ≤ PMV ≤ +0.5',
            'Population: 50, Generations: 100',
            '24 Hours (Day-ahead)'
        ]
    }
    
    return pd.DataFrame(table_data)


if __name__ == "__main__":
    # Example usage
    from phase1_data_curation import BDG2DataProcessor
    from phase2_surrogate_model import SurrogateModel
    
    # Load data and train surrogate model
    processor = BDG2DataProcessor()
    metadata = processor.load_metadata("metadata.csv")
    residential_buildings = processor.filter_residential_buildings()
    
    building_id = residential_buildings['building_id'].iloc[0]
    df, scalers = processor.process_building(building_id)
    
    # Train surrogate model
    surrogate = SurrogateModel(model_type='xgboost')
    X, y = surrogate.prepare_features(df)
    surrogate.train(X, y, validation_split=0.2, epochs=30)
    
    # Prepare weather forecast for next 24 hours
    forecast = df.tail(24).copy()
    
    # Create optimizer
    optimizer = BuildingOptimizer(surrogate, forecast, comfort_weight=0.5)
    
    # Run optimization
    print("\nRunning optimization...")
    optimized_results = optimizer.optimize(verbose=True)
    
    # Baseline comparison
    baseline_results = optimizer.baseline_control(fixed_setpoint=23.0)
    
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)
    print(f"Optimized Energy: {optimized_results['total_energy']:.2f} kWh")
    print(f"Optimized Cost: ${optimized_results['total_cost']:.2f}")
    print(f"Optimized Comfort Violations: {optimized_results['comfort_violation_hours']} hours")
    print(f"\nBaseline Energy: {baseline_results['total_energy']:.2f} kWh")
    print(f"Baseline Cost: ${baseline_results['total_cost']:.2f}")
    print(f"Baseline Comfort Violations: {baseline_results['comfort_violation_hours']} hours")
    
    # Generate Table 3
    table3 = generate_optimization_constraints_table()
    print("\n" + "="*80)
    print("Table 3: Objective Function & Optimization Constraints")
    print("="*80)
    print(table3.to_string(index=False))
