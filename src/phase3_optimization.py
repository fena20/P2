"""
Phase 3: Optimization Framework (MPC + Genetic Algorithm)
Use GA to find optimal HVAC schedules that minimize cost and maximize comfort
"""

import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import pickle
import json
from phase2_surrogate_model import SurrogateModel


class BuildingOptimizer:
    """Genetic Algorithm optimizer for building HVAC control"""
    
    def __init__(self, surrogate_model, electricity_prices=None):
        """
        Args:
            surrogate_model: Trained SurrogateModel instance
            electricity_prices: Array of hourly electricity prices ($/kWh)
        """
        self.surrogate_model = surrogate_model
        
        # Default time-of-use pricing ($/kWh)
        if electricity_prices is None:
            # Peak: 6 AM - 9 PM ($0.20/kWh), Off-peak: 9 PM - 6 AM ($0.10/kWh)
            self.electricity_prices = np.array(
                [0.10]*6 + [0.20]*15 + [0.10]*3
            )
        else:
            self.electricity_prices = electricity_prices
        
        # Comfort parameters (PMV range)
        self.comfort_temp_min = 21.0  # °C
        self.comfort_temp_max = 24.0  # °C
        self.pmv_min = -0.5
        self.pmv_max = 0.5
        
        # Optimization parameters
        self.setpoint_min = 19.0
        self.setpoint_max = 26.0
        
        # Weight for comfort vs cost trade-off
        self.comfort_weight = 100  # $/°C violation
        
        # Setup DEAP genetic algorithm
        self._setup_ga()
    
    def _setup_ga(self):
        """Setup DEAP genetic algorithm framework"""
        # Create fitness and individual classes
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Gene: HVAC setpoint temperature (°C)
        self.toolbox.register("attr_setpoint", np.random.uniform, 
                             self.setpoint_min, self.setpoint_max)
        
        # Individual: 24-hour schedule
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_setpoint, n=24)
        
        # Population
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self._mutate_setpoint, 
                             indpb=0.2, sigma=1.0)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self._evaluate_schedule)
    
    def _mutate_setpoint(self, individual, indpb, sigma):
        """Custom mutation: Gaussian mutation with bounds checking"""
        for i in range(len(individual)):
            if np.random.random() < indpb:
                individual[i] += np.random.normal(0, sigma)
                individual[i] = np.clip(individual[i], 
                                       self.setpoint_min, self.setpoint_max)
        return individual,
    
    def _evaluate_schedule(self, individual, weather_data):
        """
        Evaluate fitness of an HVAC schedule
        
        Args:
            individual: 24-hour setpoint schedule
            weather_data: Dict with weather features for 24 hours
        
        Returns:
            Tuple with single fitness value (to minimize)
        """
        # Prepare input features for surrogate model
        n_hours = 24
        X = np.zeros((n_hours, len(self.surrogate_model.input_features)))
        
        for hour in range(n_hours):
            X[hour] = [
                weather_data['outdoor_temp'][hour],
                weather_data['solar_radiation'][hour],
                weather_data['humidity'][hour],
                hour,  # hour_of_day
                weather_data['day_of_week'],
                individual[hour]  # HVAC setpoint
            ]
        
        # Predict energy consumption and indoor temperature
        # For XGBoost, we need to predict hour by hour
        if self.surrogate_model.model_type == 'xgboost':
            energy_pred = []
            temp_pred = []
            for hour in range(n_hours):
                e, t = self.surrogate_model.predict(X[hour:hour+1])
                energy_pred.append(e[0])
                temp_pred.append(t[0])
            energy_pred = np.array(energy_pred)
            temp_pred = np.array(temp_pred)
        else:
            # For LSTM, need to reshape with sequence
            X_seq = X.reshape(1, n_hours, -1)
            energy_pred, temp_pred = self.surrogate_model.predict(X_seq)
        
        # Calculate energy cost
        energy_cost = np.sum(energy_pred * self.electricity_prices)
        
        # Calculate comfort violations
        comfort_violations = 0
        for hour in range(n_hours):
            temp = temp_pred[hour] if isinstance(temp_pred, np.ndarray) else temp_pred
            if temp < self.comfort_temp_min:
                comfort_violations += (self.comfort_temp_min - temp)
            elif temp > self.comfort_temp_max:
                comfort_violations += (temp - self.comfort_temp_max)
        
        # Total objective (minimize)
        total_cost = energy_cost + self.comfort_weight * comfort_violations
        
        return (total_cost,)
    
    def optimize(self, weather_forecast, population_size=50, n_generations=100):
        """
        Run genetic algorithm optimization
        
        Args:
            weather_forecast: Dict with 24-hour weather forecast
            population_size: GA population size
            n_generations: Number of generations
        
        Returns:
            Dict with optimal schedule and metrics
        """
        print("\n" + "="*80)
        print("RUNNING GENETIC ALGORITHM OPTIMIZATION")
        print("="*80)
        print(f"Population size: {population_size}")
        print(f"Generations: {n_generations}")
        
        # Update evaluation function with weather data
        self.toolbox.register("evaluate", lambda ind: 
                            self._evaluate_schedule(ind, weather_forecast))
        
        # Initialize population
        population = self.toolbox.population(n=population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of fame (best individuals)
        hof = tools.HallOfFame(10)
        
        # Run genetic algorithm
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=0.7,  # Crossover probability
            mutpb=0.3,  # Mutation probability
            ngen=n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        # Get best solution
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Best fitness: ${best_fitness:.2f}")
        print(f"\nOptimal 24-hour setpoint schedule:")
        
        for hour in range(24):
            print(f"  Hour {hour:2d}: {best_individual[hour]:.1f}°C")
        
        # Calculate detailed metrics for best schedule
        metrics = self._calculate_detailed_metrics(best_individual, weather_forecast)
        
        return {
            'optimal_schedule': best_individual,
            'fitness': best_fitness,
            'metrics': metrics,
            'logbook': logbook,
            'hall_of_fame': [list(ind) for ind in hof]
        }
    
    def _calculate_detailed_metrics(self, schedule, weather_data):
        """Calculate detailed performance metrics"""
        n_hours = 24
        X = np.zeros((n_hours, len(self.surrogate_model.input_features)))
        
        for hour in range(n_hours):
            X[hour] = [
                weather_data['outdoor_temp'][hour],
                weather_data['solar_radiation'][hour],
                weather_data['humidity'][hour],
                hour,
                weather_data['day_of_week'],
                schedule[hour]
            ]
        
        # Predict
        if self.surrogate_model.model_type == 'xgboost':
            energy_pred = []
            temp_pred = []
            for hour in range(n_hours):
                e, t = self.surrogate_model.predict(X[hour:hour+1])
                energy_pred.append(e[0])
                temp_pred.append(t[0])
            energy_pred = np.array(energy_pred)
            temp_pred = np.array(temp_pred)
        else:
            X_seq = X.reshape(1, n_hours, -1)
            energy_pred, temp_pred = self.surrogate_model.predict(X_seq)
        
        # Calculate metrics
        total_energy = np.sum(energy_pred)
        total_cost = np.sum(energy_pred * self.electricity_prices)
        
        comfort_violations = 0
        violation_hours = 0
        for hour in range(n_hours):
            temp = temp_pred[hour]
            if temp < self.comfort_temp_min or temp > self.comfort_temp_max:
                violation_hours += 1
                if temp < self.comfort_temp_min:
                    comfort_violations += (self.comfort_temp_min - temp)
                else:
                    comfort_violations += (temp - self.comfort_temp_max)
        
        return {
            'total_energy_kwh': float(total_energy),
            'total_cost_dollars': float(total_cost),
            'comfort_violations_hours': int(violation_hours),
            'comfort_violation_magnitude': float(comfort_violations),
            'hourly_energy': energy_pred.tolist() if isinstance(energy_pred, np.ndarray) else energy_pred,
            'hourly_temperature': temp_pred.tolist() if isinstance(temp_pred, np.ndarray) else temp_pred,
            'hourly_setpoint': list(schedule)
        }
    
    def evaluate_baseline(self, weather_forecast, fixed_setpoint=23.0):
        """Evaluate baseline controller (fixed setpoint)"""
        print("\n" + "="*80)
        print("EVALUATING BASELINE CONTROLLER")
        print("="*80)
        print(f"Fixed setpoint: {fixed_setpoint}°C")
        
        baseline_schedule = [fixed_setpoint] * 24
        metrics = self._calculate_detailed_metrics(baseline_schedule, weather_forecast)
        
        print(f"\nBaseline Performance:")
        print(f"  Total Energy: {metrics['total_energy_kwh']:.2f} kWh")
        print(f"  Total Cost: ${metrics['total_cost_dollars']:.2f}")
        print(f"  Comfort Violations: {metrics['comfort_violations_hours']} hours")
        
        return {
            'baseline_schedule': baseline_schedule,
            'metrics': metrics
        }


def generate_table3():
    """Generate Table 3: Objective Function & Optimization Constraints"""
    print("\n" + "="*80)
    print("GENERATING TABLE 3: OPTIMIZATION PARAMETERS")
    print("="*80)
    
    table3_data = {
        'Parameter': [
            'Objective Function',
            'Decision Variable',
            'Comfort Range (Temperature)',
            'PMV (Predicted Mean Vote)',
            'Algorithm',
            'Time Horizon'
        ],
        'Description': [
            'Minimize: J = C_energy + w · D_comfort',
            'HVAC Setpoint (T_set)',
            'Indoor Temperature',
            'Thermal Comfort Index',
            'Genetic Algorithm (GA)',
            'Prediction Window'
        ],
        'Value / Constraint': [
            'Cost vs. Comfort Trade-off',
            '19°C ≤ T_set ≤ 26°C',
            '21°C ≤ T_indoor ≤ 24°C',
            '-0.5 ≤ PMV ≤ +0.5',
            'Population: 50, Generations: 100',
            '24 Hours (Day-ahead)'
        ]
    }
    
    table3 = pd.DataFrame(table3_data)
    
    # Save table
    table3.to_csv('tables/table3_optimization_parameters.csv', index=False)
    
    with open('tables/table3_optimization_parameters.txt', 'w') as f:
        f.write("Table 3: Objective Function & Optimization Constraints\n")
        f.write("="*100 + "\n\n")
        f.write(table3.to_string(index=False))
    
    print("\nTable 3 saved to tables/")
    print("\n" + table3.to_string(index=False))
    
    return table3


def main():
    """Execute Phase 3: Optimization Framework"""
    print("="*80)
    print("PHASE 3: OPTIMIZATION FRAMEWORK (MPC + GENETIC ALGORITHM)")
    print("="*80)
    
    # Generate Table 3
    table3 = generate_table3()
    
    # Load surrogate model
    print("\nLoading surrogate model...")
    surrogate_model = SurrogateModel(model_type='xgboost')
    surrogate_model.load('results/surrogate_model_xgboost')
    
    # Create sample weather forecast (typical summer day)
    print("\nCreating weather forecast for optimization...")
    weather_forecast = {
        'outdoor_temp': [
            20, 19, 18, 18, 19, 21,  # Night/Early morning
            23, 26, 28, 30, 32, 33,  # Morning/Noon
            34, 33, 32, 30, 28, 26,  # Afternoon
            24, 23, 22, 21, 20, 20   # Evening/Night
        ],
        'solar_radiation': [
            0, 0, 0, 0, 0, 50,       # Night/Dawn
            200, 400, 600, 750, 800, 850,  # Morning/Noon
            800, 750, 600, 400, 200, 50,   # Afternoon
            0, 0, 0, 0, 0, 0         # Evening/Night
        ],
        'humidity': [
            60, 65, 70, 70, 65, 60,  # Night
            55, 50, 45, 40, 38, 35,  # Day
            35, 38, 40, 45, 50, 55,  # Afternoon
            58, 60, 62, 63, 62, 61   # Evening
        ],
        'day_of_week': 2  # Wednesday
    }
    
    # Initialize optimizer
    optimizer = BuildingOptimizer(surrogate_model)
    
    # Evaluate baseline
    baseline_results = optimizer.evaluate_baseline(weather_forecast, fixed_setpoint=23.0)
    
    # Run optimization
    optimal_results = optimizer.optimize(
        weather_forecast, 
        population_size=50, 
        n_generations=100
    )
    
    # Save results
    results = {
        'baseline': baseline_results,
        'optimal': optimal_results,
        'weather_forecast': weather_forecast
    }
    
    with open('results/optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPARISON: BASELINE vs. OPTIMAL")
    print("="*80)
    
    baseline = baseline_results['metrics']
    optimal = optimal_results['metrics']
    
    energy_savings = (baseline['total_energy_kwh'] - optimal['total_energy_kwh']) / baseline['total_energy_kwh'] * 100
    cost_savings = (baseline['total_cost_dollars'] - optimal['total_cost_dollars']) / baseline['total_cost_dollars'] * 100
    comfort_improvement = (baseline['comfort_violations_hours'] - optimal['comfort_violations_hours']) / max(baseline['comfort_violations_hours'], 1) * 100
    
    print(f"\nEnergy Consumption:")
    print(f"  Baseline: {baseline['total_energy_kwh']:.2f} kWh")
    print(f"  Optimal:  {optimal['total_energy_kwh']:.2f} kWh")
    print(f"  Savings:  {energy_savings:.1f}%")
    
    print(f"\nEnergy Cost:")
    print(f"  Baseline: ${baseline['total_cost_dollars']:.2f}")
    print(f"  Optimal:  ${optimal['total_cost_dollars']:.2f}")
    print(f"  Savings:  {cost_savings:.1f}%")
    
    print(f"\nComfort Violations:")
    print(f"  Baseline: {baseline['comfort_violations_hours']} hours")
    print(f"  Optimal:  {optimal['comfort_violations_hours']} hours")
    print(f"  Improvement: {comfort_improvement:.1f}%")
    
    print("\n✓ Phase 3 completed successfully!")


if __name__ == "__main__":
    main()
