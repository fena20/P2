"""
Physics-Informed Deep Reinforcement Learning Environment for Residential Building Energy Management
Based on AMPds2 dataset with 1-minute resolution

State Space: [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
Action Space: Discrete(2) -> [OFF=0, ON=1] for Heat Pump
Physics: 1st-order RC thermal model with cycling penalty
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def load_ampds2_data(data_path: Optional[str] = None, n_samples: int = 10000) -> pd.DataFrame:
    """
    Load AMPds2 dataset or generate synthetic data matching the structure.
    
    Expected columns: WHE (Water Heater Energy), HPE (Heat Pump Energy), 
                     FRE (Furnace Energy), Outdoor_Temp
    
    Args:
        data_path: Path to AMPds2 CSV file (if None, generates synthetic data)
        n_samples: Number of samples to generate if using synthetic data
        
    Returns:
        DataFrame with columns: WHE, HPE, FRE, Outdoor_Temp, timestamp
    """
    if data_path is not None:
        try:
            df = pd.read_csv(data_path)
            # Ensure required columns exist
            required_cols = ['WHE', 'HPE', 'FRE', 'Outdoor_Temp']
            if all(col in df.columns for col in required_cols):
                return df[required_cols + ['timestamp'] if 'timestamp' in df.columns else required_cols]
        except Exception as e:
            print(f"Warning: Could not load data from {data_path}. Generating synthetic data. Error: {e}")
    
    # Generate synthetic AMPds2-like data
    print("Generating synthetic AMPds2-like data...")
    np.random.seed(42)
    
    # Create time series with 1-minute resolution
    timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='1min')
    
    # Generate outdoor temperature (seasonal pattern with daily variation)
    day_of_year = np.array([(t - timestamps[0]).days for t in timestamps])
    hour_of_day = np.array([t.hour + t.minute/60.0 for t in timestamps])
    
    # Base temperature: 15°C with seasonal variation (-5 to 35°C range)
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
    daily_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    outdoor_temp = seasonal_temp + daily_variation + np.random.normal(0, 2, n_samples)
    outdoor_temp = np.clip(outdoor_temp, -5, 35)
    
    # Generate energy consumption (correlated with outdoor temp)
    # Heat Pump Energy (HPE): higher when outdoor temp is low
    hpe_base = np.maximum(0, 20 - outdoor_temp) * 0.5  # kW when ON
    hpe = hpe_base * np.random.uniform(0.8, 1.2, n_samples)
    
    # Water Heater Energy (WHE): relatively constant
    whe = np.random.uniform(0.1, 0.3, n_samples)
    
    # Furnace Energy (FRE): used when very cold
    fre = np.maximum(0, 5 - outdoor_temp) * 0.3
    fre = fre * np.random.uniform(0.5, 1.5, n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Outdoor_Temp': outdoor_temp,
        'HPE': hpe,
        'WHE': whe,
        'FRE': fre
    })
    
    return df


class SmartHomeEnv(gym.Env):
    """
    Physics-Informed Gymnasium Environment for Smart Home Energy Management.
    
    Implements a 1st-order RC thermal model with cycling penalty to prevent
    short-cycling of HVAC equipment (minimum 15-minute run/off cycles).
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        max_steps: int = 1440,  # 24 hours at 1-min resolution
        comfort_setpoint: float = 22.0,  # °C
        comfort_tolerance: float = 2.0,  # °C
        min_cycle_time: int = 15,  # minutes (critical for hardware protection)
        R: float = 0.05,  # Thermal resistance (K/kW)
        C: float = 0.5,  # Thermal capacitance (kWh/K)
        hvac_power: float = 3.0,  # kW when ON
        dt: float = 1.0/60.0,  # 1 minute in hours
        w1: float = 1.0,  # Cost weight
        w2: float = 10.0,  # Discomfort weight
        w3: float = 5.0,  # Cycling penalty weight
        price_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the Smart Home Environment.
        
        Args:
            data_path: Path to AMPds2 data CSV
            max_steps: Maximum steps per episode
            comfort_setpoint: Desired indoor temperature (°C)
            comfort_tolerance: Acceptable temperature deviation (°C)
            min_cycle_time: Minimum time between state changes (minutes)
            R: Thermal resistance (K/kW)
            C: Thermal capacitance (kWh/K)
            hvac_power: HVAC power consumption when ON (kW)
            dt: Time step in hours (1 minute = 1/60 hours)
            w1, w2, w3: Reward function weights
            price_schedule: Time-of-use electricity price schedule (24 values)
            seed: Random seed
        """
        super().__init__()
        
        self.max_steps = max_steps
        self.comfort_setpoint = comfort_setpoint
        self.comfort_tolerance = comfort_tolerance
        self.min_cycle_time = min_cycle_time
        self.R = R
        self.C = C
        self.hvac_power = hvac_power
        self.dt = dt
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        
        # Load data
        self.data = load_ampds2_data(data_path)
        self.n_data_points = len(self.data)
        
        # Time-of-use pricing (peak: 17:00-20:00)
        if price_schedule is None:
            self.price_schedule = np.ones(24) * 0.12  # Base price $0.12/kWh
            self.price_schedule[17:20] = 0.25  # Peak price $0.25/kWh
        else:
            self.price_schedule = price_schedule
        
        # State space: [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
        self.observation_space = spaces.Box(
            low=np.array([15.0, -5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 35.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: Discrete(2) -> [OFF=0, ON=1]
        self.action_space = spaces.Discrete(2)
        
        # Initialize state variables
        self.current_step = 0
        self.indoor_temp = comfort_setpoint
        self.outdoor_temp = 20.0
        self.solar_rad = 0.0
        self.price = 0.12
        self.last_action = 0
        self.time_index = 0.0
        
        # Cycling penalty tracking
        self.last_action_change_step = 0
        self.action_history = []  # Track actions for cycling penalty
        
        # Episode statistics
        self.episode_cost = 0.0
        self.episode_discomfort = 0.0
        self.episode_cycles = 0
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Random start point in dataset
        start_idx = self.np_random.randint(0, max(1, self.n_data_points - self.max_steps))
        
        self.current_step = 0
        self.indoor_temp = self.comfort_setpoint + self.np_random.uniform(-1, 1)
        self.outdoor_temp = float(self.data.iloc[start_idx]['Outdoor_Temp'])
        self.solar_rad = self._get_solar_radiation(start_idx)
        self.price = self._get_price(start_idx)
        self.last_action = 0
        self.time_index = (start_idx % 1440) / 1440.0  # Normalize to [0, 1]
        
        # Reset cycling tracking
        self.last_action_change_step = 0
        self.action_history = [0]
        
        # Reset statistics
        self.episode_cost = 0.0
        self.episode_discomfort = 0.0
        self.episode_cycles = 0
        
        observation = self._get_observation()
        info = {
            'indoor_temp': self.indoor_temp,
            'outdoor_temp': self.outdoor_temp,
            'cost': 0.0,
            'discomfort': 0.0
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (OFF) or 1 (ON)
            
        Returns:
            observation: Next state
            reward: Reward value
            terminated: Episode termination flag
            truncated: Episode truncation flag
            info: Additional information
        """
        # Get current data point
        data_idx = (int(self.time_index * 1440) + self.current_step) % self.n_data_points
        row = self.data.iloc[data_idx]
        
        # Update outdoor conditions
        self.outdoor_temp = float(row['Outdoor_Temp'])
        self.solar_rad = self._get_solar_radiation(data_idx)
        self.price = self._get_price(data_idx)
        
        # Calculate HVAC heat input
        if action == 1:  # ON
            Q_hvac = self.hvac_power  # kW
        else:  # OFF
            Q_hvac = 0.0
        
        # Solar heat gain (simplified: 0.2 kW per unit solar radiation)
        Q_solar = 0.2 * self.solar_rad
        
        # Physics-Informed Thermal Model (1st-order RC)
        # T_in^{t+1} = T_in^t + dt * [(T_out - T_in)/R + (Q_HVAC + Q_Solar)/C]
        temp_diff = (self.outdoor_temp - self.indoor_temp) / self.R
        heat_input = (Q_hvac + Q_solar) / self.C
        self.indoor_temp += self.dt * (temp_diff + heat_input)
        
        # Add small noise for realism
        self.indoor_temp += self.np_random.normal(0, 0.05)
        self.indoor_temp = np.clip(self.indoor_temp, 15.0, 30.0)
        
        # Calculate reward components
        # 1. Energy Cost
        energy_cost = Q_hvac * self.price * self.dt  # $/step
        self.episode_cost += energy_cost
        
        # 2. Discomfort Penalty (degree-hours outside comfort zone)
        temp_deviation = abs(self.indoor_temp - self.comfort_setpoint)
        if temp_deviation > self.comfort_tolerance:
            discomfort = (temp_deviation - self.comfort_tolerance) * self.dt
        else:
            discomfort = 0.0
        self.episode_discomfort += discomfort
        
        # 3. Cycling Penalty (CRITICAL: Prevent short-cycling)
        cycling_penalty = 0.0
        if len(self.action_history) > 0 and action != self.action_history[-1]:
            # Action changed - check if minimum cycle time has passed
            steps_since_change = self.current_step - self.last_action_change_step
            if steps_since_change < self.min_cycle_time:
                # Penalty increases exponentially with violation severity
                violation_ratio = steps_since_change / self.min_cycle_time
                cycling_penalty = self.w3 * (1.0 - violation_ratio) ** 2
                self.episode_cycles += 1
        
        # Update action history
        if len(self.action_history) == 0 or action != self.action_history[-1]:
            self.last_action_change_step = self.current_step
        self.action_history.append(action)
        
        # Total reward (negative because we minimize cost/discomfort)
        reward = -(self.w1 * energy_cost + self.w2 * discomfort + cycling_penalty)
        
        # Update state
        self.last_action = float(action)
        self.time_index = ((int(self.time_index * 1440) + 1) % 1440) / 1440.0
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Prepare info
        info = {
            'indoor_temp': self.indoor_temp,
            'outdoor_temp': self.outdoor_temp,
            'action': action,
            'cost': energy_cost,
            'discomfort': discomfort,
            'cycling_penalty': cycling_penalty,
            'total_cost': self.episode_cost,
            'total_discomfort': self.episode_discomfort,
            'total_cycles': self.episode_cycles,
            'hvac_power': Q_hvac
        }
        
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        return np.array([
            self.indoor_temp / 30.0,  # Normalized [0, 1] from [15, 30]
            (self.outdoor_temp + 5.0) / 40.0,  # Normalized [0, 1] from [-5, 35]
            self.solar_rad,  # Already normalized [0, 1]
            self.price / 0.25,  # Normalized [0, 1] from [0, 0.25]
            self.last_action,  # Already [0, 1]
            self.time_index  # Already [0, 1]
        ], dtype=np.float32)
    
    def _get_solar_radiation(self, data_idx: int) -> float:
        """Get solar radiation (normalized 0-1) based on time of day."""
        hour = (data_idx % 1440) / 60.0  # Hour of day
        # Simple model: peak at noon (12:00), zero at night
        if 6 <= hour <= 18:
            solar = np.sin(np.pi * (hour - 6) / 12)
        else:
            solar = 0.0
        return float(np.clip(solar, 0, 1))
    
    def _get_price(self, data_idx: int) -> float:
        """Get electricity price based on time of day."""
        hour = int((data_idx % 1440) / 60)
        return float(self.price_schedule[hour % 24])
    
    def render(self, mode: str = "human"):
        """Render the environment (optional)."""
        if mode == "human":
            print(f"Step: {self.current_step}, Indoor Temp: {self.indoor_temp:.2f}°C, "
                  f"Outdoor Temp: {self.outdoor_temp:.2f}°C, Action: {self.last_action}, "
                  f"Price: ${self.price:.3f}/kWh")
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """Get episode statistics."""
        return {
            'total_cost': self.episode_cost,
            'total_discomfort': self.episode_discomfort,
            'total_cycles': self.episode_cycles,
            'avg_indoor_temp': self.indoor_temp,
            'steps': self.current_step
        }
