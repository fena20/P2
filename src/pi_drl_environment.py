"""
Physics-Informed Deep Reinforcement Learning Environment for Smart Home Energy Management
Designed for Applied Energy Journal Submission

This module implements a custom Gymnasium environment with:
1. Physics-based thermal dynamics (1st-order RC model)
2. Hardware-aware reward function with cycling penalty
3. AMPds2 dataset integration (1-minute resolution)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional
import os
import warnings
warnings.filterwarnings('ignore')


def load_ampds2_real_data(
    data_dir: str = "./data",
    use_processed: bool = True
) -> pd.DataFrame:
    """
    Load REAL AMPds2 dataset from CSV files
    
    Args:
        data_dir: Directory containing AMPds2 data
        use_processed: If True, use pre-processed file; else process from raw
    
    Returns:
        DataFrame with required columns
    """
    processed_file = os.path.join(data_dir, "ampds2_processed.csv")
    
    if use_processed and os.path.exists(processed_file):
        print(f"Loading pre-processed AMPds2 data from {processed_file}...")
        df = pd.read_csv(processed_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✓ Loaded {len(df):,} samples from REAL data")
        return df
    else:
        # Import the real data loader
        from load_real_ampds2 import load_real_ampds2_data
        print("Processing real AMPds2 data from CSV files...")
        df = load_real_ampds2_data(
            data_dir=data_dir,
            start_date="2012-04-01",
            end_date="2012-04-30",  # 1 month
            max_samples=None
        )
        # Save for future use
        df.to_csv(processed_file, index=False)
        return df


def load_ampds2_mock_data(num_samples: int = 525600) -> pd.DataFrame:
    """
    Mock function to simulate AMPds2 dataset structure.
    In production, replace with actual data loading from:
    https://github.com/Fateme9977/P3/tree/main/data
    
    Args:
        num_samples: Number of 1-minute samples (default: 1 year = 525600)
    
    Returns:
        DataFrame with columns: WHE, HPE, FRE, Outdoor_Temp, Solar_Rad, Price
    """
    np.random.seed(42)
    
    # Generate time index (1-minute resolution)
    dates = pd.date_range(start='2024-01-01', periods=num_samples, freq='1min')
    
    # Simulate realistic patterns
    hour_of_day = dates.hour.values
    day_of_year = dates.dayofyear.values
    
    # Outdoor temperature with seasonal and diurnal variation
    outdoor_temp = (
        15 +  # Base temperature
        10 * np.sin(2 * np.pi * day_of_year / 365) +  # Seasonal
        5 * np.sin(2 * np.pi * hour_of_day / 24) +  # Diurnal
        np.random.normal(0, 2, num_samples)  # Noise
    )
    
    # Solar radiation (zero at night)
    solar_rad = np.maximum(
        0,
        800 * np.sin(np.pi * (hour_of_day - 6) / 12) * 
        (1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365))
    ) * (hour_of_day >= 6) * (hour_of_day <= 18)
    
    # Time-of-use electricity pricing ($/kWh)
    price = np.where(
        (hour_of_day >= 17) & (hour_of_day <= 20),
        0.25,  # Peak price
        np.where(
            ((hour_of_day >= 7) & (hour_of_day < 17)) | ((hour_of_day > 20) & (hour_of_day <= 22)),
            0.15,  # Mid-peak
            0.08   # Off-peak
        )
    )
    
    # Simulated appliance energy consumption
    WHE = 100 + 50 * np.random.rand(num_samples)  # Water heater (W)
    FRE = 150 + 30 * np.random.rand(num_samples)  # Fridge (W)
    HPE = np.zeros(num_samples)  # Heat pump (controlled by agent)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'WHE': WHE,
        'HPE': HPE,
        'FRE': FRE,
        'Outdoor_Temp': outdoor_temp,
        'Solar_Rad': solar_rad,
        'Price': price,
        'Hour': hour_of_day,
        'DayOfYear': day_of_year
    })
    
    return df


class SmartHomeEnv(gym.Env):
    """
    Physics-Informed Smart Home Energy Management Environment
    
    State Space: [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
    Action Space: Discrete(2) -> [OFF=0, ON=1] for Heat Pump
    
    Physics Model: 1st-order RC thermal dynamics
    T_in^{t+1} = T_in^t + Δt * [(T_out - T_in)/R + (Q_HVAC + Q_Solar)/C]
    
    Reward Function (Multi-Objective):
    R = -(w1*Cost + w2*Discomfort + w3*Cycling_Penalty)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        R: float = 10.0,  # Thermal resistance (K/kW)
        C: float = 20.0,  # Thermal capacitance (kWh/K)
        hvac_power: float = 3.0,  # Heat pump power (kW)
        dt: float = 1/60,  # Time step (hours) = 1 minute
        comfort_range: Tuple[float, float] = (20.0, 24.0),
        w_cost: float = 1.0,
        w_comfort: float = 10.0,
        w_cycling: float = 5.0,
        min_cycle_time: int = 15,  # Minimum 15 minutes between switches
        episode_length: int = 1440,  # 24 hours in minutes
    ):
        super().__init__()
        
        # Load or generate data
        if data is None:
            self.data = load_ampds2_mock_data()
        else:
            self.data = data
        
        # Physics parameters
        self.R = R  # Thermal resistance
        self.C = C  # Thermal capacitance
        self.hvac_power = hvac_power
        self.dt = dt
        
        # Comfort and cost parameters
        self.T_min, self.T_max = comfort_range
        self.T_setpoint = (self.T_min + self.T_max) / 2
        self.w_cost = w_cost
        self.w_comfort = w_comfort
        self.w_cycling = w_cycling
        self.min_cycle_time = min_cycle_time
        
        # Episode parameters
        self.episode_length = episode_length
        self.max_episodes = len(self.data) // episode_length
        
        # Gym spaces
        # State: [T_in, T_out, Solar, Price, Last_Action, Time_Index_Normalized]
        self.observation_space = spaces.Box(
            low=np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([35.0, 40.0, 1000.0, 0.5, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action: [OFF=0, ON=1]
        self.action_space = spaces.Discrete(2)
        
        # State variables
        self.current_step = 0
        self.episode_start_idx = 0
        self.indoor_temp = self.T_setpoint
        self.last_action = 0
        self.time_since_last_switch = self.min_cycle_time + 1
        self.total_switches = 0
        
        # Tracking for analysis
        self.episode_history = []
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Random episode start
        if seed is not None:
            np.random.seed(seed)
        self.episode_start_idx = np.random.randint(0, self.max_episodes) * self.episode_length
        
        # Reset state variables
        self.current_step = 0
        self.indoor_temp = self.T_setpoint + np.random.uniform(-1, 1)
        self.last_action = 0
        self.time_since_last_switch = self.min_cycle_time + 1
        self.total_switches = 0
        
        # Clear history
        self.episode_history = []
        
        state = self._get_observation()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step in the environment
        
        Args:
            action: 0 (OFF) or 1 (ON) for heat pump
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current data
        idx = self.episode_start_idx + self.current_step
        current_data = self.data.iloc[idx]
        
        T_out = current_data['Outdoor_Temp']
        Q_solar = current_data['Solar_Rad'] * 0.001  # Convert W to kW, with window area factor
        price = current_data['Price']
        
        # Track switching
        if action != self.last_action:
            self.total_switches += 1
            self.time_since_last_switch = 0
        else:
            self.time_since_last_switch += 1
        
        # Physics-based temperature update (RC thermal model)
        Q_hvac = action * self.hvac_power  # kW
        
        # dT/dt = (T_out - T_in)/R + (Q_HVAC + Q_Solar)/C
        dT_dt = (T_out - self.indoor_temp) / self.R + (Q_hvac + Q_solar) / self.C
        
        # Euler integration
        self.indoor_temp = self.indoor_temp + self.dt * dT_dt
        
        # Calculate reward components
        # 1. Energy cost
        energy_consumption = action * self.hvac_power * self.dt  # kWh
        cost = energy_consumption * price
        
        # 2. Comfort penalty (degree-hours outside comfort zone)
        if self.indoor_temp < self.T_min:
            discomfort = (self.T_min - self.indoor_temp) ** 2
        elif self.indoor_temp > self.T_max:
            discomfort = (self.indoor_temp - self.T_max) ** 2
        else:
            discomfort = 0.0
        
        # 3. Cycling penalty (CRITICAL for hardware longevity)
        # Penalize switching if it happens too frequently
        if action != self.last_action and self.time_since_last_switch < self.min_cycle_time:
            cycling_penalty = 1.0 * (self.min_cycle_time - self.time_since_last_switch) / self.min_cycle_time
        else:
            cycling_penalty = 0.0
        
        # Multi-objective reward
        reward = -(
            self.w_cost * cost +
            self.w_comfort * discomfort +
            self.w_cycling * cycling_penalty
        )
        
        # Store history
        self.episode_history.append({
            'step': self.current_step,
            'indoor_temp': self.indoor_temp,
            'outdoor_temp': T_out,
            'action': action,
            'cost': cost,
            'discomfort': discomfort,
            'cycling_penalty': cycling_penalty,
            'reward': reward,
            'price': price,
            'hour': current_data['Hour']
        })
        
        # Update state
        self.last_action = action
        self.current_step += 1
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        idx = self.episode_start_idx + self.current_step
        if idx >= len(self.data):
            idx = len(self.data) - 1
            
        current_data = self.data.iloc[idx]
        
        # Normalize time index to [0, 1]
        time_normalized = (current_data['Hour'] + current_data.name % 60 / 60) / 24.0
        
        obs = np.array([
            self.indoor_temp,
            current_data['Outdoor_Temp'],
            current_data['Solar_Rad'],
            current_data['Price'],
            float(self.last_action),
            time_normalized
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict:
        """Additional information for logging"""
        return {
            'indoor_temp': self.indoor_temp,
            'total_switches': self.total_switches,
            'episode_step': self.current_step
        }
    
    def get_episode_history(self) -> pd.DataFrame:
        """Return episode history as DataFrame"""
        return pd.DataFrame(self.episode_history)


# Baseline thermostat controller for comparison
class BaselineThermostat:
    """
    Simple bang-bang thermostat controller
    Used as baseline for comparison (exhibits short-cycling)
    """
    
    def __init__(
        self,
        T_setpoint: float = 22.0,
        deadband: float = 0.5
    ):
        self.T_setpoint = T_setpoint
        self.deadband = deadband
        
    def get_action(self, T_indoor: float) -> int:
        """Simple bang-bang control"""
        if T_indoor < self.T_setpoint - self.deadband:
            return 1  # Turn ON
        elif T_indoor > self.T_setpoint + self.deadband:
            return 0  # Turn OFF
        else:
            return 0  # Stay OFF (default)
