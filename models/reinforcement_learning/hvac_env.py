"""
HVAC Control Environment for Reinforcement Learning
Gymnasium-compatible environment for building energy optimization
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class HVACControlEnv(gym.Env):
    """
    HVAC Control Environment for building energy optimization.
    
    State: [indoor_temp, outdoor_temp, humidity, hour, day_of_week, current_energy]
    Action: [HVAC_setpoint, HVAC_mode] (continuous)
    Reward: -energy_cost - comfort_penalty
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, data: np.ndarray, energy_data: np.ndarray, 
                 comfort_model=None, max_steps=1000):
        super().__init__()
        
        self.data = data
        self.energy_data = energy_data
        self.comfort_model = comfort_model
        self.max_steps = max_steps
        self.current_step = 0
        
        # State space: [indoor_temp, outdoor_temp, humidity, hour, day_of_week, current_energy]
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([40, 40, 100, 23, 6, 1000], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: [HVAC_setpoint (16-26°C), HVAC_mode (0=off, 1=cooling, 2=heating)]
        self.action_space = spaces.Box(
            low=np.array([16.0, 0.0], dtype=np.float32),
            high=np.array([26.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def _get_observation(self):
        """Get current observation from data."""
        if self.current_step >= len(self.data):
            return np.zeros(6, dtype=np.float32)
        
        # Extract features from current time step
        # Assuming data structure: [temp_features, humidity_features, time_features, ...]
        obs = np.zeros(6, dtype=np.float32)
        
        # Indoor temperature (average of indoor temps)
        temp_cols = [i for i in range(min(9, self.data.shape[1])) if i < len(self.data[0])]
        if temp_cols:
            obs[0] = np.mean([self.data[self.current_step, i] for i in temp_cols[:9]])
        
        # Outdoor temperature (if available)
        if self.data.shape[1] > 9:
            obs[1] = self.data[self.current_step, 9]  # Assuming T_out is at index 9
        
        # Humidity (average)
        rh_start = 10
        if self.data.shape[1] > rh_start:
            rh_cols = [i for i in range(rh_start, min(rh_start+9, self.data.shape[1]))]
            if rh_cols:
                obs[2] = np.mean([self.data[self.current_step, i] for i in rh_cols])
        
        # Time features (if available in data)
        if self.data.shape[1] > 20:
            obs[3] = self.data[self.current_step, -2] if self.data.shape[1] > 20 else 12  # hour
            obs[4] = self.data[self.current_step, -1] if self.data.shape[1] > 21 else 0  # day_of_week
        
        # Current energy consumption
        if self.current_step < len(self.energy_data):
            obs[5] = self.energy_data[self.current_step]
        
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_setpoint = 22.0  # Initial setpoint
        self.current_mode = 1.0  # Initial mode (cooling)
        
        observation = self._get_observation()
        info = {"step": self.current_step}
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """Execute one step in the environment."""
        # Parse action
        setpoint = np.clip(action[0], 16.0, 26.0)
        mode = int(np.clip(action[1], 0, 2))
        
        self.current_setpoint = setpoint
        self.current_mode = mode
        
        # Get current state
        obs = self._get_observation()
        indoor_temp = obs[0]
        outdoor_temp = obs[1]
        
        # Calculate energy consumption based on HVAC action
        # Energy = base_load + HVAC_load
        base_energy = obs[5] if obs[5] > 0 else 50.0
        
        # HVAC energy consumption
        if mode == 0:  # Off
            hvac_energy = 0.0
        elif mode == 1:  # Cooling
            temp_diff = max(0, indoor_temp - setpoint)
            hvac_energy = temp_diff * 10.0  # Simplified model
        else:  # Heating
            temp_diff = max(0, setpoint - indoor_temp)
            hvac_energy = temp_diff * 12.0  # Heating typically more energy
        
        total_energy = base_energy + hvac_energy
        
        # Calculate comfort penalty
        comfort_penalty = 0.0
        if self.comfort_model:
            pmv = self.comfort_model.calculate_pmv(indoor_temp, obs[2])
            # Penalty for discomfort (PMV away from 0)
            comfort_penalty = abs(pmv) * 5.0
        else:
            # Simple comfort model
            temp_diff_from_setpoint = abs(indoor_temp - setpoint)
            comfort_penalty = temp_diff_from_setpoint * 2.0
        
        # Reward: negative of (energy cost + comfort penalty)
        reward = -(total_energy * 0.01 + comfort_penalty)
        
        # Update step
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "energy": total_energy,
            "comfort_penalty": comfort_penalty,
            "setpoint": setpoint,
            "mode": mode
        }
        
        # Get next observation
        next_obs = self._get_observation()
        
        return next_obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (optional)."""
        pass
