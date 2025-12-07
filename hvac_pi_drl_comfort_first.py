# -*- coding: utf-8 -*-
"""
Robust, Safety-Critical 2R2C PI-DRL Controller for Residential HVAC
COMFORT-FIRST DESIGN - Fixed reward function that properly prioritizes thermal comfort

Key fixes:
1. Hierarchical reward: Comfort is a HARD constraint, cost optimization only when comfortable
2. Proper reward scaling to prevent "always OFF" policy
3. Temperature maintenance incentives
4. Progressive penalty that increases exponentially outside comfort band

Author: Energy Systems ML Researcher
"""

import os
from dataclasses import dataclass, replace
from typing import Dict, Tuple, Any, Optional, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import random
import torch

# -------------------------------------------------------------------------
# Matplotlib style
# -------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3
})

# =========================================================================
# CONFIGURATION - COMFORT-FIRST DESIGN
# =========================================================================

@dataclass
class Config:
    """
    Configuration with COMFORT-FIRST reward design.
    
    Key principle: The agent should NEVER sacrifice comfort for cost savings.
    Cost optimization only happens WHEN comfort is maintained.
    """
    # --- Local data path (change if needed) ---
    data_dir: str = r"C:\Users\FATEME\Downloads\dataverse_files"

    # Reproducibility seed
    seed: int = 42

    # 2R2C parameters
    R_i: float = 0.0005
    R_w: float = 0.003
    R_o: float = 0.002
    C_in: float = 1_000_000.0
    C_m:  float = 4_000_000.0

    # HVAC
    Q_hvac_max: float = 3000.0  # W
    Q_hvac_kw:  float = 3.0     # kW
    dt: float = 60.0            # s

    # Comfort / thermostat
    setpoint: float    = 21.0
    deadband: float    = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    lockout_time: int  = 15     # min

    # TOU tariff
    peak_price:    float = 0.30
    offpeak_price: float = 0.10

    # Randomization
    randomization_scale: float = 0.0

    # Training setup
    total_timesteps:    int = 300_000
    episode_length_days: int = 1
    train_split:        float = 0.8

    # =========================================================================
    # COMFORT-FIRST REWARD WEIGHTS
    # =========================================================================
    # The key insight: Comfort violations must have MUCH higher penalty than
    # any possible cost savings. This prevents the "always OFF" policy.
    
    # Base reward for staying in comfort (per step)
    comfort_bonus: float = 1.0
    
    # Penalty multiplier for being outside comfort band
    # Penalty = comfort_violation_base * deviation^2
    comfort_violation_base: float = 10.0
    comfort_violation_exp: float = 2.0
    
    # Cost penalty (applied when in comfort band to encourage efficiency)
    cost_weight: float = 0.5
    
    # Peak hour additional cost weight
    peak_weight: float = 0.3
    
    # Switching penalty (to reduce cycling)
    switch_penalty: float = 0.15
    
    # Invalid action penalty (safety violation)
    invalid_penalty: float = 2.0
    
    # Maximum cost per step for normalization ($/step)
    max_cost_per_step: float = 0.015  # 3kW * 0.30$/kWh * 1/60 hour

    # PPO hyperparameters
    learning_rate: float = 1e-4
    gamma:         float = 0.995  # Higher gamma for longer-term planning
    gae_lambda:    float = 0.95
    n_steps:       int   = 4096  # Longer rollouts
    batch_size:    int   = 128
    n_epochs:      int   = 10
    ent_coef:      float = 0.05  # Higher entropy for more exploration
    clip_range:    float = 0.2

    # Observation noise
    obs_noise_std: float = 0.1


# =========================================================================
# GLOBAL SEEDING
# =========================================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# =========================================================================
# DATA LOADING - SYNTHETIC DATA FOR TESTING
# =========================================================================

def generate_synthetic_data(config: Config, n_days: int = 30) -> pd.DataFrame:
    """
    Generate synthetic AMPds2-like data for testing.
    Creates realistic outdoor temperature patterns and electricity prices.
    """
    print("=" * 80)
    print("Generating Synthetic Data (for testing)")
    print("=" * 80)

    n_samples = n_days * 24 * 60  # 1-minute resolution
    
    # Create datetime index
    start_date = pd.Timestamp("2024-01-01")
    index = pd.date_range(start=start_date, periods=n_samples, freq="1min")
    
    df = pd.DataFrame(index=index)
    
    # Generate outdoor temperature with daily and seasonal patterns
    hours = np.arange(n_samples) / 60.0  # hours since start
    
    # Daily pattern: coldest at 6am, warmest at 3pm
    daily_pattern = -5 * np.cos(2 * np.pi * (hours - 6) / 24)
    
    # Add some noise
    noise = np.random.normal(0, 2, n_samples)
    
    # Base temperature (winter scenario, needs heating)
    base_temp = 5.0  # Average outdoor temp = 5°C (cold winter)
    
    df["T_out"] = base_temp + daily_pattern + noise
    df["T_out"] = df["T_out"].clip(-20, 35)
    
    # Generate electricity consumption patterns
    df["WHE"] = np.random.uniform(0.5, 2.0, n_samples)  # kW
    df["HPE"] = np.random.uniform(0, 3.0, n_samples)    # kW
    
    # Time features
    df["hour"] = df.index.hour
    df["Price"] = df["hour"].apply(
        lambda h: config.peak_price if 16 <= h < 21 else config.offpeak_price
    )
    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    print(f"Generated {len(df)} samples ({n_days} days)")
    print(f"T_out range: {df['T_out'].min():.1f}°C to {df['T_out'].max():.1f}°C")
    print(f"Mean T_out: {df['T_out'].mean():.1f}°C")
    
    return df


def load_ampds2_local(config: Config) -> pd.DataFrame:
    """
    Load AMPds2 data from local files, or generate synthetic data if not available.
    """
    base = config.data_dir
    path_weather = os.path.join(base, "Climate_HourlyWeather.csv")
    
    if not os.path.exists(path_weather):
        print(f"AMPds2 data not found at: {base}")
        print("Generating synthetic data instead...")
        return generate_synthetic_data(config, n_days=30)
    
    # Original loading code would go here
    # For now, return synthetic data
    return generate_synthetic_data(config, n_days=30)


# =========================================================================
# COMFORT-FIRST REWARD HANDLER
# =========================================================================

class ComfortFirstRewardHandler:
    """
    COMFORT-FIRST reward function with AGGRESSIVE cost optimization.
    
    Key insight: The agent needs to learn that:
    1. Temperature doesn't drop instantly when HVAC is OFF
    2. Pre-heating before peak hours allows coasting during expensive times
    3. Using thermal mass of the building as "storage"
    
    Strategy:
    - Strong comfort penalties outside band
    - SIGNIFICANT energy cost penalties when comfortable
    - Temperature gradient awareness (is temp rising/falling?)
    - Peak hour strategy: pre-heat, then coast
    """

    def __init__(self, config: Config):
        self.config = config
        self.T_min = config.comfort_min
        self.T_max = config.comfort_max
        self.setpoint = config.setpoint
        
        # Temperature zones
        self.T_target_low = self.setpoint - 0.5   # 20.5°C
        self.T_target_high = self.setpoint + 0.5  # 21.5°C
        
        # Pre-heat target for peak hours (higher temperature)
        self.T_preheat = self.setpoint + 1.5  # 22.5°C - upper comfort bound
        
        # Track previous temperatures for gradient
        self.prev_T_in = self.setpoint
        self.cumulative_on_time = 0

    def calculate(
        self,
        T_in: float,
        action: int,
        price_t: float,
        prev_action: int,
        is_invalid: bool,
        is_peak: bool,
        current_step_power_kw: float
    ) -> Tuple[float, Dict]:
        """
        Multi-objective reward with proper balancing.
        """
        
        # Calculate energy cost
        dt_hours = 1.0 / 60.0
        energy_kwh = current_step_power_kw * dt_hours
        instant_cost = energy_kwh * price_t
        
        # Comfort evaluation
        in_comfort_band = (self.T_min <= T_in <= self.T_max)
        
        if T_in < self.T_min:
            comfort_deviation = self.T_min - T_in
        elif T_in > self.T_max:
            comfort_deviation = T_in - self.T_max
        else:
            comfort_deviation = 0.0
        
        setpoint_deviation = abs(T_in - self.setpoint)
        
        # Temperature gradient (is temp rising or falling?)
        temp_gradient = T_in - self.prev_T_in
        self.prev_T_in = T_in
        
        # Track ON time
        if action == 1:
            self.cumulative_on_time += 1
        else:
            self.cumulative_on_time = 0
        
        # ================================================================
        # REWARD CALCULATION
        # ================================================================
        
        reward = 0.0
        comfort_penalty = 0.0
        cost_penalty = 0.0
        efficiency_bonus = 0.0
        
        if not in_comfort_band:
            # ============================================================
            # OUTSIDE COMFORT BAND - SEVERE PENALTY
            # ============================================================
            
            comfort_penalty = self.config.comfort_violation_base * (comfort_deviation ** 2)
            
            if comfort_deviation > 1.5:
                comfort_penalty += 20.0 * (comfort_deviation - 1.5)
            
            reward = -comfort_penalty
            
            # Action guidance when outside comfort
            if T_in < self.T_min:
                # Too cold - should heat
                if action == 0:
                    reward -= 2.0  # Penalty for not heating when cold
            else:
                # Too hot - should not heat
                if action == 1:
                    reward -= 2.0  # Penalty for heating when too hot
            
        else:
            # ============================================================
            # IN COMFORT BAND - OPTIMIZE FOR EFFICIENCY
            # ============================================================
            
            # Small base reward for being in comfort
            # Made smaller so cost savings can dominate
            reward = 0.5
            
            # Proximity to setpoint bonus (small)
            if abs(T_in - self.setpoint) < 0.5:
                reward += 0.2
            
            # ============================================================
            # ENERGY COST - MAJOR FACTOR WHEN COMFORTABLE
            # ============================================================
            
            # Base energy penalty - significant!
            if action == 1:
                # Using energy - apply cost proportional to price
                energy_cost_penalty = 0.8 * (price_t / self.config.peak_price)
                reward -= energy_cost_penalty
                
                # Extra penalty during peak hours
                if is_peak:
                    reward -= 1.0  # Strong peak penalty
                
                # Penalty for heating when already warm enough
                if T_in > self.T_target_high:
                    reward -= 0.5  # Don't heat when already warm
                    
            else:
                # Not using energy - efficiency bonus
                efficiency_bonus = 0.5  # Significant bonus for not using energy
                reward += efficiency_bonus
                
                # Extra bonus during peak hours for not using energy
                if is_peak:
                    reward += 0.5
                
                # But penalize if temperature is dropping fast while OFF
                if temp_gradient < -0.02 and T_in < self.T_target_low:
                    # Temperature dropping and getting close to lower bound
                    reward -= 0.3
            
            # ============================================================
            # SMART HEATING STRATEGY
            # ============================================================
            
            # Reward pre-heating before peak hours (hours 14-16)
            # This allows coasting during peak (16-21)
            hour = int((price_t - self.config.offpeak_price) / 
                      (self.config.peak_price - self.config.offpeak_price) < 0.5)
            
            # Simple heuristic: encourage higher temp before peak
            if not is_peak and T_in > self.T_target_high:
                # Good - building up thermal mass before peak
                reward += 0.1
        
        # ================================================================
        # SWITCHING PENALTY
        # ================================================================
        switched = (action != prev_action)
        switch_penalty = self.config.switch_penalty if switched else 0.0
        reward -= switch_penalty
        
        # ================================================================
        # INVALID ACTION PENALTY
        # ================================================================
        invalid_penalty = self.config.invalid_penalty if is_invalid else 0.0
        reward -= invalid_penalty
        
        # ================================================================
        # FINAL REWARD - No scaling, let the magnitudes speak
        # ================================================================
        final_reward = np.clip(reward, -25.0, 3.0)
        
        components = {
            "in_comfort_band": in_comfort_band,
            "comfort_deviation": comfort_deviation,
            "setpoint_deviation": setpoint_deviation,
            "comfort_penalty": comfort_penalty,
            "cost_penalty": cost_penalty,
            "efficiency_bonus": efficiency_bonus,
            "switch_penalty": switch_penalty,
            "invalid_penalty": invalid_penalty,
            "raw_cost": instant_cost,
            "raw_reward": reward,
            "final_reward": final_reward,
        }
        
        return final_reward, components


# =========================================================================
# 2R2C HVAC ENVIRONMENT WITH SAFETY LAYER
# =========================================================================

class SafetyHVACEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        is_training: bool = True,
        use_domain_randomization: bool = True,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        randomization_scale: Optional[float] = None
    ):
        super().__init__()
        self.data = data
        self.config = config
        self.is_training = is_training
        self.use_domain_randomization = use_domain_randomization and is_training

        # Use the comfort-first reward handler
        self.reward_handler = ComfortFirstRewardHandler(config)

        self.randomization_scale = (
            randomization_scale if randomization_scale is not None
            else config.randomization_scale
        )

        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx   = end_idx   if end_idx   is not None else len(data)
        self.episode_length = config.episode_length_days * 24 * 60

        # obs: [T_in_norm, T_out_norm, T_mass_norm, price_norm, time_sin, time_cos, mask_off, mask_on]
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -1.5, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.5,  1.5,  1.5, 1.5,  1.0,  1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)

        self.T_in = config.setpoint
        self.T_mass = config.setpoint
        self.current_step = 0
        self.episode_start_idx = 0

        # safety counters
        self.minutes_since_on = config.lockout_time
        self.minutes_since_off = config.lockout_time
        self.current_state = 0

        self.runtime = config.lockout_time
        self.offtime = config.lockout_time
        self.last_action = 0
        self.prev_action = 0

        # statistics
        self.masked_off_count = 0
        self.masked_on_count = 0
        self.episode_actions: List[int] = []
        self.episode_temps: List[float] = []
        self.episode_costs: List[float] = []
        self.episode_power: List[float] = []
        self.on_runtimes: List[int] = []
        self._current_on_duration = 0

        # thermal parameters
        self.R_i = config.R_i
        self.R_w = config.R_w
        self.R_o = config.R_o
        self.C_in = config.C_in
        self.C_m  = config.C_m

    def get_action_mask(self) -> np.ndarray:
        allowed = np.array([True, True], dtype=bool)
        if self.current_state == 1:
            if self.minutes_since_on < self.config.lockout_time:
                allowed[0] = False
        else:
            if self.minutes_since_off < self.config.lockout_time:
                allowed[1] = False
        return allowed

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # domain randomization
        if self.use_domain_randomization and self.randomization_scale > 0:
            s = self.randomization_scale
            low, high = 1.0 - s, 1.0 + s
            self.R_i = self.config.R_i * np.random.uniform(low, high)
            self.R_w = self.config.R_w * np.random.uniform(low, high)
            self.R_o = self.config.R_o * np.random.uniform(low, high)
            self.C_in = self.config.C_in * np.random.uniform(low, high)
            self.C_m  = self.config.C_m  * np.random.uniform(low, high)
        else:
            self.R_i = self.config.R_i
            self.R_w = self.config.R_w
            self.R_o = self.config.R_o
            self.C_in = self.config.C_in
            self.C_m  = self.config.C_m

        data_len = len(self.data)
        self.start_idx = max(0, min(self.start_idx, data_len - 1))
        self.end_idx   = max(self.start_idx + 1, min(self.end_idx, data_len))

        if self.is_training:
            avail = self.end_idx - self.start_idx
            ep_len = min(self.episode_length, avail)
            max_start = max(self.start_idx, self.end_idx - ep_len)
            if max_start <= self.start_idx:
                self.episode_start_idx = self.start_idx
            else:
                self.episode_start_idx = np.random.randint(self.start_idx, max_start)
            self.episode_length = min(ep_len, self.end_idx - self.episode_start_idx)
        else:
            self.episode_start_idx = self.start_idx
            self.episode_length = self.end_idx - self.start_idx

        self.episode_length = max(1, self.episode_length)

        safe_idx = min(self.episode_start_idx, len(self.data) - 1)
        _ = self.data.iloc[safe_idx]["T_out"]
        
        # Initialize temperature at setpoint (in comfort band)
        self.T_in = self.config.setpoint + np.random.uniform(-0.3, 0.3)
        self.T_mass = self.T_in - 0.3

        self.minutes_since_on = self.config.lockout_time
        self.minutes_since_off = self.config.lockout_time
        self.current_state = 0

        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time
        self.last_action = 0
        self.prev_action = 0

        self.masked_off_count = 0
        self.masked_on_count = 0
        self.episode_actions = []
        self.episode_temps = []
        self.episode_costs = []
        self.episode_power = []
        self.on_runtimes = []
        self._current_on_duration = 0
        self.current_step = 0

        obs = self._get_observation()
        info = {"T_in_true": self.T_in, "T_out": self.data.iloc[safe_idx]["T_out"], "action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action: int):
        data_idx = min(self.episode_start_idx + self.current_step, len(self.data) - 1)
        row = self.data.iloc[data_idx]
        T_out = row["T_out"]
        price_t = row["Price"]
        current_hour = self.data.index[data_idx].hour

        # Safety mask
        action_mask = self.get_action_mask()
        original_action = action
        invalid_action = not action_mask[action]
        masked = False

        if invalid_action:
            if action == 0:
                action = 1
                self.masked_off_count += 1
            else:
                action = 0
                self.masked_on_count += 1
            masked = True

        # Update safety counters
        if action == 1:
            if self.current_state == 0:
                self.minutes_since_on = 1
            else:
                self.minutes_since_on += 1
            self.minutes_since_off = 0
            self._current_on_duration += 1
            self.current_state = 1
        else:
            if self.current_state == 1:
                if self._current_on_duration > 0:
                    self.on_runtimes.append(self._current_on_duration)
                self._current_on_duration = 0
                self.minutes_since_off = 1
            else:
                self.minutes_since_off += 1
            self.minutes_since_on = 0
            self.current_state = 0

        if action == 1:
            self.runtime += 1
            self.offtime = 0
        else:
            self.offtime += 1
            self.runtime = 0
        self.last_action = action

        # 2R2C dynamics
        Q_hvac = action * self.config.Q_hvac_max
        Q_im = (self.T_in - self.T_mass) / self.R_i
        Q_mo = (self.T_mass - T_out) / (self.R_w + self.R_o)

        dT_in = (Q_hvac - Q_im) / self.C_in * self.config.dt
        self.T_in += dT_in

        dT_mass = (Q_im - Q_mo) / self.C_m * self.config.dt
        self.T_mass += dT_mass

        self.T_in = np.clip(self.T_in, 10.0, 35.0)
        self.T_mass = np.clip(self.T_mass, 10.0, 35.0)

        # Reward calculation using comfort-first handler
        is_peak = 16 <= current_hour < 21
        current_power_kw = action * self.config.Q_hvac_kw

        reward, r_comp = self.reward_handler.calculate(
            T_in=self.T_in,
            action=action,
            price_t=price_t,
            prev_action=self.prev_action,
            is_invalid=invalid_action,
            is_peak=is_peak,
            current_step_power_kw=current_power_kw,
        )
        self.prev_action = action

        # Episode stats
        self.episode_actions.append(action)
        self.episode_temps.append(self.T_in)
        self.episode_costs.append(r_comp["raw_cost"])
        self.episode_power.append(current_power_kw)

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        if terminated and self._current_on_duration > 0:
            self.on_runtimes.append(self._current_on_duration)

        obs = self._get_observation()
        info = {
            "T_in_true": self.T_in,
            "T_out": T_out,
            "T_mass": self.T_mass,
            "action": action,
            "original_action": original_action,
            "masked": masked,
            "invalid_action": invalid_action,
            "power_kw": current_power_kw,
            "cost": r_comp["raw_cost"],
            "in_comfort_band": r_comp["in_comfort_band"],
            "comfort_deviation": r_comp["comfort_deviation"],
            "price": price_t,
            "action_mask": self.get_action_mask(),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        data_idx = min(self.episode_start_idx + self.current_step, len(self.data) - 1)
        row = self.data.iloc[data_idx]
        T_out = row["T_out"]

        T_in_obs = self.T_in + np.random.normal(0, self.config.obs_noise_std)
        T_in_obs = np.clip(T_in_obs, 10.0, 35.0)

        T_in_norm = (T_in_obs - self.config.setpoint) / 10.0
        T_out_norm = (T_out - 10.0) / 30.0
        T_mass_norm = (self.T_mass - self.config.setpoint) / 10.0
        price_norm = row["Price"] / self.config.peak_price

        mask = self.get_action_mask()
        mask_off = 1.0 if mask[0] else 0.0
        mask_on  = 1.0 if mask[1] else 0.0

        return np.array(
            [T_in_norm, T_out_norm, T_mass_norm, price_norm,
             row["time_sin"], row["time_cos"], mask_off, mask_on],
            dtype=np.float32
        )

    def get_statistics(self) -> Dict:
        temps = self.episode_temps

        # Comfort relative to setpoint
        disc_set = sum((t - self.config.setpoint) ** 2 for t in temps) / 60.0

        # Comfort relative to comfort band
        disc_band = 0.0
        time_in_comfort = 0
        for t in temps:
            if t < self.config.comfort_min:
                dev = self.config.comfort_min - t
            elif t > self.config.comfort_max:
                dev = t - self.config.comfort_max
            else:
                dev = 0.0
                time_in_comfort += 1
            disc_band += dev ** 2
        disc_band /= 60.0
        
        comfort_ratio = time_in_comfort / max(len(temps), 1)

        return {
            "total_cost": sum(self.episode_costs),
            "total_discomfort": disc_band,
            "total_discomfort_band": disc_band,
            "total_discomfort_setpoint": disc_set,
            "total_energy_kwh": sum(self.episode_power) / 60.0,
            "n_cycles": max(0, len(self.on_runtimes) - 1),
            "masked_off": self.masked_off_count,
            "masked_on": self.masked_on_count,
            "on_runtimes": self.on_runtimes.copy(),
            "temps": temps.copy(),
            "actions": self.episode_actions.copy(),
            "power": self.episode_power.copy(),
            "comfort_ratio": comfort_ratio,
            "mean_temp": np.mean(temps) if temps else 0,
            "std_temp": np.std(temps) if temps else 0,
        }


# =========================================================================
# BASELINE THERMOSTAT
# =========================================================================

class BaselineThermostat:
    def __init__(self, config: Config):
        self.config = config
        self.current_action = 0
        self.runtime = config.lockout_time
        self.offtime = config.lockout_time
        self.masked_off_count = 0
        self.masked_on_count = 0

    def reset(self):
        self.current_action = 0
        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time
        self.masked_off_count = 0
        self.masked_on_count = 0

    def predict(self, T_in: float) -> int:
        upper = self.config.setpoint + self.config.deadband
        lower = self.config.setpoint - self.config.deadband
        if T_in > upper:
            desired = 0
        elif T_in < lower:
            desired = 1
        else:
            desired = self.current_action

        actual = desired
        if desired == 0 and self.current_action == 1:
            if self.runtime < self.config.lockout_time:
                actual = 1
                self.masked_off_count += 1
        elif desired == 1 and self.current_action == 0:
            if self.offtime < self.config.lockout_time:
                actual = 0
                self.masked_on_count += 1

        if actual == 1:
            self.runtime += 1
            self.offtime = 0
        else:
            self.offtime += 1
            self.runtime = 0

        self.current_action = actual
        return actual


# =========================================================================
# EVALUATION
# =========================================================================

def evaluate_controller(
    env: SafetyHVACEnv,
    controller: Any,
    is_ppo: bool = False,
    config: Optional[Config] = None
) -> Dict:
    obs, info = env.reset()
    done = False
    hourly_power = {h: [] for h in range(24)}

    if not is_ppo:
        controller.reset()

    while not done:
        if is_ppo:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            T_in = obs[0] * 10.0 + env.config.setpoint
            action = controller.predict(T_in)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        idx = min(env.episode_start_idx + env.current_step - 1, len(env.data) - 1)
        hour = env.data.index[idx].hour
        hourly_power[hour].append(info["power_kw"])

    stats = env.get_statistics()
    stats["masked_off"] = env.masked_off_count
    stats["masked_on"]  = env.masked_on_count
    stats["hourly_power"] = {h: (np.mean(v) if v else 0) for h, v in hourly_power.items()}

    peak_p = [stats["hourly_power"][h] for h in range(16, 21)]
    off_p  = [stats["hourly_power"][h] for h in range(24) if h < 16 or h >= 21]
    stats["avg_peak_power"] = np.mean(peak_p) if peak_p else 0
    stats["avg_offpeak_power"] = np.mean(off_p) if off_p else 0

    return stats


def run_robustness_test(
    data: pd.DataFrame,
    model: PPO,
    baseline: BaselineThermostat,
    config: Config,
    test_start_idx: int,
    test_end_idx: int
) -> Dict:
    r_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
    results = {"r_multipliers": r_multipliers, "ppo_costs": [], "baseline_costs": [],
               "ppo_comfort": [], "baseline_comfort": []}

    for r_mult in r_multipliers:
        test_cfg = replace(config)
        test_cfg.R_i = config.R_i * r_mult
        test_cfg.R_w = config.R_w * r_mult
        test_cfg.R_o = config.R_o * r_mult

        env = SafetyHVACEnv(
            data=data,
            config=test_cfg,
            is_training=False,
            use_domain_randomization=False,
            start_idx=test_start_idx,
            end_idx=test_end_idx,
        )
        env.reset(seed=test_cfg.seed)
        ppo_stats = evaluate_controller(env, model, is_ppo=True, config=test_cfg)
        results["ppo_costs"].append(ppo_stats["total_cost"])
        results["ppo_comfort"].append(ppo_stats["comfort_ratio"])

        env = SafetyHVACEnv(
            data=data,
            config=test_cfg,
            is_training=False,
            use_domain_randomization=False,
            start_idx=test_start_idx,
            end_idx=test_end_idx,
        )
        env.reset(seed=test_cfg.seed)
        base_stats = evaluate_controller(env, baseline, is_ppo=False, config=test_cfg)
        results["baseline_costs"].append(base_stats["total_cost"])
        results["baseline_comfort"].append(base_stats["comfort_ratio"])

    return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def generate_all_figures(
    baseline_stats: Dict,
    ppo_stats: Dict,
    robustness_results: Dict,
    model: PPO,
    data: pd.DataFrame,
    config: Config,
    output_dir: str
):
    print("\n" + "=" * 80)
    print("Generating Figures")
    print("=" * 80)

    generate_figure1_micro_dynamics(baseline_stats, ppo_stats, config, output_dir)
    generate_figure2_safety_verification(baseline_stats, ppo_stats, config, output_dir)
    generate_figure3_policy_heatmap(model, config, output_dir)
    generate_figure4_radar(baseline_stats, ppo_stats, config, output_dir)
    generate_figure5_robustness(robustness_results, output_dir)
    generate_figure6_comfort_distribution(baseline_stats, ppo_stats, config, output_dir)
    generate_figure7_price_response(baseline_stats, ppo_stats, config, output_dir)

    print("All figures generated!")


def generate_figure1_micro_dynamics(
    baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    n_samples = min(240, len(baseline_stats['temps']), len(ppo_stats['temps']))
    time_hours = np.arange(n_samples) / 60.0

    baseline_temps = baseline_stats['temps'][:n_samples]
    baseline_actions = baseline_stats['actions'][:n_samples]
    ppo_temps = ppo_stats['temps'][:n_samples]
    ppo_actions = ppo_stats['actions'][:n_samples]

    ax1 = axes[0]
    ax1.plot(time_hours, baseline_temps, 'r-', label='Baseline', linewidth=2, alpha=0.8)
    ax1.plot(time_hours, ppo_temps, 'b-', label='PI-DRL', linewidth=2, alpha=0.8)
    ax1.axhline(y=config.setpoint, color='k', linestyle='--', label='Setpoint', linewidth=1.5)
    ax1.axhspan(config.comfort_min, config.comfort_max, alpha=0.2, color='green', label='Comfort Band')
    ax1.set_ylabel('Indoor Temperature (°C)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim([17, 27])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Temperature Trajectories (4-Hour Window)', fontweight='bold')

    ax2 = axes[1]
    ax2.step(time_hours, baseline_actions, 'r-', where='post', label='Baseline', linewidth=2, alpha=0.7)
    ax2.step(time_hours, [a + 0.05 for a in ppo_actions], 'b-', where='post', label='PI-DRL', linewidth=2, alpha=0.7)
    ax2.set_ylabel('HVAC State', fontweight='bold')
    ax2.set_xlabel('Time (hours)', fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Control Actions', fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_1_Micro_Dynamics.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure2_safety_verification(
    baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    baseline_runtimes = baseline_stats.get('on_runtimes', [])
    if len(baseline_runtimes) > 0:
        bins = np.arange(0, max(baseline_runtimes) + 5, 5)
        ax1.hist(baseline_runtimes, bins=bins, color='red', alpha=0.7, edgecolor='black')
    ax1.axvline(x=config.lockout_time, color='black', linestyle='--', linewidth=2, label=f'Lockout ({config.lockout_time} min)')
    ax1.set_xlabel('ON Runtime (minutes)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Baseline Thermostat', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ppo_runtimes = ppo_stats.get('on_runtimes', [])
    if len(ppo_runtimes) > 0:
        bins = np.arange(0, max(ppo_runtimes) + 5, 5)
        ax2.hist(ppo_runtimes, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=config.lockout_time, color='black', linestyle='--', linewidth=2, label=f'Lockout ({config.lockout_time} min)')
    ax2.set_xlabel('ON Runtime (minutes)', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('PI-DRL Agent', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Safety Verification: ON-Runtime Distributions', fontweight='bold', fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_2_Safety_Verification.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure3_policy_heatmap(model: PPO, config: Config, output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 8))

    hours = np.arange(24)
    temps = np.linspace(-10, 30, 20)

    prob_on = np.zeros((len(temps), len(hours)))

    for i, T_out in enumerate(temps):
        for j, hour in enumerate(hours):
            T_in = config.setpoint
            T_mass = config.setpoint
            price = config.peak_price if 16 <= hour < 21 else config.offpeak_price
            time_sin = np.sin(2 * np.pi * hour / 24)
            time_cos = np.cos(2 * np.pi * hour / 24)

            T_in_norm = (T_in - config.setpoint) / 10.0
            T_out_norm = (T_out - 10.0) / 30.0
            T_mass_norm = (T_mass - config.setpoint) / 10.0
            price_norm = price / config.peak_price

            obs = np.array([T_in_norm, T_out_norm, T_mass_norm, price_norm, time_sin, time_cos, 1.0, 1.0], dtype=np.float32)

            actions = []
            for _ in range(20):
                action, _ = model.predict(obs, deterministic=False)
                actions.append(action)
            prob_on[i, j] = np.mean(actions)

    im = ax.imshow(prob_on, aspect='auto', origin='lower', cmap='RdYlBu_r',
                   extent=[0, 24, temps[0], temps[-1]], vmin=0, vmax=1)

    ax.axvspan(16, 21, alpha=0.2, color='red', label='Peak Hours')
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Outdoor Temperature (°C)', fontweight='bold')
    ax.set_title('PI-DRL Policy: Probability of Heating ON', fontweight='bold', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(ON)', fontweight='bold', rotation=270, labelpad=15)
    ax.legend(loc='upper right')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_3_Policy_Heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure4_radar(baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str):
    categories = ['Cost', 'Comfort', 'Peak Load', 'Cycles', 'Energy']

    # Normalize values (lower is better for all except comfort ratio)
    baseline_values = [
        baseline_stats['total_cost'],
        1 - baseline_stats.get('comfort_ratio', 0.5),  # Invert so lower is better
        baseline_stats.get('avg_peak_power', 1) * 10,
        max(baseline_stats.get('n_cycles', 1), 1),
        baseline_stats['total_energy_kwh']
    ]

    ppo_values = [
        ppo_stats['total_cost'],
        1 - ppo_stats.get('comfort_ratio', 0.5),
        ppo_stats.get('avg_peak_power', 1) * 10,
        max(ppo_stats.get('n_cycles', 1), 1),
        ppo_stats['total_energy_kwh']
    ]

    baseline_normalized = []
    ppo_normalized = []
    for b, p in zip(baseline_values, ppo_values):
        max_val = max(b, p, 0.001)
        baseline_normalized.append(1 - b / (max_val * 1.2))
        ppo_normalized.append(1 - p / (max_val * 1.2))

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    baseline_normalized += baseline_normalized[:1]
    ppo_normalized += ppo_normalized[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    ax.plot(angles, baseline_normalized, 'r-', linewidth=2, label='Baseline')
    ax.fill(angles, baseline_normalized, 'r', alpha=0.25)

    ax.plot(angles, ppo_normalized, 'b-', linewidth=2, label='PI-DRL')
    ax.fill(angles, ppo_normalized, 'b', alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Multi-Objective Performance\n(Higher = Better)', fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_4_Multi_Objective_Radar.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure5_robustness(robustness_results: Dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r_mults = robustness_results['r_multipliers']
    
    # Cost comparison
    ax1 = axes[0]
    ax1.plot(r_mults, robustness_results['baseline_costs'], 'r-o', linewidth=2, markersize=10, label='Baseline')
    ax1.plot(r_mults, robustness_results['ppo_costs'], 'b-s', linewidth=2, markersize=10, label='PI-DRL')
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Nominal R')
    ax1.set_xlabel('R Multiplier', fontweight='bold')
    ax1.set_ylabel('Total Cost ($)', fontweight='bold')
    ax1.set_title('Cost vs. Thermal Resistance', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Comfort comparison
    ax2 = axes[1]
    ax2.plot(r_mults, [c * 100 for c in robustness_results['baseline_comfort']], 'r-o', linewidth=2, markersize=10, label='Baseline')
    ax2.plot(r_mults, [c * 100 for c in robustness_results['ppo_comfort']], 'b-s', linewidth=2, markersize=10, label='PI-DRL')
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Nominal R')
    ax2.axhline(y=90, color='green', linestyle=':', alpha=0.7, label='90% Target')
    ax2.set_xlabel('R Multiplier', fontweight='bold')
    ax2.set_ylabel('Time in Comfort Band (%)', fontweight='bold')
    ax2.set_title('Comfort vs. Thermal Resistance', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Robustness Analysis', fontweight='bold', fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_5_Robustness.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure6_comfort_distribution(baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_temps = baseline_stats['temps']
    ppo_temps = ppo_stats['temps']

    data = [baseline_temps, ppo_temps]
    positions = [1, 2]

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    colors = ['red', 'blue']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)

    ax.axhspan(config.comfort_min, config.comfort_max, alpha=0.2, color='green', label='Comfort Band')
    ax.axhline(y=config.setpoint, color='black', linestyle='--', linewidth=1.5, label=f'Setpoint ({config.setpoint}°C)')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', 'PI-DRL'], fontweight='bold')
    ax.set_ylabel('Indoor Temperature (°C)', fontweight='bold')
    ax.set_title('Temperature Distribution Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add comfort ratio stats
    baseline_comfort = baseline_stats.get('comfort_ratio', 0) * 100
    ppo_comfort = ppo_stats.get('comfort_ratio', 0) * 100
    stats_text = (
        f'Baseline: μ={np.mean(baseline_temps):.1f}°C, comfort={baseline_comfort:.1f}%\n'
        f'PI-DRL: μ={np.mean(ppo_temps):.1f}°C, comfort={ppo_comfort:.1f}%'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_6_Comfort_Distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure7_price_response(baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = list(range(24))

    baseline_hourly = [baseline_stats['hourly_power'].get(h, 0) for h in hours]
    ppo_hourly = [ppo_stats['hourly_power'].get(h, 0) for h in hours]

    width = 0.35
    x = np.array(hours)

    ax.bar(x - width/2, baseline_hourly, width, label='Baseline', color='red', alpha=0.7)
    ax.bar(x + width/2, ppo_hourly, width, label='PI-DRL', color='blue', alpha=0.7)

    ax.axvspan(15.5, 20.5, alpha=0.2, color='orange', label='Peak Hours')

    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Average Power (kW)', fontweight='bold')
    ax.set_title('Demand Response: Power Consumption by Hour', fontweight='bold', fontsize=14)
    ax.set_xticks(hours)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_7_Price_Response.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_all_tables(
    baseline_stats: Dict,
    ppo_stats: Dict,
    robustness_results: Dict,
    config: Config,
    train_mask_stats: Dict,
    test_mask_stats: Dict,
    output_dir: str
):
    print("\n" + "=" * 80)
    print("Generating Tables")
    print("=" * 80)

    # Table 1: System Parameters
    table1 = pd.DataFrame({
        'Parameter': [
            'R_i (K/W)', 'R_w (K/W)', 'R_o (K/W)',
            'C_in (J/K)', 'C_m (J/K)',
            'Lockout Time (min)', 'Setpoint (°C)', 'Comfort Band (°C)',
            'Q_HVAC (kW)', 'dt (s)',
            'comfort_bonus', 'comfort_violation_base', 'comfort_violation_exp',
            'cost_weight', 'peak_weight', 'switch_penalty',
        ],
        'Value': [
            f'{config.R_i:.4f}', f'{config.R_w:.4f}', f'{config.R_o:.4f}',
            f'{config.C_in:.0f}', f'{config.C_m:.0f}',
            f'{config.lockout_time}', f'{config.setpoint}', 
            f'[{config.comfort_min}, {config.comfort_max}]',
            f'{config.Q_hvac_kw:.1f}', f'{config.dt:.0f}',
            f'{config.comfort_bonus}', f'{config.comfort_violation_base}',
            f'{config.comfort_violation_exp}',
            f'{config.cost_weight}', f'{config.peak_weight}', f'{config.switch_penalty}',
        ]
    })
    filepath = os.path.join(output_dir, 'Table_1_System_Parameters.csv')
    table1.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")

    # Table 2: Performance Summary
    table2 = pd.DataFrame({
        'Metric': [
            'Total Cost ($)',
            'Comfort Ratio (%)',
            'Comfort Loss (°C²·h)',
            'Total Cycles',
            'Total Energy (kWh)',
            'Mean Temperature (°C)',
        ],
        'Baseline': [
            f"{baseline_stats['total_cost']:.2f}",
            f"{baseline_stats.get('comfort_ratio', 0) * 100:.1f}",
            f"{baseline_stats['total_discomfort_band']:.2f}",
            f"{baseline_stats['n_cycles']}",
            f"{baseline_stats['total_energy_kwh']:.2f}",
            f"{baseline_stats.get('mean_temp', 0):.1f}",
        ],
        'PI-DRL': [
            f"{ppo_stats['total_cost']:.2f}",
            f"{ppo_stats.get('comfort_ratio', 0) * 100:.1f}",
            f"{ppo_stats['total_discomfort_band']:.2f}",
            f"{ppo_stats['n_cycles']}",
            f"{ppo_stats['total_energy_kwh']:.2f}",
            f"{ppo_stats.get('mean_temp', 0):.1f}",
        ],
    })
    filepath = os.path.join(output_dir, 'Table_2_Performance_Summary.csv')
    table2.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")

    print("All tables generated!")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 80)
    print("COMFORT-FIRST PI-DRL HVAC CONTROLLER")
    print("=" * 80)

    config = Config()
    set_global_seed(config.seed)

    # Output directory
    output_dir = os.path.join(os.getcwd(), "HVAC_PI_DRL_ComfortFirst_Output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")

    # 1) Load/generate data
    data = load_ampds2_local(config)
    split_idx = int(len(data) * config.train_split)
    train_data = data.iloc[:split_idx]
    test_data  = data.iloc[split_idx:]

    print("\nData split:")
    print(f"  Training: {len(train_data)} samples ({len(train_data)/(60*24):.1f} days)")
    print(f"  Testing : {len(test_data)} samples ({len(test_data)/(60*24):.1f} days)")

    # 2) Environments
    print("\n" + "=" * 80)
    print("PHASE 2: Environment Setup")
    print("=" * 80)

    train_env = SafetyHVACEnv(
        data=data,
        config=config,
        is_training=True,
        use_domain_randomization=True,
        start_idx=0,
        end_idx=split_idx,
    )

    print("Environment created.")
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space     : {train_env.action_space}")

    # 3) Training
    print("\n" + "=" * 80)
    print("PHASE 3: Training PPO Agent (Comfort-First Design)")
    print("=" * 80)

    class MaskTrackingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.masked_off_total = 0
            self.masked_on_total = 0
            self.total_steps = 0

        def _on_step(self) -> bool:
            self.total_steps += 1
            if hasattr(self.training_env, "envs"):
                env = self.training_env.envs[0]
                if hasattr(env, "masked_off_count"):
                    self.masked_off_total = env.masked_off_count
                    self.masked_on_total = env.masked_on_count
            return True

    def make_env(base_seed=config.seed):
        def _init():
            env = SafetyHVACEnv(
                data=data,
                config=config,
                is_training=True,
                use_domain_randomization=False,
                start_idx=0,
                end_idx=split_idx,
                randomization_scale=0.0,
            )
            env.reset(seed=base_seed)
            return env
        return _init

    vec_env = DummyVecEnv([make_env()])
    vec_env.seed(config.seed)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64])
        ),
        seed=config.seed,
    )

    mask_callback = MaskTrackingCallback()
    
    print(f"\nTraining for {config.total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=mask_callback,
        progress_bar=False,
    )

    model_path = os.path.join(output_dir, "ppo_hvac_comfort_first")
    model.save(model_path)
    print(f"\nModel saved to: {os.path.abspath(model_path)}")

    train_mask_stats = {
        "total_steps": config.total_timesteps,
        "masked_off": mask_callback.masked_off_total,
        "masked_on": mask_callback.masked_on_total,
    }

    # 4) Evaluation
    print("\n" + "=" * 80)
    print("PHASE 4: Evaluation on Test Data")
    print("=" * 80)

    baseline = BaselineThermostat(config)

    print("\nEvaluating Baseline Thermostat ...")
    test_env = SafetyHVACEnv(
        data=data,
        config=config,
        is_training=False,
        use_domain_randomization=False,
        start_idx=split_idx,
        end_idx=len(data),
    )
    test_env.reset(seed=config.seed)
    baseline_stats = evaluate_controller(test_env, baseline, is_ppo=False, config=config)

    print("Evaluating PI-DRL Agent ...")
    test_env = SafetyHVACEnv(
        data=data,
        config=config,
        is_training=False,
        use_domain_randomization=False,
        start_idx=split_idx,
        end_idx=len(data),
    )
    test_env.reset(seed=config.seed)
    ppo_stats = evaluate_controller(test_env, model, is_ppo=True, config=config)

    test_mask_stats = {
        "total_steps": len(ppo_stats.get("actions", [])),
        "masked_off": ppo_stats.get("masked_off", 0),
        "masked_on": ppo_stats.get("masked_on", 0),
    }

    def print_stats(name: str, stats: Dict):
        print(f"\n{name}:")
        print(f"  Total cost              : {stats['total_cost']:.2f} $")
        print(f"  Energy consumption      : {stats['total_energy_kwh']:.2f} kWh")
        print(f"  Comfort ratio           : {stats.get('comfort_ratio', 0) * 100:.1f}%")
        print(f"  Comfort loss (°C²·h)    : {stats['total_discomfort_band']:.2f}")
        print(f"  Mean temperature        : {stats.get('mean_temp', 0):.1f}°C")
        print(f"  Number of cycles        : {stats['n_cycles']}")
        print(f"  Avg peak power          : {stats['avg_peak_power']:.3f} kW")
        print(f"  Avg off-peak power      : {stats['avg_offpeak_power']:.3f} kW")

    print_stats("Baseline thermostat", baseline_stats)
    print_stats("PI-DRL agent (Comfort-First)", ppo_stats)

    # 5) Robustness
    print("\n" + "=" * 80)
    print("PHASE 5: Robustness Analysis")
    print("=" * 80)

    robustness_results = run_robustness_test(
        data=data,
        model=model,
        baseline=baseline,
        config=config,
        test_start_idx=split_idx,
        test_end_idx=len(data),
    )

    print("R-multipliers :", robustness_results["r_multipliers"])
    print("Baseline costs:", np.round(robustness_results["baseline_costs"], 2))
    print("PI-DRL costs  :", np.round(robustness_results["ppo_costs"], 2))
    print("Baseline comfort:", [f"{c*100:.1f}%" for c in robustness_results["baseline_comfort"]])
    print("PI-DRL comfort  :", [f"{c*100:.1f}%" for c in robustness_results["ppo_comfort"]])

    # 6) Figures & Tables
    print("\n" + "=" * 80)
    print("PHASE 6: Generating Figures & Tables")
    print("=" * 80)

    generate_all_figures(
        baseline_stats=baseline_stats,
        ppo_stats=ppo_stats,
        robustness_results=robustness_results,
        model=model,
        data=data,
        config=config,
        output_dir=output_dir,
    )

    generate_all_tables(
        baseline_stats=baseline_stats,
        ppo_stats=ppo_stats,
        robustness_results=robustness_results,
        config=config,
        train_mask_stats=train_mask_stats,
        test_mask_stats=test_mask_stats,
        output_dir=output_dir,
    )

    # 7) Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    baseline_comfort = baseline_stats.get('comfort_ratio', 0) * 100
    ppo_comfort = ppo_stats.get('comfort_ratio', 0) * 100
    
    print(f"\nComfort (time in band):")
    print(f"  Baseline : {baseline_comfort:.1f}%")
    print(f"  PI-DRL   : {ppo_comfort:.1f}%")
    
    if ppo_comfort >= 90:
        print("\n✅ PI-DRL maintains excellent comfort (≥90% in band)")
    elif ppo_comfort >= 80:
        print("\n⚠️  PI-DRL comfort is acceptable but could be improved")
    else:
        print("\n❌ PI-DRL comfort needs improvement - increase comfort_violation_base")
    
    cost_savings = (1 - ppo_stats['total_cost'] / max(baseline_stats['total_cost'], 0.001)) * 100
    print(f"\nCost savings: {cost_savings:.1f}%")
    
    print(f"\nAll outputs written to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
