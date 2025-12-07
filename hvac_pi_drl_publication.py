# -*- coding: utf-8 -*-
"""
Publication-Ready PI-DRL HVAC Controller for Applied Energy
============================================================

Key improvements over previous versions:
1. STRONG cycling penalty with minimum runtime enforcement
2. Multi-day episodes for realistic pre-heating/coasting strategies
3. Multiple climate scenarios (cold winter, mild winter, spring)
4. Statistical analysis across multiple seeds
5. Comprehensive performance metrics

Author: Energy Systems ML Researcher
"""

import os
from dataclasses import dataclass, replace, field
from typing import Dict, Tuple, Any, Optional, List
import json

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
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
})


# =========================================================================
# CONFIGURATION
# =========================================================================

@dataclass
class Config:
    """Configuration for publication-quality experiments."""
    data_dir: str = r"C:\Users\FATEME\Downloads\dataverse_files"
    seed: int = 42
    
    # 2R2C thermal parameters
    R_i: float = 0.001
    R_w: float = 0.004
    R_o: float = 0.003
    C_in: float = 500_000.0
    C_m: float = 2_000_000.0
    
    # HVAC
    Q_hvac_max: float = 3000.0
    Q_hvac_kw: float = 3.0
    dt: float = 60.0
    
    # Comfort settings
    setpoint: float = 21.0
    deadband: float = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    lockout_time: int = 15  # minutes - HARD constraint
    
    # Minimum runtime before switching (soft constraint via reward)
    min_runtime_soft: int = 30  # minutes
    
    # TOU tariff
    peak_price: float = 0.30
    offpeak_price: float = 0.10
    
    # Training - LONGER EPISODES for multi-day learning
    total_timesteps: int = 200_000
    episode_length_days: int = 2  # 2 days per episode
    train_split: float = 0.8
    
    # PPO hyperparameters - tuned for exploration and stability
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.05  # Higher entropy for more exploration
    clip_range: float = 0.2
    
    obs_noise_std: float = 0.1
    
    # Scenario-specific outdoor temperature
    scenario_name: str = "mild_winter"
    T_out_base: float = 8.0
    T_out_amplitude: float = 6.0


@dataclass
class ScenarioConfig:
    """Climate scenario configuration."""
    name: str
    T_out_base: float  # Base outdoor temperature
    T_out_amplitude: float  # Daily amplitude
    T_out_noise_std: float  # Temperature noise
    description: str


# Define multiple climate scenarios
SCENARIOS = {
    "cold_winter": ScenarioConfig(
        name="cold_winter",
        T_out_base=-2.0,
        T_out_amplitude=5.0,
        T_out_noise_std=3.0,
        description="Cold winter (-7°C to 3°C)"
    ),
    "mild_winter": ScenarioConfig(
        name="mild_winter",
        T_out_base=5.0,
        T_out_amplitude=6.0,
        T_out_noise_std=2.5,
        description="Mild winter (-1°C to 11°C)"
    ),
    "spring": ScenarioConfig(
        name="spring",
        T_out_base=12.0,
        T_out_amplitude=8.0,
        T_out_noise_std=2.0,
        description="Spring (4°C to 20°C)"
    ),
    "shoulder": ScenarioConfig(
        name="shoulder",
        T_out_base=15.0,
        T_out_amplitude=7.0,
        T_out_noise_std=2.0,
        description="Shoulder season (8°C to 22°C)"
    ),
}


def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
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
# DATA GENERATION
# =========================================================================

def generate_scenario_data(config: Config, scenario: ScenarioConfig, n_days: int = 30) -> pd.DataFrame:
    """Generate synthetic data for a specific climate scenario."""
    n_samples = n_days * 24 * 60
    start_date = pd.Timestamp("2024-01-01")
    index = pd.date_range(start=start_date, periods=n_samples, freq="1min")
    df = pd.DataFrame(index=index)
    
    hours = np.arange(n_samples) / 60.0
    
    # Daily temperature pattern: coldest at 6am, warmest at 3pm
    daily_pattern = -scenario.T_out_amplitude * np.cos(2 * np.pi * (hours - 6) / 24)
    
    # Weekly variation (warmer on some days)
    weekly_pattern = 2.0 * np.sin(2 * np.pi * hours / (24 * 7))
    
    # Random noise
    noise = np.random.normal(0, scenario.T_out_noise_std, n_samples)
    
    # Combine
    df["T_out"] = scenario.T_out_base + daily_pattern + weekly_pattern + noise
    df["T_out"] = df["T_out"].clip(-25, 35)
    
    # Price structure
    df["hour"] = df.index.hour
    df["Price"] = df["hour"].apply(
        lambda h: config.peak_price if 16 <= h < 21 else config.offpeak_price
    )
    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Add day of week for pre-heating patterns
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    
    return df


# =========================================================================
# IMPROVED REWARD HANDLER WITH STRONG CYCLING PENALTY
# =========================================================================

class PublicationRewardHandler:
    """
    Publication-quality reward function with:
    1. Strong cycling penalty (short cycles heavily penalized)
    2. Pre-heating incentives before peak hours
    3. Proper comfort-cost trade-off
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.T_min = config.comfort_min
        self.T_max = config.comfort_max
        self.setpoint = config.setpoint
        
        # Track runtime for cycling penalty
        self.current_runtime = 0
        self.current_offtime = 0
        self.last_switch_step = 0
        self.total_switches = 0
        
    def reset(self):
        """Reset tracking variables at episode start."""
        self.current_runtime = self.config.min_runtime_soft
        self.current_offtime = self.config.min_runtime_soft
        self.last_switch_step = 0
        self.total_switches = 0
        
    def calculate(
        self,
        T_in: float,
        action: int,
        price_t: float,
        prev_action: int,
        is_invalid: bool,
        is_peak: bool,
        current_step_power_kw: float,
        current_step: int,
        hour: int
    ) -> Tuple[float, Dict]:
        """
        Calculate reward with strong cycling penalty.
        """
        
        # Energy cost
        dt_hours = 1.0 / 60.0
        energy_kwh = current_step_power_kw * dt_hours
        instant_cost = energy_kwh * price_t
        
        # ================================================================
        # COMFORT TERM
        # ================================================================
        in_comfort_band = (self.T_min <= T_in <= self.T_max)
        
        if T_in < self.T_min:
            comfort_deviation = self.T_min - T_in
        elif T_in > self.T_max:
            comfort_deviation = T_in - self.T_max
        else:
            comfort_deviation = 0.0
        
        if in_comfort_band:
            setpoint_dist = abs(T_in - self.setpoint)
            comfort_term = 0.5 + 0.5 * max(0, 1 - setpoint_dist / 2.0)
        else:
            # Strong penalty outside comfort
            comfort_term = -3.0 * (comfort_deviation ** 1.5)
        
        # ================================================================
        # COST TERM
        # ================================================================
        max_cost_per_step = 0.015
        if action == 1:
            cost_term = 0.4 * (instant_cost / max_cost_per_step)
            if is_peak:
                cost_term *= 1.8  # Strong peak penalty
        else:
            cost_term = 0.0
        
        # ================================================================
        # CYCLING PENALTY - KEY IMPROVEMENT
        # ================================================================
        switched = (action != prev_action)
        cycling_penalty = 0.0
        
        if switched:
            self.total_switches += 1
            
            # Base switching penalty
            cycling_penalty = 0.3
            
            # Additional penalty for short cycles
            if prev_action == 1:
                # Turning OFF - penalize short ON cycles
                if self.current_runtime < self.config.min_runtime_soft:
                    # Exponential penalty for very short cycles
                    shortness = 1.0 - (self.current_runtime / self.config.min_runtime_soft)
                    cycling_penalty += 0.5 * (shortness ** 2)
                self.current_runtime = 0
                self.current_offtime = 1
            else:
                # Turning ON - penalize short OFF cycles
                if self.current_offtime < self.config.min_runtime_soft:
                    shortness = 1.0 - (self.current_offtime / self.config.min_runtime_soft)
                    cycling_penalty += 0.5 * (shortness ** 2)
                self.current_offtime = 0
                self.current_runtime = 1
        else:
            # Not switching - update counters
            if action == 1:
                self.current_runtime += 1
            else:
                self.current_offtime += 1
        
        # ================================================================
        # PRE-HEATING INCENTIVE
        # ================================================================
        preheat_bonus = 0.0
        
        # Hours before peak (14-16): encourage pre-heating
        if 14 <= hour < 16 and not is_peak:
            if action == 1 and T_in < self.setpoint + 1.5:
                preheat_bonus = 0.15  # Encourage heating before peak
        
        # During peak: reward coasting (high temp, HVAC off)
        if is_peak and action == 0 and in_comfort_band:
            if T_in > self.setpoint:
                preheat_bonus = 0.2  # Reward for coasting during peak
        
        # ================================================================
        # ACTION GUIDANCE (SHAPING) - CRITICAL FOR PREVENTING ALWAYS-ON
        # ================================================================
        shaping = 0.0
        
        if in_comfort_band:
            # In comfort band - encourage appropriate actions based on temperature
            if T_in > self.setpoint + 1.0:
                # Warm side of comfort band
                if action == 0:
                    shaping += 0.2  # Good: not heating when warm
                else:
                    shaping -= 0.3  # Bad: heating when already warm
            elif T_in < self.setpoint - 0.5:
                # Cool side of comfort band
                if action == 1:
                    shaping += 0.1  # Good: heating when cool
                # Not heating when cool is okay if temperature is stable
            elif self.setpoint - 0.5 <= T_in <= self.setpoint + 0.5:
                # Near setpoint - encourage efficiency
                if action == 0:
                    shaping += 0.15  # Good: not heating when at setpoint
        else:
            # Outside comfort band - strong guidance
            if T_in < self.T_min:
                # Too cold
                if action == 0:
                    shaping -= 0.5  # Very bad: not heating when cold
                else:
                    shaping += 0.1  # Good: heating when cold
            elif T_in > self.T_max:
                # Too hot (OVERHEATING)
                if action == 1:
                    shaping -= 1.0  # VERY bad: heating when already too hot!
                else:
                    shaping += 0.2  # Good: not heating when hot
        
        # ================================================================
        # INVALID ACTION PENALTY
        # ================================================================
        invalid_penalty = 1.5 if is_invalid else 0.0
        
        # ================================================================
        # FINAL REWARD
        # ================================================================
        reward = (
            comfort_term 
            - cost_term 
            - cycling_penalty 
            + preheat_bonus 
            + shaping 
            - invalid_penalty
        )
        
        final_reward = np.clip(reward, -15.0, 2.0)
        
        components = {
            "in_comfort_band": in_comfort_band,
            "comfort_deviation": comfort_deviation,
            "comfort_term": comfort_term,
            "cost_term": cost_term,
            "cycling_penalty": cycling_penalty,
            "preheat_bonus": preheat_bonus,
            "shaping": shaping,
            "invalid_penalty": invalid_penalty,
            "raw_cost": instant_cost,
            "final_reward": final_reward,
            "current_runtime": self.current_runtime,
            "current_offtime": self.current_offtime,
        }
        
        return final_reward, components


# =========================================================================
# ENVIRONMENT
# =========================================================================

class PublicationHVACEnv(gym.Env):
    """Publication-quality HVAC environment with multi-day episodes."""
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        is_training: bool = True,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        super().__init__()
        self.data = data
        self.config = config
        self.is_training = is_training

        self.reward_handler = PublicationRewardHandler(config)

        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx = end_idx if end_idx is not None else len(data)
        self.episode_length = config.episode_length_days * 24 * 60

        # Observation: [T_in, T_out, T_mass, price, time_sin, time_cos, mask_off, mask_on, runtime_norm, is_peak]
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -1.5, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        self._init_state()

    def _init_state(self):
        self.T_in = self.config.setpoint
        self.T_mass = self.config.setpoint
        self.current_step = 0
        self.episode_start_idx = 0
        
        self.minutes_since_on = self.config.lockout_time
        self.minutes_since_off = self.config.lockout_time
        self.current_state = 0
        self.prev_action = 0
        
        self.masked_off_count = 0
        self.masked_on_count = 0
        self.episode_actions = []
        self.episode_temps = []
        self.episode_costs = []
        self.episode_power = []
        self.on_runtimes = []
        self._current_on_duration = 0
        
        self.R_i = self.config.R_i
        self.R_w = self.config.R_w
        self.R_o = self.config.R_o
        self.C_in = self.config.C_in
        self.C_m = self.config.C_m

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
        self._init_state()
        self.reward_handler.reset()

        data_len = len(self.data)
        self.start_idx = max(0, min(self.start_idx, data_len - 1))
        self.end_idx = max(self.start_idx + 1, min(self.end_idx, data_len))

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

        # Initialize temperature with some variation
        self.T_in = self.config.setpoint + np.random.uniform(-0.5, 0.5)
        self.T_mass = self.T_in - np.random.uniform(0, 0.3)

        obs = self._get_observation()
        return obs, {"T_in_true": self.T_in}

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
        
        if invalid_action:
            action = 1 if action == 0 else 0
            if original_action == 0:
                self.masked_off_count += 1
            else:
                self.masked_on_count += 1

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

        # Reward
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
            current_step=self.current_step,
            hour=current_hour
        )
        self.prev_action = action

        # Episode stats
        self.episode_actions.append(action)
        self.episode_temps.append(self.T_in)
        self.episode_costs.append(r_comp["raw_cost"])
        self.episode_power.append(current_power_kw)

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        
        if terminated and self._current_on_duration > 0:
            self.on_runtimes.append(self._current_on_duration)

        obs = self._get_observation()
        info = {
            "T_in_true": self.T_in,
            "T_out": T_out,
            "action": action,
            "power_kw": current_power_kw,
            "cost": r_comp["raw_cost"],
            "in_comfort_band": r_comp["in_comfort_band"],
            "cycling_penalty": r_comp["cycling_penalty"],
        }

        return obs, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        data_idx = min(self.episode_start_idx + self.current_step, len(self.data) - 1)
        row = self.data.iloc[data_idx]
        T_out = row["T_out"]
        current_hour = self.data.index[data_idx].hour

        T_in_obs = self.T_in + np.random.normal(0, self.config.obs_noise_std)
        T_in_obs = np.clip(T_in_obs, 10.0, 35.0)

        T_in_norm = (T_in_obs - self.config.setpoint) / 10.0
        T_out_norm = (T_out - 10.0) / 30.0
        T_mass_norm = (self.T_mass - self.config.setpoint) / 10.0
        price_norm = row["Price"] / self.config.peak_price

        mask = self.get_action_mask()
        
        # Runtime normalization (how long current state has been active)
        if self.current_state == 1:
            runtime_norm = min(self.minutes_since_on / 60.0, 1.0)
        else:
            runtime_norm = min(self.minutes_since_off / 60.0, 1.0)
        
        is_peak = 1.0 if 16 <= current_hour < 21 else 0.0
        
        return np.array([
            T_in_norm, T_out_norm, T_mass_norm, price_norm,
            row["time_sin"], row["time_cos"],
            1.0 if mask[0] else 0.0, 1.0 if mask[1] else 0.0,
            runtime_norm, is_peak
        ], dtype=np.float32)

    def get_statistics(self) -> Dict:
        temps = self.episode_temps
        
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
        
        # Calculate actual cycles (ON->OFF transitions)
        n_cycles = 0
        for i in range(1, len(self.episode_actions)):
            if self.episode_actions[i-1] == 1 and self.episode_actions[i] == 0:
                n_cycles += 1

        # Calculate average cycle length
        avg_on_duration = np.mean(self.on_runtimes) if self.on_runtimes else 0
        
        # Calculate short cycles (< min_runtime_soft)
        short_cycles = sum(1 for r in self.on_runtimes if r < self.config.min_runtime_soft)

        return {
            "total_cost": sum(self.episode_costs),
            "total_discomfort": disc_band,
            "total_energy_kwh": sum(self.episode_power) / 60.0,
            "n_cycles": n_cycles,
            "comfort_ratio": comfort_ratio,
            "mean_temp": np.mean(temps) if temps else 0,
            "std_temp": np.std(temps) if temps else 0,
            "on_ratio": np.mean(self.episode_actions) if self.episode_actions else 0,
            "avg_on_duration": avg_on_duration,
            "short_cycles": short_cycles,
            "on_runtimes": self.on_runtimes.copy(),
            "temps": temps.copy(),
            "actions": self.episode_actions.copy(),
        }


# =========================================================================
# BASELINE THERMOSTAT
# =========================================================================

class SmartThermostat:
    """
    Improved baseline with minimum runtime enforcement.
    More representative of modern smart thermostats.
    """
    def __init__(self, config: Config):
        self.config = config
        self.current_action = 0
        self.runtime = config.lockout_time
        self.offtime = config.lockout_time
        self.min_runtime = config.min_runtime_soft

    def reset(self):
        self.current_action = 0
        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time

    def predict(self, T_in: float, hour: int = 0) -> int:
        upper = self.config.setpoint + self.config.deadband
        lower = self.config.setpoint - self.config.deadband
        
        # Basic thermostat logic
        if T_in > upper:
            desired = 0
        elif T_in < lower:
            desired = 1
        else:
            desired = self.current_action

        actual = desired
        
        # Enforce minimum runtime (lockout)
        if desired == 0 and self.current_action == 1:
            if self.runtime < self.config.lockout_time:
                actual = 1
        elif desired == 1 and self.current_action == 0:
            if self.offtime < self.config.lockout_time:
                actual = 0

        # Update counters
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
    env: PublicationHVACEnv,
    controller,
    is_ppo: bool = False,
    seed: int = 42
) -> Dict:
    """Evaluate a controller and return statistics."""
    env.reset(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    hourly_power = {h: [] for h in range(24)}

    if not is_ppo:
        controller.reset()

    while not done:
        if is_ppo:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            T_in = obs[0] * 10.0 + env.config.setpoint
            hour = int((obs[4] * 12 / np.pi + 6) % 24)  # Approximate hour from time_sin
            action = controller.predict(T_in, hour)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        idx = min(env.episode_start_idx + env.current_step - 1, len(env.data) - 1)
        hour = env.data.index[idx].hour
        hourly_power[hour].append(info["power_kw"])

    stats = env.get_statistics()
    stats["hourly_power"] = {h: (np.mean(v) if v else 0) for h, v in hourly_power.items()}

    peak_p = [stats["hourly_power"][h] for h in range(16, 21)]
    off_p = [stats["hourly_power"][h] for h in range(24) if h < 16 or h >= 21]
    stats["avg_peak_power"] = np.mean(peak_p) if peak_p else 0
    stats["avg_offpeak_power"] = np.mean(off_p) if off_p else 0

    return stats


def run_statistical_evaluation(
    data: pd.DataFrame,
    model: PPO,
    baseline: SmartThermostat,
    config: Config,
    test_start_idx: int,
    test_end_idx: int,
    n_seeds: int = 5,
    n_days_per_eval: int = 3
) -> Dict:
    """
    Run evaluation across multiple seeds and aggregate statistics.
    """
    seeds = [42 + i * 17 for i in range(n_seeds)]
    
    baseline_results = []
    ppo_results = []
    
    for seed in seeds:
        set_global_seed(seed)
        
        # Evaluate baseline
        env = PublicationHVACEnv(
            data=data, config=config, is_training=False,
            start_idx=test_start_idx, end_idx=test_end_idx
        )
        baseline_stats = evaluate_controller(env, baseline, is_ppo=False, seed=seed)
        baseline_results.append(baseline_stats)
        
        # Evaluate PI-DRL
        env = PublicationHVACEnv(
            data=data, config=config, is_training=False,
            start_idx=test_start_idx, end_idx=test_end_idx
        )
        ppo_stats = evaluate_controller(env, model, is_ppo=True, seed=seed)
        ppo_results.append(ppo_stats)
    
    # Aggregate statistics
    metrics = ["total_cost", "total_energy_kwh", "comfort_ratio", "n_cycles", 
               "mean_temp", "avg_on_duration", "short_cycles"]
    
    aggregated = {"baseline": {}, "ppo": {}, "comparison": {}}
    
    for metric in metrics:
        baseline_vals = [r[metric] for r in baseline_results]
        ppo_vals = [r[metric] for r in ppo_results]
        
        aggregated["baseline"][metric] = {
            "mean": np.mean(baseline_vals),
            "std": np.std(baseline_vals),
            "values": baseline_vals
        }
        aggregated["ppo"][metric] = {
            "mean": np.mean(ppo_vals),
            "std": np.std(ppo_vals),
            "values": ppo_vals
        }
        
        # Statistical test (paired t-test)
        if len(baseline_vals) >= 2:
            t_stat, p_value = stats.ttest_rel(baseline_vals, ppo_vals)
            aggregated["comparison"][metric] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "improvement_pct": (np.mean(baseline_vals) - np.mean(ppo_vals)) / np.mean(baseline_vals) * 100
            }
    
    return aggregated


# =========================================================================
# VISUALIZATION
# =========================================================================

def generate_publication_figures(
    results_by_scenario: Dict,
    output_dir: str
):
    """Generate publication-quality figures."""
    
    # Figure 1: Performance comparison across scenarios
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = list(results_by_scenario.keys())
    metrics = ["total_cost", "comfort_ratio", "n_cycles", "total_energy_kwh"]
    titles = ["Total Cost ($)", "Comfort Ratio (%)", "Number of Cycles", "Energy (kWh)"]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        baseline_means = []
        baseline_stds = []
        ppo_means = []
        ppo_stds = []
        
        for scenario in scenarios:
            res = results_by_scenario[scenario]
            baseline_means.append(res["baseline"][metric]["mean"])
            baseline_stds.append(res["baseline"][metric]["std"])
            ppo_means.append(res["ppo"][metric]["mean"])
            ppo_stds.append(res["ppo"][metric]["std"])
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        # Convert comfort ratio to percentage
        if metric == "comfort_ratio":
            baseline_means = [m * 100 for m in baseline_means]
            baseline_stds = [s * 100 for s in baseline_stds]
            ppo_means = [m * 100 for m in ppo_means]
            ppo_stds = [s * 100 for s in ppo_stds]
        
        bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                       label='Baseline', color='#d62728', alpha=0.8, capsize=3)
        bars2 = ax.bar(x + width/2, ppo_means, width, yerr=ppo_stds,
                       label='PI-DRL', color='#1f77b4', alpha=0.8, capsize=3)
        
        ax.set_xlabel('Climate Scenario', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_Performance_Comparison.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Figure 2: Statistical significance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = []
    p_values = []
    metric_labels = []
    
    # Use first scenario for detailed stats
    first_scenario = scenarios[0]
    res = results_by_scenario[first_scenario]
    
    for metric in ["total_cost", "total_energy_kwh", "n_cycles"]:
        if metric in res["comparison"]:
            improvements.append(res["comparison"][metric]["improvement_pct"])
            p_values.append(res["comparison"][metric]["p_value"])
            metric_labels.append(metric.replace('_', ' ').title())
    
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    bars = ax.barh(metric_labels, improvements, color=colors, alpha=0.7)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Improvement (%)', fontweight='bold')
    ax.set_title(f'PI-DRL Improvements ({first_scenario.replace("_", " ").title()})\n'
                 f'Green = Statistically Significant (p < 0.05)', fontweight='bold')
    
    # Add p-value annotations
    for i, (imp, p) in enumerate(zip(improvements, p_values)):
        ax.annotate(f'p={p:.3f}', xy=(imp, i), xytext=(5, 0), 
                    textcoords='offset points', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_Statistical_Significance.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figures saved to {output_dir}")


def generate_results_table(results_by_scenario: Dict, output_dir: str):
    """Generate publication-quality results table."""
    
    rows = []
    for scenario, results in results_by_scenario.items():
        row = {
            "Scenario": scenario.replace("_", " ").title(),
            "Baseline Cost ($)": f"{results['baseline']['total_cost']['mean']:.2f} ± {results['baseline']['total_cost']['std']:.2f}",
            "PI-DRL Cost ($)": f"{results['ppo']['total_cost']['mean']:.2f} ± {results['ppo']['total_cost']['std']:.2f}",
            "Cost Reduction (%)": f"{results['comparison']['total_cost']['improvement_pct']:.1f}",
            "Baseline Comfort (%)": f"{results['baseline']['comfort_ratio']['mean']*100:.1f} ± {results['baseline']['comfort_ratio']['std']*100:.1f}",
            "PI-DRL Comfort (%)": f"{results['ppo']['comfort_ratio']['mean']*100:.1f} ± {results['ppo']['comfort_ratio']['std']*100:.1f}",
            "Baseline Cycles": f"{results['baseline']['n_cycles']['mean']:.0f} ± {results['baseline']['n_cycles']['std']:.0f}",
            "PI-DRL Cycles": f"{results['ppo']['n_cycles']['mean']:.0f} ± {results['ppo']['n_cycles']['std']:.0f}",
            "p-value (cost)": f"{results['comparison']['total_cost']['p_value']:.4f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'Table_Results_Summary.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'Table_Results_Summary.tex')
    df.to_latex(latex_path, index=False, escape=False)
    
    print(f"Tables saved to {output_dir}")
    return df


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 80)
    print("PUBLICATION-READY PI-DRL HVAC CONTROLLER")
    print("For Applied Energy Journal")
    print("=" * 80)

    config = Config()
    output_dir = os.path.join(os.getcwd(), "HVAC_Publication_Output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Number of seeds for statistical analysis
    N_SEEDS = 5
    
    # Scenarios to evaluate
    scenarios_to_run = ["cold_winter", "mild_winter", "spring"]
    
    results_by_scenario = {}
    
    for scenario_name in scenarios_to_run:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"{'='*80}")
        
        scenario = SCENARIOS[scenario_name]
        config.scenario_name = scenario_name
        
        # Generate data for this scenario
        set_global_seed(config.seed)
        data = generate_scenario_data(config, scenario, n_days=30)
        split_idx = int(len(data) * config.train_split)
        
        print(f"\nScenario: {scenario.description}")
        print(f"T_out mean: {data['T_out'].mean():.1f}°C, "
              f"min: {data['T_out'].min():.1f}°C, max: {data['T_out'].max():.1f}°C")
        print(f"Train: {split_idx} samples ({split_idx//(24*60)} days)")
        print(f"Test: {len(data)-split_idx} samples ({(len(data)-split_idx)//(24*60)} days)")
        
        # Training
        print(f"\n--- Training PI-DRL Agent ---")
        
        def make_env():
            def _init():
                env = PublicationHVACEnv(
                    data=data, config=config, is_training=True,
                    start_idx=0, end_idx=split_idx
                )
                env.reset(seed=config.seed)
                return env
            return _init

        vec_env = DummyVecEnv([make_env()])
        
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            ent_coef=config.ent_coef,
            clip_range=config.clip_range,
            verbose=0,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            seed=config.seed,
        )
        
        model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
        
        # Save model
        model_path = os.path.join(output_dir, f"ppo_{scenario_name}")
        model.save(model_path)
        print(f"Model saved: {model_path}")
        
        # Statistical evaluation
        print(f"\n--- Statistical Evaluation ({N_SEEDS} seeds) ---")
        
        baseline = SmartThermostat(config)
        
        scenario_results = run_statistical_evaluation(
            data=data,
            model=model,
            baseline=baseline,
            config=config,
            test_start_idx=split_idx,
            test_end_idx=len(data),
            n_seeds=N_SEEDS,
            n_days_per_eval=config.episode_length_days
        )
        
        results_by_scenario[scenario_name] = scenario_results
        
        # Print results
        print(f"\nResults for {scenario_name}:")
        print("-" * 50)
        
        for metric in ["total_cost", "comfort_ratio", "n_cycles", "total_energy_kwh"]:
            baseline_mean = scenario_results["baseline"][metric]["mean"]
            baseline_std = scenario_results["baseline"][metric]["std"]
            ppo_mean = scenario_results["ppo"][metric]["mean"]
            ppo_std = scenario_results["ppo"][metric]["std"]
            
            if metric in scenario_results["comparison"]:
                p_val = scenario_results["comparison"][metric]["p_value"]
                improvement = scenario_results["comparison"][metric]["improvement_pct"]
                sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else "")
            else:
                p_val = 1.0
                improvement = 0
                sig = ""
            
            if metric == "comfort_ratio":
                print(f"  {metric}: Baseline={baseline_mean*100:.1f}±{baseline_std*100:.1f}%, "
                      f"PI-DRL={ppo_mean*100:.1f}±{ppo_std*100:.1f}% "
                      f"(Δ={improvement:+.1f}% {sig})")
            else:
                print(f"  {metric}: Baseline={baseline_mean:.2f}±{baseline_std:.2f}, "
                      f"PI-DRL={ppo_mean:.2f}±{ppo_std:.2f} "
                      f"(Δ={improvement:+.1f}% {sig})")
    
    # Generate figures and tables
    print(f"\n{'='*80}")
    print("GENERATING PUBLICATION FIGURES AND TABLES")
    print(f"{'='*80}")
    
    generate_publication_figures(results_by_scenario, output_dir)
    results_df = generate_results_table(results_by_scenario, output_dir)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print("\n" + results_df.to_string(index=False))
    
        # Save all results as JSON (with proper type conversion)
    json_results = {}
    for scenario, results in results_by_scenario.items():
        json_results[scenario] = {
            "baseline": {k: {"mean": float(v["mean"]), "std": float(v["std"])} 
                        for k, v in results["baseline"].items()},
            "ppo": {k: {"mean": float(v["mean"]), "std": float(v["std"])} 
                   for k, v in results["ppo"].items()},
            "comparison": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer, bool)) else vv 
                              for kk, vv in v.items()} 
                          for k, v in results["comparison"].items()}
        }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   - Models: ppo_*.zip")
    print(f"   - Figures: Figure_*.png")
    print(f"   - Tables: Table_*.csv, Table_*.tex")
    print(f"   - Raw data: results.json")


if __name__ == "__main__":
    main()
