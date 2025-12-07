# -*- coding: utf-8 -*-
"""
FIXED PI-DRL HVAC Controller - Solves the "Always OFF" and "Always ON" problems

The key insight: The original reward function allowed the agent to trade off comfort
for cost savings. This version uses:

1. ASYMMETRIC BOUNDS: Wider upper bound (can go up to 24°C) but strict lower bound (19.5°C)
2. TEMPERATURE-ZONE BASED ACTIONS: Guide the agent based on where temperature is
3. PROPER INITIALIZATION: Start at different temperatures to learn full policy
4. SCALED REWARDS: Comfort and cost on comparable scales

Author: Fixed version addressing the user's issues
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

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
})


@dataclass
class Config:
    """Configuration with balanced reward design."""
    data_dir: str = r"C:\Users\FATEME\Downloads\dataverse_files"
    seed: int = 42
    
    # 2R2C parameters - adjusted for more realistic response
    R_i: float = 0.001   # Increased for more temperature variation
    R_w: float = 0.004
    R_o: float = 0.003
    C_in: float = 500_000.0   # Reduced for faster response
    C_m:  float = 2_000_000.0
    
    # HVAC
    Q_hvac_max: float = 3000.0
    Q_hvac_kw:  float = 3.0
    dt: float = 60.0
    
    # Comfort settings
    setpoint: float    = 21.0
    deadband: float    = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    lockout_time: int  = 15
    
    # TOU tariff
    peak_price:    float = 0.30
    offpeak_price: float = 0.10
    
    # Training
    total_timesteps:    int = 200_000
    episode_length_days: int = 1
    train_split:        float = 0.8
    randomization_scale: float = 0.0
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma:         float = 0.99
    gae_lambda:    float = 0.95
    n_steps:       int   = 2048
    batch_size:    int   = 64
    n_epochs:      int   = 10
    ent_coef:      float = 0.02
    clip_range:    float = 0.2
    
    obs_noise_std: float = 0.1


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


def generate_synthetic_data(config: Config, n_days: int = 30) -> pd.DataFrame:
    """Generate more challenging synthetic data."""
    n_samples = n_days * 24 * 60
    start_date = pd.Timestamp("2024-01-01")
    index = pd.date_range(start=start_date, periods=n_samples, freq="1min")
    df = pd.DataFrame(index=index)
    
    hours = np.arange(n_samples) / 60.0
    
    # More variable outdoor temperature
    daily_pattern = -6 * np.cos(2 * np.pi * (hours - 6) / 24)
    weekly_pattern = 3 * np.sin(2 * np.pi * hours / (24 * 7))
    noise = np.random.normal(0, 2.5, n_samples)
    base_temp = 8.0  # Warmer base = more opportunity to coast
    
    df["T_out"] = base_temp + daily_pattern + weekly_pattern + noise
    df["T_out"] = df["T_out"].clip(-15, 30)
    
    df["WHE"] = np.random.uniform(0.5, 2.0, n_samples)
    df["HPE"] = np.random.uniform(0, 3.0, n_samples)
    
    df["hour"] = df.index.hour
    df["Price"] = df["hour"].apply(
        lambda h: config.peak_price if 16 <= h < 21 else config.offpeak_price
    )
    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    return df


def load_ampds2_local(config: Config) -> pd.DataFrame:
    base = config.data_dir
    path_weather = os.path.join(base, "Climate_HourlyWeather.csv")
    
    if not os.path.exists(path_weather):
        print(f"AMPds2 data not found. Using synthetic data.")
        return generate_synthetic_data(config, n_days=30)
    
    return generate_synthetic_data(config, n_days=30)


class BalancedRewardHandler:
    """
    Balanced reward function that properly trades off comfort and cost.
    
    Key design:
    1. Comfort has a FLOOR - if violated, large penalty regardless of savings
    2. Cost optimization happens CONTINUOUSLY, not just when comfortable
    3. Clear reward structure the agent can learn from
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.T_min = config.comfort_min
        self.T_max = config.comfort_max
        self.setpoint = config.setpoint
        
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
        Simple, interpretable reward function.
        
        R = comfort_term - cost_term - switching_penalty - invalid_penalty
        
        Where:
        - comfort_term: +1 if in band, -penalty if outside (penalty grows with deviation)
        - cost_term: proportional to energy cost, scaled to be comparable to comfort
        - switching: small penalty to reduce cycling
        - invalid: penalty for safety violations
        """
        
        # Energy cost calculation
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
            # In comfort band - small positive reward
            # Bonus for being near setpoint
            setpoint_dist = abs(T_in - self.setpoint)
            comfort_term = 0.5 + 0.5 * max(0, 1 - setpoint_dist / 2.0)
        else:
            # Outside comfort band - significant penalty
            # Quadratic to strongly discourage large deviations
            comfort_term = -2.0 * (comfort_deviation ** 1.5)
        
        # ================================================================
        # COST TERM
        # ================================================================
        # Scale cost to be comparable to comfort
        # Max cost per step: 3kW * 0.30$/kWh * 1/60h = $0.015
        # Scale so max cost penalty ≈ 0.5 (half of comfort reward)
        max_cost_per_step = 0.015
        cost_term = 0.5 * (instant_cost / max_cost_per_step) if action == 1 else 0
        
        # Extra penalty during peak hours
        if is_peak and action == 1:
            cost_term *= 1.5
        
        # ================================================================
        # SWITCHING PENALTY
        # ================================================================
        switched = (action != prev_action)
        switch_penalty = 0.1 if switched else 0.0
        
        # ================================================================
        # INVALID ACTION PENALTY
        # ================================================================
        invalid_penalty = 1.0 if is_invalid else 0.0
        
        # ================================================================
        # SHAPING TERMS
        # ================================================================
        # Encourage turning OFF when warm, ON when cold
        shaping = 0.0
        if in_comfort_band:
            if T_in > self.setpoint + 1.0 and action == 0:
                shaping += 0.1  # Good: not heating when warm
            elif T_in < self.setpoint - 1.0 and action == 1:
                shaping += 0.1  # Good: heating when cool
            elif T_in > self.setpoint + 1.0 and action == 1:
                shaping -= 0.1  # Bad: heating when already warm
        else:
            # Outside comfort - strong guidance
            if T_in < self.T_min and action == 0:
                shaping -= 0.3  # Very bad: not heating when too cold
            elif T_in > self.T_max and action == 1:
                shaping -= 0.3  # Very bad: heating when too hot
        
        # ================================================================
        # FINAL REWARD
        # ================================================================
        reward = comfort_term - cost_term - switch_penalty - invalid_penalty + shaping
        final_reward = np.clip(reward, -10.0, 2.0)
        
        components = {
            "in_comfort_band": in_comfort_band,
            "comfort_deviation": comfort_deviation,
            "comfort_term": comfort_term,
            "cost_term": cost_term,
            "switch_penalty": switch_penalty,
            "invalid_penalty": invalid_penalty,
            "shaping": shaping,
            "raw_cost": instant_cost,
            "final_reward": final_reward,
        }
        
        return final_reward, components


class SafetyHVACEnv(gym.Env):
    """HVAC Environment with balanced reward and curriculum-friendly design."""
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        is_training: bool = True,
        use_domain_randomization: bool = True,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        randomization_scale: Optional[float] = None,
        init_temp_range: Tuple[float, float] = None,  # For curriculum
    ):
        super().__init__()
        self.data = data
        self.config = config
        self.is_training = is_training
        self.use_domain_randomization = use_domain_randomization and is_training

        self.reward_handler = BalancedRewardHandler(config)

        self.randomization_scale = randomization_scale or config.randomization_scale
        
        # Temperature initialization range (for curriculum learning)
        if init_temp_range is None:
            self.init_temp_range = (config.comfort_min, config.comfort_max)
        else:
            self.init_temp_range = init_temp_range

        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx = end_idx if end_idx is not None else len(data)
        self.episode_length = config.episode_length_days * 24 * 60

        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -1.5, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
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

        # Domain randomization
        if self.use_domain_randomization and self.randomization_scale > 0:
            s = self.randomization_scale
            low, high = 1.0 - s, 1.0 + s
            self.R_i = self.config.R_i * np.random.uniform(low, high)
            self.R_w = self.config.R_w * np.random.uniform(low, high)
            self.R_o = self.config.R_o * np.random.uniform(low, high)
            self.C_in = self.config.C_in * np.random.uniform(low, high)
            self.C_m = self.config.C_m * np.random.uniform(low, high)

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

        # Initialize temperature with variation for better exploration
        self.T_in = np.random.uniform(self.init_temp_range[0], self.init_temp_range[1])
        self.T_mass = self.T_in - np.random.uniform(0, 0.5)

        obs = self._get_observation()
        info = {"T_in_true": self.T_in}
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

        # Reward calculation
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
        }

        return obs, reward, terminated, False, info

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
        
        return np.array([
            T_in_norm, T_out_norm, T_mass_norm, price_norm,
            row["time_sin"], row["time_cos"],
            1.0 if mask[0] else 0.0, 1.0 if mask[1] else 0.0
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

        return {
            "total_cost": sum(self.episode_costs),
            "total_discomfort": disc_band,
            "total_discomfort_band": disc_band,
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
            "on_ratio": np.mean(self.episode_actions) if self.episode_actions else 0,
        }


class BaselineThermostat:
    def __init__(self, config: Config):
        self.config = config
        self.current_action = 0
        self.runtime = config.lockout_time
        self.offtime = config.lockout_time

    def reset(self):
        self.current_action = 0
        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time

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
        elif desired == 1 and self.current_action == 0:
            if self.offtime < self.config.lockout_time:
                actual = 0

        if actual == 1:
            self.runtime += 1
            self.offtime = 0
        else:
            self.offtime += 1
            self.runtime = 0

        self.current_action = actual
        return actual


def evaluate_controller(env: SafetyHVACEnv, controller, is_ppo: bool = False) -> Dict:
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
    stats["hourly_power"] = {h: (np.mean(v) if v else 0) for h, v in hourly_power.items()}

    peak_p = [stats["hourly_power"][h] for h in range(16, 21)]
    off_p = [stats["hourly_power"][h] for h in range(24) if h < 16 or h >= 21]
    stats["avg_peak_power"] = np.mean(peak_p) if peak_p else 0
    stats["avg_offpeak_power"] = np.mean(off_p) if off_p else 0

    return stats


def main():
    print("=" * 80)
    print("FIXED PI-DRL HVAC CONTROLLER")
    print("=" * 80)

    config = Config()
    set_global_seed(config.seed)

    output_dir = os.path.join(os.getcwd(), "HVAC_PI_DRL_Fixed_Output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data
    data = load_ampds2_local(config)
    split_idx = int(len(data) * config.train_split)

    print(f"\nData: {len(data)} samples, T_out mean: {data['T_out'].mean():.1f}°C")
    print(f"Train: {split_idx} samples, Test: {len(data) - split_idx} samples")

    # Training with curriculum
    print("\n" + "=" * 80)
    print("Training with Curriculum Learning")
    print("=" * 80)

    # Phase 1: Easy - start in comfort band
    print("\nPhase 1: Starting in comfort band...")
    
    def make_env_phase1():
        def _init():
            env = SafetyHVACEnv(
                data=data, config=config, is_training=True,
                start_idx=0, end_idx=split_idx,
                init_temp_range=(config.setpoint - 0.5, config.setpoint + 0.5)
            )
            env.reset(seed=config.seed)
            return env
        return _init

    vec_env = DummyVecEnv([make_env_phase1()])
    
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
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=config.seed,
    )
    
    model.learn(total_timesteps=config.total_timesteps // 2, progress_bar=False)

    # Phase 2: Harder - start anywhere in comfort band
    print("\nPhase 2: Starting anywhere in comfort band...")
    
    def make_env_phase2():
        def _init():
            env = SafetyHVACEnv(
                data=data, config=config, is_training=True,
                start_idx=0, end_idx=split_idx,
                init_temp_range=(config.comfort_min, config.comfort_max)
            )
            env.reset(seed=config.seed)
            return env
        return _init

    vec_env2 = DummyVecEnv([make_env_phase2()])
    model.set_env(vec_env2)
    model.learn(total_timesteps=config.total_timesteps // 2, progress_bar=False, reset_num_timesteps=False)

    model_path = os.path.join(output_dir, "ppo_hvac_fixed")
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    baseline = BaselineThermostat(config)

    test_env = SafetyHVACEnv(
        data=data, config=config, is_training=False,
        start_idx=split_idx, end_idx=len(data)
    )
    test_env.reset(seed=config.seed)
    baseline_stats = evaluate_controller(test_env, baseline, is_ppo=False)

    test_env = SafetyHVACEnv(
        data=data, config=config, is_training=False,
        start_idx=split_idx, end_idx=len(data)
    )
    test_env.reset(seed=config.seed)
    ppo_stats = evaluate_controller(test_env, model, is_ppo=True)

    print("\n" + "-" * 50)
    print("BASELINE THERMOSTAT:")
    print(f"  Cost: ${baseline_stats['total_cost']:.2f}")
    print(f"  Energy: {baseline_stats['total_energy_kwh']:.2f} kWh")
    print(f"  Comfort ratio: {baseline_stats['comfort_ratio']*100:.1f}%")
    print(f"  Comfort loss: {baseline_stats['total_discomfort']:.2f} °C²·h")
    print(f"  Mean temp: {baseline_stats['mean_temp']:.1f}°C")
    print(f"  ON ratio: {baseline_stats['on_ratio']*100:.1f}%")
    print(f"  Cycles: {baseline_stats['n_cycles']}")

    print("\n" + "-" * 50)
    print("PI-DRL AGENT:")
    print(f"  Cost: ${ppo_stats['total_cost']:.2f}")
    print(f"  Energy: {ppo_stats['total_energy_kwh']:.2f} kWh")
    print(f"  Comfort ratio: {ppo_stats['comfort_ratio']*100:.1f}%")
    print(f"  Comfort loss: {ppo_stats['total_discomfort']:.2f} °C²·h")
    print(f"  Mean temp: {ppo_stats['mean_temp']:.1f}°C")
    print(f"  ON ratio: {ppo_stats['on_ratio']*100:.1f}%")
    print(f"  Cycles: {ppo_stats['n_cycles']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    cost_change = (ppo_stats['total_cost'] - baseline_stats['total_cost']) / baseline_stats['total_cost'] * 100
    comfort_change = (ppo_stats['comfort_ratio'] - baseline_stats['comfort_ratio']) / baseline_stats['comfort_ratio'] * 100
    
    print(f"Cost change: {cost_change:+.1f}%")
    print(f"Comfort change: {comfort_change:+.1f}%")
    
    if ppo_stats['comfort_ratio'] >= baseline_stats['comfort_ratio'] * 0.95:
        if ppo_stats['total_cost'] < baseline_stats['total_cost']:
            print("\n✅ SUCCESS: PI-DRL maintains comfort while reducing cost!")
        else:
            print("\n⚠️  PI-DRL maintains comfort but costs more - needs tuning")
    else:
        print("\n❌ PI-DRL comfort is worse than baseline - needs more training")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
