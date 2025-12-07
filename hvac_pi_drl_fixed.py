# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:09:07 2025

@author: FATEME
"""

"""
Robust, Safety-Critical 2R2C PI-DRL Controller for Residential HVAC
Using Local AMPds2 Dataset (No Synthetic Data)

- Loads AMPds2 CSV files from a local folder
- Builds a 2R2C thermal model with safety lockout
- Trains a PPO-based PI-DRL controller
- Compares against a baseline thermostat
- Generates figures and tables in a local output folder

Author: Energy Systems ML Researcher
"""

import os
from dataclasses import dataclass
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

# Added for reproducibility
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
# CONFIGURATION
# =========================================================================

@dataclass
class Config:
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
    total_timesteps:    int = 150_000
    episode_length_days: int = 1
    train_split:        float = 0.8

    # Reward weights – COMFORT DOMINANT (fixed)
    # Comfort must be prioritized MUCH more than cost
    w_cost:   float = 0.01      # Reduced from 0.08
    w_disc:   float = 100.0     # Increased from 15 - CRITICAL for comfort
    w_switch: float = 0.5       # Reduced from 1
    w_peak:   float = 0.05      # Reduced from 0.15
    w_invalid: float = 10.0     # Increased from 7

    # Normalization scales - Adjusted for better balance
    cost_scale:   float = 0.01
    disc_scale:   float = 0.01   # Increased from 0.001 to reduce amplification
    switch_scale: float = 1.0
    peak_scale:   float = 0.01

    # PPO hyperparameters
    learning_rate: float = 1e-4
    gamma:         float = 0.97
    gae_lambda:    float = 0.95
    n_steps:       int   = 2048
    batch_size:    int   = 64
    n_epochs:      int   = 10
    ent_coef:      float = 0.05
    clip_range:    float = 0.15

    # Observation noise
    obs_noise_std: float = 0.1


# =========================================================================
# GLOBAL SEEDING
# =========================================================================

def set_global_seed(seed: int):
    """
    Fix all random seeds to make results reproducible.
    """
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
# DATA LOADING FROM LOCAL AMPds2
# =========================================================================

def load_ampds2_local(config: Config) -> pd.DataFrame:
    """
    Loads AMPds2 from local CSV files (no synthetic data).
    Required files inside config.data_dir:
      - Climate_HourlyWeather.csv  (for outdoor temperature)
      - Electricity_WHE.csv       (whole-house electricity)
      - Electricity_HPE.csv       (heat pump or main heating circuit)
    Returns:
      DataFrame with index DatetimeIndex (1-min), columns:
      T_out [°C], WHE [kW], HPE [kW], Price, hour, time_sin, time_cos
    """
    print("=" * 80)
    print("PHASE 1: Loading AMPds2 from LOCAL folder")
    print("=" * 80)

    base = config.data_dir
    print(f"Using data directory: {os.path.abspath(base)}")

    path_weather = os.path.join(base, "Climate_HourlyWeather.csv")
    path_whe     = os.path.join(base, "Electricity_WHE.csv")
    path_hpe     = os.path.join(base, "Electricity_HPE.csv")

    for p in [path_weather, path_whe, path_hpe]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                f"Make sure AMPds2 CSVs are extracted into: {base}"
            )

    # --- Weather (hourly) ---
    df_weather = pd.read_csv(path_weather)
    if "Date/Time" not in df_weather.columns:
        raise RuntimeError("Climate_HourlyWeather.csv must contain 'Date/Time' column.")
    df_weather["datetime"] = pd.to_datetime(df_weather["Date/Time"])
    df_weather = df_weather.set_index("datetime").sort_index()

    # pick temperature column
    temp_col = None
    for c in df_weather.columns:
        if c == "Temp (C)" or "temp" in c.lower():
            temp_col = c
            break
    if temp_col is None:
        raise RuntimeError("Could not find temperature column in Climate_HourlyWeather.csv")

    df_weather = df_weather[[temp_col]].copy()
    df_weather = df_weather.resample("1min").interpolate("time").ffill().bfill()

    # --- Electricity WHE & HPE (sub-meter) ---
    def _load_power(path: str) -> pd.DataFrame:
        d = pd.read_csv(path)
        if "unix_ts" not in d.columns:
            raise RuntimeError(f"{os.path.basename(path)} must contain 'unix_ts' column.")
        d["datetime"] = pd.to_datetime(d["unix_ts"], unit="s")
        d = d.set_index("datetime").sort_index()
        d = d.resample("1min").mean()
        return d

    df_whe = _load_power(path_whe)
    df_hpe = _load_power(path_hpe)

    def _pick_P(df: pd.DataFrame, name: str) -> str:
        if "P" in df.columns:
            return "P"
        for c in df.columns:
            if any(k in c.lower() for k in ["p", "power", "watt"]):
                return c
        raise RuntimeError(f"Could not detect power column in {name}")

    col_whe = _pick_P(df_whe, "Electricity_WHE")
    col_hpe = _pick_P(df_hpe, "Electricity_HPE")

    # --- Align time range ---
    start = max(df_weather.index.min(), df_whe.index.min(), df_hpe.index.min())
    end   = min(df_weather.index.max(), df_whe.index.max(), df_hpe.index.max())
    if start >= end:
        raise RuntimeError("No overlapping time range between weather and electricity data.")

    common_index = pd.date_range(start=start, end=end, freq="1min")

    df_weather = df_weather.reindex(common_index, method="nearest")
    df_whe     = df_whe.reindex(common_index, method="nearest")
    df_hpe     = df_hpe.reindex(common_index, method="nearest")

    df = pd.DataFrame(index=common_index)
    df["T_out"] = df_weather[temp_col].values

    def _to_kw(x: np.ndarray) -> np.ndarray:
        m = np.nanmean(x)
        if m is None or np.isnan(m):
            return x
        return x / 1000.0 if m > 100 else x

    df["WHE"] = _to_kw(df_whe[col_whe].values.astype(float))
    df["HPE"] = _to_kw(df_hpe[col_hpe].values.astype(float))

    df["hour"] = df.index.hour
    df["Price"] = df["hour"].apply(
        lambda h: config.peak_price if 16 <= h < 21 else config.offpeak_price
    )

    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df = df.ffill().bfill()
    df["T_out"] = df["T_out"].clip(-40, 50)

    print(f"Final dataset: {len(df)} samples ({len(df)/(60*24):.1f} days)")
    print(f"Time range   : {df.index.min()}  →  {df.index.max()}")
    print(f"T_out range  : {df['T_out'].min():.1f}°C  to  {df['T_out'].max():.1f}°C")

    return df


# =========================================================================
# ADAPTIVE REWARD HANDLER (BAND + SETPOINT) - FIXED VERSION
# =========================================================================

class AdaptiveRewardHandler:
    """
    Reward function with STRONG preference for comfort over energy saving.
    FIXED: Comfort penalties are now MUCH stronger and more immediate.
    """

    def __init__(self, config: Config):
        self.config = config
        self.T_min = config.comfort_min
        self.T_max = config.comfort_max
        # Hard safety comfort band (slightly wider)
        self.hard_min = self.T_min - 1.0
        self.hard_max = self.T_max + 1.0

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

        dt_hours = 1.0 / 60.0
        energy_kwh = current_step_power_kw * dt_hours
        instant_cost = energy_kwh * price_t

        # ========== COMFORT TERMS (STRONGER) ==========
        in_soft_band = (self.T_min <= T_in <= self.T_max)
        in_hard_band = (self.hard_min <= T_in <= self.hard_max)

        # Deviation from comfort band (CRITICAL)
        if T_in < self.T_min:
            dev_band = self.T_min - T_in
        elif T_in > self.T_max:
            dev_band = T_in - self.T_max
        else:
            dev_band = 0.0

        # Deviation from setpoint (secondary)
        dev_set = abs(T_in - self.config.setpoint)

        # CRITICAL: Use exponential penalty for comfort violations
        # This makes violations MUCH more costly
        if dev_band > 0:
            # Exponential penalty: small violations are costly, large ones are VERY costly
            disc_band = (dev_band ** 2) * (1.0 + 5.0 * dev_band)
        else:
            disc_band = 0.0

        # Setpoint tracking (quadratic, less aggressive)
        disc_set = dev_set ** 2

        # Hard band penalty (EXTREME penalty if outside hard band)
        hard_penalty = 0.0
        if not in_hard_band:
            if T_in < self.hard_min:
                d_h = self.hard_min - T_in
            else:
                d_h = T_in - self.hard_max
            # Extreme penalty: 100x for hard violations
            hard_penalty = (d_h ** 2) * 200.0

        # Normalize comfort penalty (using updated disc_scale)
        disc_term = (
            disc_set +           # Setpoint tracking
            10.0 * disc_band +   # Band violation (10x weight)
            hard_penalty          # Hard violation
        ) / max(self.config.disc_scale, 1e-6)

        # STRONG bonus for staying inside comfort band
        bonus = 20.0 if in_soft_band else 0.0

        # ========== COST TERMS (REDUCED) ==========
        # Only apply cost penalty if we're in comfort band
        # If outside comfort, cost is irrelevant
        if in_soft_band:
            cost_term = instant_cost / self.config.cost_scale
        else:
            # Outside comfort: cost penalty is minimal
            cost_term = 0.01 * (instant_cost / self.config.cost_scale)

        # ========== PEAK TERMS (REDUCED) ==========
        peak_term = 0.0
        if is_peak:
            if in_soft_band:
                # Only penalize peak if we're comfortable
                peak_term = (energy_kwh / self.config.peak_scale)
            else:
                # Outside comfort: peak penalty is minimal
                peak_term = 0.01 * (energy_kwh / self.config.peak_scale)

        # ========== SWITCHING TERM (REDUCED) ==========
        switch_term = (1.0 if action != prev_action else 0.0) / self.config.switch_scale

        # ========== INVALID ACTION (STRONG PENALTY) ==========
        invalid_term = 10.0 if is_invalid else 0.0

        # ========== FINAL REWARD CALCULATION ==========
        total_penalty = (
            self.config.w_cost    * cost_term +
            self.config.w_disc    * disc_term +
            self.config.w_switch  * switch_term +
            self.config.w_peak    * peak_term +
            self.config.w_invalid * invalid_term
        )

        # Reward = bonus - penalty
        # Removed the 0.005 scaling - let the weights do their job
        raw_reward = bonus - total_penalty
        
        # Clip to reasonable range but keep it meaningful
        reward = np.clip(raw_reward, -100.0, 100.0)

        components = {
            "cost_term":     cost_term,
            "disc_term":     disc_term,
            "disc_set":      disc_set,
            "disc_band":     disc_band,
            "switch_term":   switch_term,
            "peak_term":     peak_term,
            "invalid_term":  invalid_term,
            "bonus":         bonus,
            "raw_cost":      instant_cost,
            "raw_discomfort": disc_band,
            "raw_reward":    raw_reward,
            "scaled_reward": reward,
        }

        return reward, components


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

        self.reward_handler = AdaptiveRewardHandler(config)

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
        self.T_in = self.config.setpoint + np.random.uniform(-0.5, 0.5)
        self.T_mass = self.T_in - 0.5

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

        # 2) safety mask
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

        # update safety counters
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

        # 3) 2R2C dynamics
        Q_hvac = action * self.config.Q_hvac_max
        Q_im = (self.T_in - self.T_mass) / self.R_i
        Q_mo = (self.T_mass - T_out) / (self.R_w + self.R_o)

        dT_in = (Q_hvac - Q_im) / self.C_in * self.config.dt
        self.T_in += dT_in

        dT_mass = (Q_im - Q_mo) / self.C_m * self.config.dt
        self.T_mass += dT_mass

        self.T_in = np.clip(self.T_in, 10.0, 35.0)
        self.T_mass = np.clip(self.T_mass, 10.0, 35.0)

        # 4) reward
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

        # 5) episode stats
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
            "discomfort": r_comp["raw_discomfort"],
            "price": price_t,
            "reward_bonus": r_comp["bonus"],
            "reward_cost_term": r_comp["cost_term"],
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

        # comfort relative to setpoint
        disc_set = sum((t - self.config.setpoint) ** 2 for t in temps) / 60.0

        # comfort relative to comfort band
        disc_band = 0.0
        for t in temps:
            if t < self.config.comfort_min:
                dev = self.config.comfort_min - t
            elif t > self.config.comfort_max:
                dev = t - self.config.comfort_max
            else:
                dev = 0.0
            disc_band += dev ** 2
        disc_band /= 60.0

        return {
            "total_cost": sum(self.episode_costs),
            # primary comfort metric for backward compatibility: band-based
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
# EVALUATION & ROBUSTNESS
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
    results = {"r_multipliers": r_multipliers, "ppo_costs": [], "baseline_costs": []}

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
    print("PHASE 4: Generating Figures")
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
    ax1.axhline(
        y=config.setpoint, color='k', linestyle='--',
        label='Setpoint (21°C)', linewidth=1.5
    )
    ax1.axhspan(
        config.comfort_min, config.comfort_max,
        alpha=0.2, color='green', label='Comfort Band'
    )
    ax1.set_ylabel('Indoor Temperature (°C)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim([18, 26])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Temperature Trajectories (4-Hour Window)', fontweight='bold')

    ax2 = axes[1]
    ax2.step(time_hours, baseline_actions, 'r-', where='post',
             label='Baseline', linewidth=2, alpha=0.7)
    ax2.step(time_hours, [a + 0.05 for a in ppo_actions], 'b-', where='post',
             label='PI-DRL', linewidth=2, alpha=0.7)
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
        ax1.hist(
            baseline_runtimes, bins=bins,
            color='red', alpha=0.7, edgecolor='black'
        )
    ax1.axvline(
        x=config.lockout_time, color='black',
        linestyle='--', linewidth=2, label=f'Lockout ({config.lockout_time} min)'
    )
    ax1.set_xlabel('ON Runtime (minutes)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Baseline Thermostat', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    baseline_violations = sum(1 for r in baseline_runtimes if r < config.lockout_time)
    ax1.text(
        0.95, 0.95, f'Violations < 15 min: {baseline_violations}',
        transform=ax1.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    ax2 = axes[1]
    ppo_runtimes = ppo_stats.get('on_runtimes', [])
    if len(ppo_runtimes) > 0:
        bins = np.arange(0, max(ppo_runtimes) + 5, 5)
        ax2.hist(
            ppo_runtimes, bins=bins,
            color='blue', alpha=0.7, edgecolor='black'
        )
    ax2.axvline(
        x=config.lockout_time, color='black',
        linestyle='--', linewidth=2, label=f'Lockout ({config.lockout_time} min)'
    )
    ax2.set_xlabel('ON Runtime (minutes)', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('PI-DRL Agent', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ppo_violations = sum(1 for r in ppo_runtimes if r < config.lockout_time)
    ax2.text(
        0.95, 0.95, f'Violations < 15 min: {ppo_violations}',
        transform=ax2.transAxes, ha='right', va='top',
        bbox=dict(
            boxstyle='round',
            facecolor='lightgreen' if ppo_violations == 0 else 'wheat',
            alpha=0.8
        )
    )

    plt.suptitle('Safety Verification: ON-Runtime Distributions',
                 fontweight='bold', fontsize=14)
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

            mask_off = 1.0
            mask_on = 1.0

            obs = np.array([
                T_in_norm, T_out_norm, T_mass_norm,
                price_norm, time_sin, time_cos, mask_off, mask_on
            ], dtype=np.float32)

            actions = []
            for _ in range(20):
                action, _ = model.predict(obs, deterministic=False)
                actions.append(action)
            prob_on[i, j] = np.mean(actions)

    im = ax.imshow(
        prob_on, aspect='auto', origin='lower', cmap='RdYlBu_r',
        extent=[0, 24, temps[0], temps[-1]], vmin=0, vmax=1
    )

    ax.axvspan(16, 21, alpha=0.2, color='red', label='Peak Hours (16:00-21:00)')

    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Outdoor Temperature (°C)', fontweight='bold')
    ax.set_title('PI-DRL Policy: Probability of Heating ON',
                 fontweight='bold', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(ON)', fontweight='bold', rotation=270, labelpad=15)

    ax.legend(loc='upper right')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_3_Policy_Heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure4_radar(
    baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str
):
    categories = ['Total Cost', 'Discomfort (band)', 'Peak Load', 'Cycles', 'Energy']

    baseline_values = [
        baseline_stats['total_cost'],
        baseline_stats['total_discomfort_band'],
        baseline_stats.get('avg_peak_power', 1) * 10,
        max(baseline_stats.get('n_cycles', 1), 1),
        baseline_stats['total_energy_kwh']
    ]

    ppo_values = [
        ppo_stats['total_cost'],
        ppo_stats['total_discomfort_band'],
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
    ax.set_title('Multi-Objective Performance\n(Higher = Better)',
                 fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_4_Multi_Objective_Radar.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure5_robustness(robustness_results: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    r_mults = robustness_results['r_multipliers']
    ppo_costs = robustness_results['ppo_costs']
    baseline_costs = robustness_results['baseline_costs']

    ax.plot(r_mults, baseline_costs, 'r-o', linewidth=2,
            markersize=10, label='Baseline')
    ax.plot(r_mults, ppo_costs, 'b-s', linewidth=2,
            markersize=10, label='PI-DRL')

    ax.axvline(x=1.0, color='gray', linestyle='--',
               alpha=0.5, label='Nominal R')

    ax.set_xlabel('R Multiplier', fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontweight='bold')
    ax.set_title(
        'Robustness Analysis: Cost vs. Thermal Resistance Variation',
        fontweight='bold', fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_5_Robustness.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure6_comfort_distribution(
    baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str
):
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_temps = baseline_stats['temps']
    ppo_temps = ppo_stats['temps']

    data = [baseline_temps, ppo_temps]
    positions = [1, 2]

    parts = ax.violinplot(data, positions=positions,
                          showmeans=True, showmedians=True)

    colors = ['red', 'blue']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)

    ax.axhspan(
        config.comfort_min, config.comfort_max,
        alpha=0.2, color='green', label='Comfort Band'
    )
    ax.axhline(
        y=config.setpoint, color='black', linestyle='--',
        linewidth=1.5, label=f'Setpoint ({config.setpoint}°C)'
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', 'PI-DRL'], fontweight='bold')
    ax.set_ylabel('Indoor Temperature (°C)', fontweight='bold')
    ax.set_title('Temperature Distribution Comparison',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    stats_text = (
        f'Baseline: μ={np.mean(baseline_temps):.1f}°C, '
        f'σ={np.std(baseline_temps):.2f}°C\n'
        f'PI-DRL: μ={np.mean(ppo_temps):.1f}°C, '
        f'σ={np.std(ppo_temps):.2f}°C'
    )
    ax.text(
        0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_6_Comfort_Distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure7_price_response(
    baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str
):
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = list(range(24))

    baseline_hourly = [baseline_stats['hourly_power'].get(h, 0) for h in hours]
    ppo_hourly = [ppo_stats['hourly_power'].get(h, 0) for h in hours]

    width = 0.35
    x = np.array(hours)

    ax.bar(x - width/2, baseline_hourly, width,
           label='Baseline', color='red', alpha=0.7)
    ax.bar(x + width/2, ppo_hourly, width,
           label='PI-DRL', color='blue', alpha=0.7)

    ax.axvspan(15.5, 20.5, alpha=0.2, color='orange',
               label='Peak Hours (16:00-21:00)')

    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Average Power (kW)', fontweight='bold')
    ax.set_title('Demand Response: Power Consumption by Hour',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(hours)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    peak_hours = range(16, 21)
    baseline_peak = np.mean(
        [baseline_stats['hourly_power'].get(h, 0) for h in peak_hours]
    )
    ppo_peak = np.mean(
        [ppo_stats['hourly_power'].get(h, 0) for h in peak_hours]
    )

    if baseline_peak > 0:
        reduction = (baseline_peak - ppo_peak) / baseline_peak * 100
        ax.text(
            0.98, 0.98, f'Peak Power Reduction: {reduction:.1f}%',
            transform=ax.transAxes, fontsize=11, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        )

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
    print("PHASE 5: Generating Tables")
    print("=" * 80)

    table1 = pd.DataFrame({
        'Parameter': [
            'R_i (K/W)', 'R_w (K/W)', 'R_o (K/W)',
            'C_in (J/K)', 'C_m (J/K)',
            'Lockout Time (min)', 'Setpoint (°C)', 'Deadband (°C)',
            'Q_HVAC (kW)', 'dt (s)',
            'PPO Learning Rate', 'PPO Gamma', 'PPO gae_lambda', 'PPO clip_range',
            'PPO n_steps', 'PPO batch_size',
            'w_cost', 'w_disc', 'w_switch', 'w_peak', 'w_invalid'
        ],
        'Value': [
            f'{config.R_i:.4f}', f'{config.R_w:.4f}', f'{config.R_o:.4f}',
            f'{config.C_in:.0f}', f'{config.C_m:.0f}',
            f'{config.lockout_time}', f'{config.setpoint}', f'{config.deadband}',
            f'{config.Q_hvac_kw:.1f}', f'{config.dt:.0f}',
            f'{config.learning_rate}', f'{config.gamma}', f'{config.gae_lambda}',
            f'{config.clip_range}',
            f'{config.n_steps}', f'{config.batch_size}',
            f'{config.w_cost}', f'{config.w_disc}', f'{config.w_switch}',
            f'{config.w_peak}', f'{config.w_invalid}'
        ]
    })
    filepath = os.path.join(output_dir, 'Table_1_System_Parameters.csv')
    table1.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")

    # Performance summary with both comfort metrics
    table2 = pd.DataFrame({
        'Metric': [
            'Total Cost ($)',
            'Comfort loss (band, °C²·h)',
            'Comfort loss (setpoint, °C²·h)',
            'Total Cycles',
            'Total Energy (kWh)'
        ],
        'Baseline': [
            f"{baseline_stats['total_cost']:.2f}",
            f"{baseline_stats['total_discomfort_band']:.2f}",
            f"{baseline_stats['total_discomfort_setpoint']:.2f}",
            f"{baseline_stats['n_cycles']}",
            f"{baseline_stats['total_energy_kwh']:.2f}"
        ],
        'PI-DRL': [
            f"{ppo_stats['total_cost']:.2f}",
            f"{ppo_stats['total_discomfort_band']:.2f}",
            f"{ppo_stats['total_discomfort_setpoint']:.2f}",
            f"{ppo_stats['n_cycles']}",
            f"{ppo_stats['total_energy_kwh']:.2f}"
        ],
        'Improvement (%)': [
            f"{(1 - ppo_stats['total_cost'] / max(baseline_stats['total_cost'], 0.001)) * 100:.1f}",
            f"{(1 - ppo_stats['total_discomfort_band'] / max(baseline_stats['total_discomfort_band'], 0.001)) * 100:.1f}",
            f"{(1 - ppo_stats['total_discomfort_setpoint'] / max(baseline_stats['total_discomfort_setpoint'], 0.001)) * 100:.1f}",
            f"{(1 - ppo_stats['n_cycles'] / max(baseline_stats['n_cycles'], 1)) * 100:.1f}",
            f"{(1 - ppo_stats['total_energy_kwh'] / max(baseline_stats['total_energy_kwh'], 0.001)) * 100:.1f}"
        ]
    })
    filepath = os.path.join(output_dir, 'Table_2_Performance_Summary.csv')
    table2.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")

    baseline_peak = baseline_stats.get('avg_peak_power', 0)
    baseline_offpeak = baseline_stats.get('avg_offpeak_power', 0)
    ppo_peak = ppo_stats.get('avg_peak_power', 0)
    ppo_offpeak = ppo_stats.get('avg_offpeak_power', 0)

    peak_reduction = (baseline_peak - ppo_peak) / max(baseline_peak, 0.001) * 100

    table3 = pd.DataFrame({
        'Metric': [
            'Avg Peak Power (kW)',
            'Avg Off-Peak Power (kW)',
            'Peak Reduction (%)'
        ],
        'Baseline': [
            f"{baseline_peak:.3f}",
            f"{baseline_offpeak:.3f}",
            '-'
        ],
        'PI-DRL': [
            f"{ppo_peak:.3f}",
            f"{ppo_offpeak:.3f}",
            f"{peak_reduction:.1f}"
        ]
    })
    filepath = os.path.join(output_dir, 'Table_3_Grid_Impact.csv')
    table3.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")

    train_total = train_mask_stats.get('total_steps', 0)
    train_masked_off = train_mask_stats.get('masked_off', 0)
    train_masked_on = train_mask_stats.get('masked_on', 0)
    train_pct = (train_masked_off + train_masked_on) / max(train_total, 1) * 100

    test_total = test_mask_stats.get('total_steps', 0)
    test_masked_off = test_mask_stats.get('masked_off', 0)
    test_masked_on = test_mask_stats.get('masked_on', 0)
    test_pct = (test_masked_off + test_masked_on) / max(test_total, 1) * 100

    table4 = pd.DataFrame({
        'Phase': ['Training', 'Testing'],
        'Total Timesteps': [train_total, test_total],
        'Masked OFF': [train_masked_off, test_masked_off],
        'Masked ON': [train_masked_on, test_masked_on],
        'Mask Active (%)': [f"{train_pct:.2f}", f"{test_pct:.2f}"]
    })
    filepath = os.path.join(output_dir, 'Table_4_Safety_Shield_Activity.csv')
    table4.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")

    print("All tables generated!")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 80)
    print("ROBUST, SAFETY-CRITICAL 2R2C PI-DRL CONTROLLER (AMPds2 LOCAL)")
    print("=" * 80)

    config = Config()
    set_global_seed(config.seed)

    # Output directory in current working folder
    output_dir = os.path.join(os.getcwd(), "HVAC_PI_DRL_Output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")

    # 1) Load data
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
    test_env = SafetyHVACEnv(
        data=data,
        config=config,
        is_training=False,
        use_domain_randomization=False,
        start_idx=split_idx,
        end_idx=len(data),
    )
    test_env.reset(seed=config.seed)

    print("Environments created.")
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space     : {train_env.action_space}")

    # 3) Curriculum training
    print("\n" + "=" * 80)
    print("PHASE 3: Curriculum Training PPO Agent")
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

    curriculum_phases = [
        (0.0,  config.total_timesteps // 3, "Phase 1: Nominal Parameters"),
        (0.10, config.total_timesteps // 3, "Phase 2: ±10% Randomization"),
        (0.15, config.total_timesteps // 3, "Phase 3: ±15% Randomization"),
    ]

    model: Optional[PPO] = None
    total_trained_steps = 0
    mask_callback = MaskTrackingCallback()

    for rand_scale, phase_steps, phase_name in curriculum_phases:
        print("\n" + "=" * 60)
        print(f"{phase_name}  ({phase_steps:,} timesteps, rand_scale = {rand_scale:.2f})")
        print("=" * 60)

        def make_env(scale=rand_scale, base_seed=config.seed):
            def _init():
                env = SafetyHVACEnv(
                    data=data,
                    config=config,
                    is_training=True,
                    use_domain_randomization=(scale > 0),
                    start_idx=0,
                    end_idx=split_idx,
                    randomization_scale=scale,
                )
                env.reset(seed=base_seed)
                return env
            return _init

        vec_env = DummyVecEnv([make_env()])
        vec_env.seed(config.seed)

        if model is None:
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
                vf_coef=0.25,
                clip_range_vf=0.2,
                max_grad_norm=0.5,
                verbose=1,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])]
                ),
                seed=config.seed,
            )

        else:
            model.set_env(vec_env)

        model.learn(
            total_timesteps=phase_steps,
            callback=mask_callback,
            progress_bar=True,
            reset_num_timesteps=False,
        )

        total_trained_steps += phase_steps
        print(f"Completed {phase_name}. Cumulative steps: {total_trained_steps:,}")

    model_path = os.path.join(output_dir, "ppo_hvac_model")
    model.save(model_path)
    print(f"\nModel saved to: {os.path.abspath(model_path)}")

    train_mask_stats = {
        "total_steps": total_trained_steps,
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
        print(f"  Total cost                     : {stats['total_cost']:.2f} $")
        print(f"  Energy consumption             : {stats['total_energy_kwh']:.2f} kWh")
        print(f"  Comfort loss (band, °C²·h)     : {stats['total_discomfort_band']:.2f}")
        print(f"  Comfort loss (setpoint, °C²·h) : {stats['total_discomfort_setpoint']:.2f}")
        print(f"  Number of cycles               : {stats['n_cycles']}")
        print(f"  Avg peak power (16–21)         : {stats['avg_peak_power']:.3f} kW")
        print(f"  Avg off-peak power             : {stats['avg_offpeak_power']:.3f} kW")

    print_stats("Baseline thermostat", baseline_stats)
    print_stats("PI-DRL agent",       ppo_stats)

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
    print("PI-DRL costs :",  np.round(robustness_results["ppo_costs"], 2))

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

    # 7) Comfort summary
    temps_baseline = np.array(baseline_stats.get("temps", []), dtype=float)
    temps_ppo = np.array(ppo_stats.get("temps", []), dtype=float)

    def frac_in_band(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        m = (arr >= config.comfort_min) & (arr <= config.comfort_max)
        return float(m.mean())

    fb = frac_in_band(temps_baseline)
    fp = frac_in_band(temps_ppo)

    print("\nComfort statistics (fraction of time inside comfort band):")
    print(f"  Baseline : {fb * 100:.1f}%")
    print(f"  PI-DRL   : {fp * 100:.1f}%")

    if fp < 0.90:
        print(
            "\n⚠️  Warning: PI-DRL spends a significant fraction of time "
            f"outside [{config.comfort_min:.1f}°C, {config.comfort_max:.1f}°C]. "
            "Consider increasing w_disc further or adjusting reward structure."
        )
    else:
        print("\n✅  Comfort constraint looks acceptable for PI-DRL (≥ 90% in band).")

    print(f"\nAll outputs written under: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
