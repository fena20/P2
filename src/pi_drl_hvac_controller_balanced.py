# -*- coding: utf-8 -*-
"""
BALANCED 2R2C PI-DRL Controller - نسخه متعادل و واقع‌بینانه

هدف: بهتر از baseline در همه جنبه‌ها:
- Comfort خوب (نه بدتر از baseline)
- Cost کمتر از baseline
- Cycling معقول (نه صفر، نه زیاد)

استراتژی:
1. Comfort مهم است اما نه به قیمت Always ON
2. Agent باید یاد بگیرد در deadband مثل thermostat رفتار کند
3. Cost optimization فقط وقتی که comfort تضمین شده
4. Temperature deadband را توسط reward تقویت کنیم

Author: Fixed by AI Assistant
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
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3
})

@dataclass
class Config:
    data_dir: str = r"C:\Users\FATEME\Downloads\dataverse_files"
    seed: int = 42

    # 2R2C parameters
    R_i: float = 0.0005
    R_w: float = 0.003
    R_o: float = 0.002
    C_in: float = 1_000_000.0
    C_m:  float = 4_000_000.0

    # HVAC
    Q_hvac_max: float = 3000.0
    Q_hvac_kw:  float = 3.0
    dt: float = 60.0

    # Comfort settings
    setpoint: float    = 21.0
    deadband: float    = 1.5      # مثل baseline: [19.5, 22.5]
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    lockout_time: int  = 15

    # TOU tariff
    peak_price:    float = 0.30
    offpeak_price: float = 0.10

    # Randomization
    randomization_scale: float = 0.0

    # Training
    total_timesteps:    int = 200_000  # افزایش timesteps
    episode_length_days: int = 2       # episodes بلندتر برای یادگیری بهتر
    train_split:        float = 0.8

    # ========== BALANCED REWARD WEIGHTS ==========
    # استراتژی: Comfort مهم اما متعادل
    w_comfort_violation: float = 50.0   # جریمه شدید برای خروج از comfort band
    w_temp_deviation: float = 2.0       # جریمه متوسط برای دوری از setpoint در deadband
    w_cost: float = 1.0                 # هزینه مهم است
    w_unnecessary_on: float = 5.0       # جریمه برای ON بودن وقتی نیازی نیست
    w_peak: float = 2.0                 # Peak shaving مهم است
    w_switch: float = 0.1               # Cycling کم اهمیت است
    w_invalid: float = 20.0

    # PPO hyperparameters - تنظیمات محافظه‌کارانه
    learning_rate: float = 3e-4
    gamma:         float = 0.98        # کمی کوتاه‌مدت‌تر
    gae_lambda:    float = 0.95
    n_steps:       int   = 2048
    batch_size:    int   = 64
    n_epochs:      int   = 10
    ent_coef:      float = 0.02        # exploration بیشتر
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


def load_ampds2_local(config: Config) -> pd.DataFrame:
    """Same as before - data loading"""
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
            raise FileNotFoundError(f"Required file not found: {p}")

    df_weather = pd.read_csv(path_weather)
    if "Date/Time" not in df_weather.columns:
        raise RuntimeError("Climate_HourlyWeather.csv must contain 'Date/Time' column.")
    df_weather["datetime"] = pd.to_datetime(df_weather["Date/Time"])
    df_weather = df_weather.set_index("datetime").sort_index()

    temp_col = None
    for c in df_weather.columns:
        if c == "Temp (C)" or "temp" in c.lower():
            temp_col = c
            break
    if temp_col is None:
        raise RuntimeError("Could not find temperature column")

    df_weather = df_weather[[temp_col]].copy()
    df_weather = df_weather.resample("1min").interpolate("time").ffill().bfill()

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

    start = max(df_weather.index.min(), df_whe.index.min(), df_hpe.index.min())
    end   = min(df_weather.index.max(), df_whe.index.max(), df_hpe.index.max())
    if start >= end:
        raise RuntimeError("No overlapping time range")

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
# BALANCED REWARD HANDLER - کلید اصلی موفقیت
# =========================================================================

class BalancedRewardHandler:
    """
    Reward function متعادل که مثل ترموستات هوشمند عمل می‌کند:
    
    1. اگر دما در deadband [setpoint-deadband, setpoint+deadband] است:
       - می‌تواند OFF باشد (برای صرفه‌جویی)
       - جریمه‌ی خفیف برای دوری از setpoint
       
    2. اگر دما خارج deadband اما داخل comfort band است:
       - باید به setpoint برگردد
       - جریمه متوسط
       
    3. اگر دما خارج comfort band است:
       - جریمه شدید
       - اورژانسی!
    """

    def __init__(self, config: Config):
        self.config = config
        self.setpoint = config.setpoint
        self.deadband = config.deadband
        self.comfort_min = config.comfort_min
        self.comfort_max = config.comfort_max
        
        # محدوده deadband - مثل thermostat
        self.lower_deadband = self.setpoint - self.deadband  # 19.5
        self.upper_deadband = self.setpoint + self.deadband  # 22.5

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

        # ============ تحلیل وضعیت دما ============
        in_deadband = (self.lower_deadband <= T_in <= self.upper_deadband)
        in_comfort_band = (self.comfort_min <= T_in <= self.comfort_max)
        
        # ============ 1. COMFORT PENALTY با Action-Aware Logic ============
        comfort_penalty = 0.0
        
        if not in_comfort_band:
            # خارج از comfort band → اورژانسی!
            if T_in < self.comfort_min:
                violation = self.comfort_min - T_in
                # اگر سرد است و OFF است → جریمه بیشتر!
                if action == 0:
                    comfort_penalty = self.config.w_comfort_violation * (violation ** 3)  # cubic!
                else:
                    comfort_penalty = self.config.w_comfort_violation * (violation ** 2)  # quadratic
            else:
                violation = T_in - self.comfort_max
                comfort_penalty = self.config.w_comfort_violation * (violation ** 2)
            
        elif not in_deadband:
            # در comfort band اما خارج deadband → برگرد به deadband
            if T_in < self.lower_deadband:
                deviation = self.lower_deadband - T_in
                # اگر سرد است و OFF است → جریمه بیشتر
                if action == 0:
                    comfort_penalty = self.config.w_temp_deviation * (deviation ** 3)  # cubic!
                else:
                    comfort_penalty = self.config.w_temp_deviation * (deviation ** 2)
            else:
                deviation = T_in - self.upper_deadband
                comfort_penalty = self.config.w_temp_deviation * (deviation ** 2)
        
        # ============ 2. UNNECESSARY ON PENALTY ============
        # اگر در deadband هستیم و ON هستیم، جریمه بده
        unnecessary_on_penalty = 0.0
        if in_deadband and action == 1:
            # فقط اگر دما بالای setpoint است، ON بودن غیرضروری است
            if T_in >= self.setpoint:
                unnecessary_on_penalty = self.config.w_unnecessary_on
        
        # ============ 3. COST TERM ============
        cost_term = self.config.w_cost * instant_cost
        
        # ============ 4. PEAK SHAVING ============
        peak_penalty = 0.0
        if is_peak and action == 1:
            # فقط اگر راحت هستیم، peak penalty بده
            if in_deadband:
                peak_penalty = self.config.w_peak * energy_kwh
        
        # ============ 5. SWITCHING PENALTY ============
        switch_penalty = 0.0
        if action != prev_action:
            switch_penalty = self.config.w_switch
        
        # ============ 6. INVALID ACTION ============
        invalid_penalty = self.config.w_invalid if is_invalid else 0.0
        
        # ============ TOTAL REWARD ============
        total_penalty = (
            comfort_penalty +
            unnecessary_on_penalty +
            cost_term +
            peak_penalty +
            switch_penalty +
            invalid_penalty
        )
        
        # Baseline reward: اگر در comfort band هستیم
        baseline_reward = 1.0 if in_comfort_band else 0.0
        
        reward = baseline_reward - total_penalty
        
        # Scale reasonably
        reward = np.clip(reward, -100.0, 10.0)

        components = {
            "comfort_penalty": comfort_penalty,
            "unnecessary_on": unnecessary_on_penalty,
            "cost_term": cost_term,
            "peak_penalty": peak_penalty,
            "switch_penalty": switch_penalty,
            "invalid_penalty": invalid_penalty,
            "baseline_reward": baseline_reward,
            "raw_cost": instant_cost,
            "raw_discomfort": comfort_penalty,
            "in_deadband": 1.0 if in_deadband else 0.0,
            "in_comfort": 1.0 if in_comfort_band else 0.0,
            "total_penalty": total_penalty,
            "final_reward": reward,
        }

        return reward, components


# =========================================================================
# ENVIRONMENT - همان قبل با reward handler جدید
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

        self.reward_handler = BalancedRewardHandler(config)

        self.randomization_scale = (
            randomization_scale if randomization_scale is not None
            else config.randomization_scale
        )

        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx   = end_idx   if end_idx   is not None else len(data)
        self.episode_length = config.episode_length_days * 24 * 60

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

        self.minutes_since_on = config.lockout_time
        self.minutes_since_off = config.lockout_time
        self.current_state = 0

        self.runtime = config.lockout_time
        self.offtime = config.lockout_time
        self.last_action = 0
        self.prev_action = 0

        self.masked_off_count = 0
        self.masked_on_count = 0
        self.episode_actions: List[int] = []
        self.episode_temps: List[float] = []
        self.episode_costs: List[float] = []
        self.episode_power: List[float] = []
        self.on_runtimes: List[int] = []
        self._current_on_duration = 0

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

        Q_hvac = action * self.config.Q_hvac_max
        Q_im = (self.T_in - self.T_mass) / self.R_i
        Q_mo = (self.T_mass - T_out) / (self.R_w + self.R_o)

        dT_in = (Q_hvac - Q_im) / self.C_in * self.config.dt
        self.T_in += dT_in

        dT_mass = (Q_im - Q_mo) / self.C_m * self.config.dt
        self.T_mass += dT_mass

        self.T_in = np.clip(self.T_in, 10.0, 35.0)
        self.T_mass = np.clip(self.T_mass, 10.0, 35.0)

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
            "in_deadband": r_comp["in_deadband"],
            "in_comfort": r_comp["in_comfort"],
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

        disc_set = sum((t - self.config.setpoint) ** 2 for t in temps) / 60.0

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


def main():
    print("=" * 80)
    print("BALANCED PI-DRL CONTROLLER - واقع‌بینانه و متعادل")
    print("=" * 80)

    config = Config()
    set_global_seed(config.seed)

    output_dir = os.path.join(os.getcwd(), "HVAC_PI_DRL_Output_Balanced")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")

    data = load_ampds2_local(config)
    split_idx = int(len(data) * config.train_split)

    print("\nData split:")
    print(f"  Training: {len(data[:split_idx])} samples ({len(data[:split_idx])/(60*24):.1f} days)")
    print(f"  Testing : {len(data[split_idx:])} samples ({len(data[split_idx:])/(60*24):.1f} days)")

    print("\n" + "=" * 80)
    print("PHASE 2: Training with BALANCED reward function")
    print("=" * 80)
    print("\nReward strategy:")
    print("  - Comfort violations: Heavy penalty")
    print("  - Unnecessary ON in deadband: Moderate penalty")
    print("  - Cost optimization: Always active")
    print("  - Peak shaving: Active when comfortable")
    print("  → Goal: Beat baseline in ALL metrics")

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
        (0.0,  config.total_timesteps // 2, "Phase 1: Learn basic thermostat behavior"),
        (0.10, config.total_timesteps // 2, "Phase 2: Robustness with randomization"),
    ]

    model: Optional[PPO] = None
    total_trained_steps = 0
    mask_callback = MaskTrackingCallback()

    for rand_scale, phase_steps, phase_name in curriculum_phases:
        print("\n" + "=" * 60)
        print(f"{phase_name}  ({phase_steps:,} timesteps)")
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
                vf_coef=0.5,
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
        print(f"Completed {phase_name}. Total: {total_trained_steps:,}")

    model_path = os.path.join(output_dir, "ppo_hvac_balanced")
    model.save(model_path)
    print(f"\nModel saved to: {os.path.abspath(model_path)}")

    print("\n" + "=" * 80)
    print("PHASE 3: Evaluation")
    print("=" * 80)

    baseline = BaselineThermostat(config)

    print("\nEvaluating Baseline...")
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

    print("Evaluating PI-DRL...")
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

    def print_stats(name: str, stats: Dict):
        print(f"\n{name}:")
        print(f"  Total cost                     : ${stats['total_cost']:.2f}")
        print(f"  Energy consumption             : {stats['total_energy_kwh']:.2f} kWh")
        print(f"  Comfort loss (band, °C²·h)     : {stats['total_discomfort_band']:.2f}")
        print(f"  Comfort loss (setpoint, °C²·h) : {stats['total_discomfort_setpoint']:.2f}")
        print(f"  Number of cycles               : {stats['n_cycles']}")
        print(f"  Avg peak power (16–21)         : {stats['avg_peak_power']:.3f} kW")
        print(f"  Avg off-peak power             : {stats['avg_offpeak_power']:.3f} kW")

    print_stats("Baseline thermostat", baseline_stats)
    print_stats("PI-DRL agent",       ppo_stats)

    print("\n" + "=" * 80)
    print("COMPARISON & IMPROVEMENTS")
    print("=" * 80)

    cost_improvement = (1 - ppo_stats['total_cost'] / baseline_stats['total_cost']) * 100
    comfort_improvement = (1 - ppo_stats['total_discomfort_band'] / baseline_stats['total_discomfort_band']) * 100
    energy_improvement = (1 - ppo_stats['total_energy_kwh'] / baseline_stats['total_energy_kwh']) * 100
    peak_reduction = (1 - ppo_stats['avg_peak_power'] / max(baseline_stats['avg_peak_power'], 0.001)) * 100

    print(f"\nCost improvement        : {cost_improvement:+.1f}%")
    print(f"Comfort improvement     : {comfort_improvement:+.1f}%")
    print(f"Energy improvement      : {energy_improvement:+.1f}%")
    print(f"Peak power reduction    : {peak_reduction:+.1f}%")
    print(f"\nCycles - Baseline: {baseline_stats['n_cycles']}, PI-DRL: {ppo_stats['n_cycles']}")

    # Comfort band analysis
    temps_baseline = np.array(baseline_stats.get("temps", []), dtype=float)
    temps_ppo = np.array(ppo_stats.get("temps", []), dtype=float)

    def frac_in_band(arr: np.ndarray, low: float, high: float) -> float:
        if arr.size == 0:
            return 0.0
        return float(((arr >= low) & (arr <= high)).mean())

    fb_comfort = frac_in_band(temps_baseline, config.comfort_min, config.comfort_max)
    fp_comfort = frac_in_band(temps_ppo, config.comfort_min, config.comfort_max)
    
    fb_dead = frac_in_band(temps_baseline, config.setpoint - config.deadband, 
                           config.setpoint + config.deadband)
    fp_dead = frac_in_band(temps_ppo, config.setpoint - config.deadband, 
                           config.setpoint + config.deadband)

    print("\n" + "=" * 80)
    print("COMFORT ANALYSIS")
    print("=" * 80)
    print(f"\nTime in comfort band [{config.comfort_min}°C - {config.comfort_max}°C]:")
    print(f"  Baseline : {fb_comfort * 100:.1f}%")
    print(f"  PI-DRL   : {fp_comfort * 100:.1f}%")
    
    print(f"\nTime in deadband [{config.setpoint - config.deadband}°C - {config.setpoint + config.deadband}°C]:")
    print(f"  Baseline : {fb_dead * 100:.1f}%")
    print(f"  PI-DRL   : {fp_dead * 100:.1f}%")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    improvements = {
        "Cost": cost_improvement > 0,
        "Comfort": comfort_improvement > 0,
        "Energy": energy_improvement > 0,
        "Peak": peak_reduction > 0,
    }
    
    num_improvements = sum(improvements.values())
    
    if num_improvements >= 3:
        print("\n✅ SUCCESS! PI-DRL beats baseline in majority of metrics")
    elif num_improvements >= 2:
        print("\n⚠️  PARTIAL SUCCESS: PI-DRL beats baseline in some metrics")
    else:
        print("\n❌ FAILURE: PI-DRL does not beat baseline")
        print("\nSuggested fixes:")
        if not improvements["Comfort"]:
            print("  - Increase w_comfort_violation")
        if not improvements["Cost"]:
            print("  - Increase w_cost or w_peak")
        if ppo_stats['n_cycles'] == 0:
            print("  - Reduce w_unnecessary_on (agent is stuck ON)")

    print(f"\nAll outputs in: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
