# -*- coding: utf-8 -*-
"""
SAFE PI-DRL Controller با Hard Safety Constraint

رویکرد: Safety Layer که اجازه نمی‌دهد agent از comfort خارج شود.

اگر T نزدیک به comfort_min → action=OFF ممنوع
اگر T نزدیک به comfort_max → action=ON ممنوع

Agent فقط در داخل comfort band می‌تواند cost را optimize کند.
"""

import os
from dataclasses import dataclass, replace
from typing import Dict, Tuple, Any, Optional, List
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
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
    deadband: float    = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    lockout_time: int  = 15

    # 🔑 SAFETY MARGINS (کلید موفقیت!)
    safety_margin_low: float = 0.5   # اگر T < 20°C → action=OFF ممنوع
    safety_margin_high: float = 0.5  # اگر T > 23.5°C → action=ON ممنوع

    # TOU tariff
    peak_price:    float = 0.30
    offpeak_price: float = 0.10

    randomization_scale: float = 0.0

    # Training
    total_timesteps:    int = 200_000
    episode_length_days: int = 2
    train_split:        float = 0.8

    # Reward weights - حالا می‌توانند متعادل باشند چون safety layer داریم!
    w_cost: float = 1.0
    w_peak: float = 1.0
    w_unnecessary_on: float = 2.0
    w_switch: float = 1.0
    w_comfort: float = 10.0  # فقط برای fine-tuning در داخل safety zone

    # PPO
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


# [Data loading - same as before, abbreviated]
def load_ampds2_local(config: Config) -> pd.DataFrame:
    """Load AMPds2 - same implementation as balanced version"""
    print("=" * 80)
    print("Loading AMPds2...")
    # [Full implementation needed - copy from balanced version]
    raise NotImplementedError("Copy full data loading from balanced version")


# =========================================================================
# SAFE REWARD HANDLER
# =========================================================================

class SafeRewardHandler:
    """
    Reward ساده چون safety layer اصلی کار را می‌کند
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.setpoint = config.setpoint
        self.deadband = config.deadband
        self.comfort_min = config.comfort_min
        self.comfort_max = config.comfort_max
        
        self.lower_deadband = self.setpoint - self.deadband
        self.upper_deadband = self.setpoint + self.deadband
    
    def calculate(
        self,
        T_in: float,
        action: int,
        price_t: float,
        prev_action: int,
        is_peak: bool,
        current_step_power_kw: float,
        was_overridden: bool
    ) -> Tuple[float, Dict]:
        
        dt_hours = 1.0 / 60.0
        energy_kwh = current_step_power_kw * dt_hours
        instant_cost = energy_kwh * price_t
        
        in_deadband = (self.lower_deadband <= T_in <= self.upper_deadband)
        in_comfort = (self.comfort_min <= T_in <= self.comfort_max)
        
        # ============ PRIMARY: Cost Optimization ============
        cost_term = self.config.w_cost * instant_cost
        
        if is_peak and action == 1:
            peak_penalty = self.config.w_peak * energy_kwh
        else:
            peak_penalty = 0.0
        
        # Unnecessary ON در deadband بالا
        unnecessary_on = 0.0
        if in_deadband and action == 1 and T_in >= self.setpoint:
            unnecessary_on = self.config.w_unnecessary_on
        
        # Switching
        switch_penalty = 0.0
        if action != prev_action:
            switch_penalty = self.config.w_switch
        
        # Comfort (خفیف - فقط برای fine-tuning)
        comfort_penalty = 0.0
        if not in_comfort:
            if T_in < self.comfort_min:
                dev = self.comfort_min - T_in
            else:
                dev = T_in - self.comfort_max
            comfort_penalty = self.config.w_comfort * (dev ** 2)
        
        # Override penalty
        override_penalty = 10.0 if was_overridden else 0.0
        
        # Final reward
        total_penalty = (
            cost_term +
            peak_penalty +
            unnecessary_on +
            switch_penalty +
            comfort_penalty +
            override_penalty
        )
        
        baseline_reward = 1.0 if in_comfort else 0.0
        reward = baseline_reward - total_penalty
        
        components = {
            "cost_term": cost_term,
            "peak_penalty": peak_penalty,
            "unnecessary_on": unnecessary_on,
            "switch_penalty": switch_penalty,
            "comfort_penalty": comfort_penalty,
            "override_penalty": override_penalty,
            "was_overridden": was_overridden,
            "in_comfort": in_comfort,
            "in_deadband": in_deadband,
            "final_reward": reward,
        }
        
        return reward, components


# =========================================================================
# SAFE HVAC ENVIRONMENT با Hard Safety Layer
# =========================================================================

class SafeHVACEnv(gym.Env):
    """
    Environment با Safety Layer که اجازه نمی‌دهد از comfort خارج شویم
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        is_training: bool = True,
        use_domain_randomization: bool = True,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        super().__init__()
        self.data = data
        self.config = config
        self.is_training = is_training
        self.use_domain_randomization = use_domain_randomization and is_training

        self.reward_handler = SafeRewardHandler(config)

        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx   = end_idx   if end_idx   is not None else len(data)
        self.episode_length = config.episode_length_days * 24 * 60

        # obs: [T_in_norm, T_out_norm, T_mass_norm, price_norm, time_sin, time_cos, 
        #       safety_mask_off, safety_mask_on, lockout_mask_off, lockout_mask_on]
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -1.5, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.5,  1.5,  1.5, 1.5,  1.0,  1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)

        self.T_in = config.setpoint
        self.T_mass = config.setpoint
        self.current_step = 0
        self.episode_start_idx = 0

        # Safety & lockout
        self.minutes_since_on = config.lockout_time
        self.minutes_since_off = config.lockout_time
        self.current_state = 0

        self.runtime = config.lockout_time
        self.offtime = config.lockout_time
        self.last_action = 0
        self.prev_action = 0

        # Statistics
        self.safety_overrides = 0
        self.lockout_overrides = 0
        self.episode_actions: List[int] = []
        self.episode_temps: List[float] = []
        self.episode_costs: List[float] = []
        self.episode_power: List[float] = []
        self.on_runtimes: List[int] = []
        self._current_on_duration = 0

        # Thermal parameters
        self.R_i = config.R_i
        self.R_w = config.R_w
        self.R_o = config.R_o
        self.C_in = config.C_in
        self.C_m  = config.C_m

    def get_safety_mask(self) -> np.ndarray:
        """
        🔑 SAFETY LAYER: جلوگیری از خروج از comfort band
        
        اگر T نزدیک به comfort_min → action=OFF ممنوع
        اگر T نزدیک به comfort_max → action=ON ممنوع
        """
        allowed = np.array([True, True], dtype=bool)
        
        # Safety threshold
        safe_min = self.config.comfort_min + self.config.safety_margin_low
        safe_max = self.config.comfort_max - self.config.safety_margin_high
        
        if self.T_in < safe_min:
            # خیلی سرد است → action=OFF ممنوع!
            allowed[0] = False
            
        if self.T_in > safe_max:
            # خیلی گرم است → action=ON ممنوع!
            allowed[1] = False
        
        return allowed

    def get_lockout_mask(self) -> np.ndarray:
        """Lockout safety (مثل قبل)"""
        allowed = np.array([True, True], dtype=bool)
        if self.current_state == 1:
            if self.minutes_since_on < self.config.lockout_time:
                allowed[0] = False
        else:
            if self.minutes_since_off < self.config.lockout_time:
                allowed[1] = False
        return allowed

    def get_combined_mask(self) -> np.ndarray:
        """ترکیب safety + lockout masks"""
        safety_mask = self.get_safety_mask()
        lockout_mask = self.get_lockout_mask()
        return safety_mask & lockout_mask

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        # Domain randomization (if enabled)
        if self.use_domain_randomization and self.config.randomization_scale > 0:
            s = self.config.randomization_scale
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

        # Initialize temperature (در comfort band)
        self.T_in = self.config.setpoint + np.random.uniform(-0.5, 0.5)
        self.T_mass = self.T_in - 0.5

        self.minutes_since_on = self.config.lockout_time
        self.minutes_since_off = self.config.lockout_time
        self.current_state = 0

        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time
        self.last_action = 0
        self.prev_action = 0

        self.safety_overrides = 0
        self.lockout_overrides = 0
        self.episode_actions = []
        self.episode_temps = []
        self.episode_costs = []
        self.episode_power = []
        self.on_runtimes = []
        self._current_on_duration = 0
        self.current_step = 0

        obs = self._get_observation()
        safe_idx = min(self.episode_start_idx, len(self.data) - 1)
        info = {
            "T_in_true": self.T_in,
            "T_out": self.data.iloc[safe_idx]["T_out"],
            "action_mask": self.get_combined_mask()
        }
        return obs, info

    def step(self, action: int):
        data_idx = min(self.episode_start_idx + self.current_step, len(self.data) - 1)
        row = self.data.iloc[data_idx]
        T_out = row["T_out"]
        price_t = row["Price"]
        current_hour = self.data.index[data_idx].hour

        # 🔑 APPLY SAFETY + LOCKOUT MASKS
        combined_mask = self.get_combined_mask()
        original_action = action
        was_overridden = False

        if not combined_mask[action]:
            # Action not allowed! Override it.
            if combined_mask[0]:
                action = 0
            elif combined_mask[1]:
                action = 1
            else:
                # Both actions forbidden (should not happen)
                action = self.last_action
            
            was_overridden = True
            
            # Check which mask caused override
            safety_mask = self.get_safety_mask()
            lockout_mask = self.get_lockout_mask()
            
            if not safety_mask[original_action]:
                self.safety_overrides += 1
            if not lockout_mask[original_action]:
                self.lockout_overrides += 1

        # Update state tracking
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

        # Reward
        is_peak = 16 <= current_hour < 21
        current_power_kw = action * self.config.Q_hvac_kw

        reward, r_comp = self.reward_handler.calculate(
            T_in=self.T_in,
            action=action,
            price_t=price_t,
            prev_action=self.prev_action,
            is_peak=is_peak,
            current_step_power_kw=current_power_kw,
            was_overridden=was_overridden
        )
        self.prev_action = action

        # Episode stats
        self.episode_actions.append(action)
        self.episode_temps.append(self.T_in)
        self.episode_costs.append(r_comp["cost_term"])
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
            "was_overridden": was_overridden,
            "power_kw": current_power_kw,
            "cost": r_comp["cost_term"],
            "price": price_t,
            "action_mask": self.get_combined_mask(),
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

        # Masks
        safety_mask = self.get_safety_mask()
        lockout_mask = self.get_lockout_mask()
        
        safety_mask_off = 1.0 if safety_mask[0] else 0.0
        safety_mask_on  = 1.0 if safety_mask[1] else 0.0
        lockout_mask_off = 1.0 if lockout_mask[0] else 0.0
        lockout_mask_on  = 1.0 if lockout_mask[1] else 0.0

        return np.array(
            [T_in_norm, T_out_norm, T_mass_norm, price_norm,
             row["time_sin"], row["time_cos"],
             safety_mask_off, safety_mask_on,
             lockout_mask_off, lockout_mask_on],
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
            "safety_overrides": self.safety_overrides,
            "lockout_overrides": self.lockout_overrides,
            "on_runtimes": self.on_runtimes.copy(),
            "temps": temps.copy(),
            "actions": self.episode_actions.copy(),
            "power": self.episode_power.copy(),
        }


# [Baseline thermostat, evaluation, main - same as balanced version]
# [Abbreviated for space - full implementation needed]

def main():
    print("=" * 80)
    print("SAFE PI-DRL Controller با Hard Safety Layer")
    print("=" * 80)
    print("\n🔑 Key Innovation: Safety Layer")
    print("  - اگر T < 20°C → action=OFF ممنوع")
    print("  - اگر T > 23.5°C → action=ON ممنوع")
    print("  → Agent فقط در داخل comfort می‌تواند cost را optimize کند")
    
    config = Config()
    set_global_seed(config.seed)
    
    print("\n⚠️  NOTE: Data loading needed - copy from balanced version")
    print("Then run full training & evaluation")


if __name__ == "__main__":
    main()
