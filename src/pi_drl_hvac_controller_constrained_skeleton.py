# -*- coding: utf-8 -*-
"""
CONSTRAINED PI-DRL Controller با Comfort Constraint سخت

رویکرد جدید: به جای tuning weights، یک constraint سخت تعریف می‌کنیم:
  "Agent باید حداقل 90% زمان را داخل comfort band باشد"

این رویکرد مشابه Safe RL و Constrained MDP است.
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
    deadband: float    = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    lockout_time: int  = 15

    # TOU tariff
    peak_price:    float = 0.30
    offpeak_price: float = 0.10

    randomization_scale: float = 0.0

    # Training
    total_timesteps:    int = 300_000
    episode_length_days: int = 3
    train_split:        float = 0.8

    # ========== CONSTRAINED APPROACH ==========
    # هدف: minimize cost
    # Constraint: time_in_comfort >= 90%
    
    # Primary objective
    w_cost: float = 1.0
    w_peak: float = 0.5
    w_switch: float = 1.0
    
    # Lagrangian multiplier (تطبیقی)
    lambda_comfort: float = 100.0  # شروع با مقدار بالا
    lambda_update_rate: float = 1.1  # ضریب افزایش
    
    # Constraint thresholds
    comfort_threshold: float = 0.90  # حداقل 90% باید در comfort باشد
    max_cycles_per_day: int = 200    # حداکثر 200 سیکل در روز

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


# Load data function (same as before - abbreviated for space)
def load_ampds2_local(config: Config) -> pd.DataFrame:
    """Load AMPds2 data - same as balanced version"""
    print("=" * 80)
    print("Loading AMPds2...")
    print("=" * 80)
    
    base = config.data_dir
    path_weather = os.path.join(base, "Climate_HourlyWeather.csv")
    path_whe     = os.path.join(base, "Electricity_WHE.csv")
    path_hpe     = os.path.join(base, "Electricity_HPE.csv")
    
    for p in [path_weather, path_whe, path_hpe]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    
    # [Full implementation same as balanced version]
    # Abbreviated here for brevity
    
    # Simplified placeholder
    print("⚠️  Using placeholder - implement full data loading")
    dates = pd.date_range('2024-01-01', periods=60*24*30, freq='1min')
    df = pd.DataFrame({
        'T_out': 10 + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / (60*24)) + np.random.randn(len(dates)) * 2,
        'WHE': np.random.rand(len(dates)) * 2,
        'HPE': np.random.rand(len(dates)) * 1.5,
        'hour': dates.hour,
    }, index=dates)
    df['Price'] = df['hour'].apply(lambda h: config.peak_price if 16 <= h < 21 else config.offpeak_price)
    df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


# =========================================================================
# CONSTRAINED REWARD HANDLER
# =========================================================================

class ConstrainedRewardHandler:
    """
    رویکرد Lagrangian:
    
    L(θ, λ) = Cost - λ * (comfort_violation)
    
    λ به‌صورت تطبیقی update می‌شود:
      - اگر comfort constraint نقض شود → λ افزایش
      - اگر comfort OK باشد → λ کاهش (یا ثابت)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.comfort_min = config.comfort_min
        self.comfort_max = config.comfort_max
        self.setpoint = config.setpoint
        self.deadband = config.deadband
        
        self.lower_deadband = self.setpoint - self.deadband
        self.upper_deadband = self.setpoint + self.deadband
        
        # Lagrangian multiplier (این را در training update می‌کنیم)
        self.lambda_comfort = config.lambda_comfort
        
        # Statistics برای tracking constraint
        self.comfort_violations = []
        self.total_steps = 0
        self.comfort_steps = 0
    
    def calculate(
        self,
        T_in: float,
        action: int,
        price_t: float,
        prev_action: int,
        is_peak: bool,
        current_step_power_kw: float
    ) -> Tuple[float, Dict]:
        
        dt_hours = 1.0 / 60.0
        energy_kwh = current_step_power_kw * dt_hours
        instant_cost = energy_kwh * price_t
        
        in_comfort = (self.comfort_min <= T_in <= self.comfort_max)
        in_deadband = (self.lower_deadband <= T_in <= self.upper_deadband)
        
        # Track comfort
        self.total_steps += 1
        if in_comfort:
            self.comfort_steps += 1
        
        # ============ PRIMARY OBJECTIVE: Minimize Cost ============
        cost_term = self.config.w_cost * instant_cost
        
        if is_peak and action == 1:
            peak_penalty = self.config.w_peak * energy_kwh
        else:
            peak_penalty = 0.0
        
        switch_penalty = 0.0
        if action != prev_action:
            switch_penalty = self.config.w_switch
        
        # ============ CONSTRAINT: Comfort Violation ============
        # این ترم با λ ضرب می‌شود (Lagrangian)
        comfort_violation = 0.0
        
        if not in_comfort:
            # خارج comfort = نقض constraint
            if T_in < self.comfort_min:
                dev = self.comfort_min - T_in
            else:
                dev = T_in - self.comfort_max
            
            # Penalty متناسب با شدت نقض
            comfort_violation = dev ** 2
        
        elif not in_deadband:
            # نزدیک شدن به مرز comfort
            if T_in < self.lower_deadband:
                dev = self.lower_deadband - T_in
            else:
                dev = T_in - self.upper_deadband
            comfort_violation = 0.1 * (dev ** 2)  # خفیف‌تر
        
        # ============ LAGRANGIAN REWARD ============
        # reward = -cost - λ * comfort_violation
        # (منفی چون PPO maximize می‌کند)
        
        primary_objective = -(cost_term + peak_penalty + switch_penalty)
        constraint_penalty = self.lambda_comfort * comfort_violation
        
        reward = primary_objective - constraint_penalty
        
        # Bonus برای بودن در comfort
        if in_comfort:
            reward += 0.5
        
        components = {
            "cost_term": cost_term,
            "peak_penalty": peak_penalty,
            "switch_penalty": switch_penalty,
            "comfort_violation": comfort_violation,
            "lambda_comfort": self.lambda_comfort,
            "constraint_penalty": constraint_penalty,
            "primary_objective": primary_objective,
            "in_comfort": 1.0 if in_comfort else 0.0,
            "final_reward": reward,
        }
        
        return reward, components
    
    def get_comfort_ratio(self) -> float:
        """نسبت زمان در comfort band"""
        if self.total_steps == 0:
            return 1.0
        return self.comfort_steps / self.total_steps
    
    def update_lambda(self):
        """Update Lagrangian multiplier based on constraint satisfaction"""
        comfort_ratio = self.get_comfort_ratio()
        
        if comfort_ratio < self.config.comfort_threshold:
            # Constraint نقض شده → λ را افزایش بده
            self.lambda_comfort *= self.config.lambda_update_rate
            print(f"  ⚠️  Comfort ratio {comfort_ratio:.1%} < {self.config.comfort_threshold:.1%}")
            print(f"  → Increasing λ to {self.lambda_comfort:.1f}")
        else:
            # Constraint OK
            print(f"  ✅ Comfort ratio {comfort_ratio:.1%} >= {self.config.comfort_threshold:.1%}")
        
        # Reset counters
        self.total_steps = 0
        self.comfort_steps = 0


# [Environment implementation - similar to balanced version with constrained reward]
# [Abbreviated here - would be full implementation]

def main():
    print("=" * 80)
    print("CONSTRAINED PI-DRL CONTROLLER")
    print("=" * 80)
    print("\nApproach: Lagrangian/Constrained RL")
    print("  Primary objective: Minimize cost")
    print("  Hard constraint: >=90% time in comfort band")
    print("  λ (Lagrangian multiplier) updates adaptively")
    
    config = Config()
    set_global_seed(config.seed)
    
    print("\n⚠️  NOTE: This is a skeleton implementation")
    print("Full implementation requires:")
    print("  1. Complete data loading")
    print("  2. Environment with ConstrainedRewardHandler")
    print("  3. Callback to update λ periodically")
    print("  4. Training loop with constraint monitoring")
    
    print("\n📚 For immediate use, please use:")
    print("  src/pi_drl_hvac_controller_balanced.py")
    print("  with updated AGGRESSIVE weights")


if __name__ == "__main__":
    main()
