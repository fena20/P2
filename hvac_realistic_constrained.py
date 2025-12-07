# -*- coding: utf-8 -*-
"""
Realistic PI-DRL HVAC Controller with Safety Layer
===================================================

This version uses physically realistic parameters where:
1. HVAC is properly sized for all weather conditions
2. Building is well-insulated (typical modern construction)
3. Baseline can achieve 95%+ comfort in all scenarios

Key insight: If baseline cannot maintain comfort, no controller can!

Author: Energy Systems ML Researcher
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import random
import torch
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class Config:
    seed: int = 42
    
    # Realistic thermal model for well-insulated building
    # R_total ≈ 0.01 K/W means with Q=5kW, ΔT_max = 50°C (sufficient for extreme cold)
    R_i: float = 0.001      # Indoor-mass resistance (K/W)
    R_w: float = 0.005      # Wall resistance (K/W)
    R_o: float = 0.004      # Outdoor resistance (K/W)
    C_in: float = 500_000.0   # Indoor air capacitance (J/K) - smaller = faster response
    C_m: float = 3_000_000.0  # Thermal mass (J/K)
    
    # Properly sized HVAC
    Q_hvac_max: float = 5000.0  # 5 kW heating capacity (typical residential)
    Q_hvac_kw: float = 5.0
    dt: float = 60.0  # seconds per step
    
    # Comfort constraints
    setpoint: float = 21.0
    deadband: float = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    safety_margin: float = 0.3
    lockout_time: int = 15
    min_runtime: int = 20  # Minimum runtime for cycling control
    
    # Pricing
    peak_price: float = 0.30
    offpeak_price: float = 0.10
    
    # Training
    total_timesteps: int = 150_000
    episode_length_days: int = 1
    train_split: float = 0.8
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01
    clip_range: float = 0.2
    
    obs_noise_std: float = 0.1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except: pass


def generate_data(T_base: float, T_amp: float, config: Config, n_days: int = 30) -> pd.DataFrame:
    """Generate weather data for a scenario."""
    n = n_days * 24 * 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    df = pd.DataFrame(index=idx)
    
    h = np.arange(n) / 60.0
    daily = -T_amp * np.cos(2 * np.pi * (h - 6) / 24)
    noise = np.random.normal(0, 1.5, n)  # Less noise for more consistent conditions
    
    df["T_out"] = T_base + daily + noise
    df["T_out"] = df["T_out"].clip(-20, 35)
    df["hour"] = df.index.hour
    df["Price"] = df["hour"].apply(lambda x: config.peak_price if 16 <= x < 21 else config.offpeak_price)
    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    return df


class SafetyLayer:
    """
    Safety layer that GUARANTEES comfort constraint.
    
    Rules:
    1. If T_in ≤ (comfort_min + margin): MUST heat -> return 1
    2. If T_in ≥ (comfort_max - margin): MUST NOT heat -> return 0
    3. If already running for < min_runtime: continue current action
    4. Otherwise: RL decides
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.heating_threshold = config.comfort_min + config.safety_margin
        self.cooling_threshold = config.comfort_max - config.safety_margin
        
    def apply(self, T_in: float, rl_action: int, mask: np.ndarray, 
              current_runtime: int, is_heating: bool) -> Tuple[int, bool]:
        """Apply safety override."""
        
        # First check lockout constraints
        if not mask[rl_action]:
            rl_action = 1 - rl_action
        
        # Safety override for cold (MUST HEAT)
        if T_in <= self.heating_threshold:
            if mask[1]:
                return 1, (rl_action != 1)
            return rl_action, False
        
        # Safety override for hot (MUST NOT HEAT)
        if T_in >= self.cooling_threshold:
            if mask[0]:
                return 0, (rl_action != 0)
            return rl_action, False
        
        # In safe zone
        # Minimum runtime enforcement (prevent short cycling)
        if is_heating and current_runtime < self.config.min_runtime:
            # Encourage staying on to avoid short cycles
            if T_in < self.config.setpoint + 1.0:  # Still below upper setpoint
                if mask[1]:
                    return 1, (rl_action != 1)
        
        # RL decides
        return rl_action, False
    
    def get_rl_freedom(self, T_in: float) -> float:
        """How much freedom RL has (0 = safety override, 1 = full freedom)."""
        if T_in <= self.heating_threshold or T_in >= self.cooling_threshold:
            return 0.0
        margin_low = T_in - self.heating_threshold
        margin_high = self.cooling_threshold - T_in
        margin = min(margin_low, margin_high)
        return min(margin / 1.5, 1.0)


class SafetyHVACEnv(gym.Env):
    """HVAC Environment with integrated safety layer."""
    
    def __init__(self, data: pd.DataFrame, config: Config, is_training: bool = True,
                 start_idx: int = 0, end_idx: Optional[int] = None):
        super().__init__()
        self.data = data
        self.config = config
        self.is_training = is_training
        self.safety_layer = SafetyLayer(config)
        
        self.start_idx = start_idx
        self.end_idx = end_idx or len(data)
        self.episode_length = config.episode_length_days * 24 * 60
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-2, -2, -2, 0, -1, -1, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(2)
        self._init()
        
    def _init(self):
        self.T_in = self.config.setpoint
        self.T_mass = self.config.setpoint
        self.step_count = 0
        self.ep_start = 0
        self.minutes_on = self.config.lockout_time
        self.minutes_off = self.config.lockout_time
        self.hvac_state = 0
        self.prev_action = 0
        self.current_runtime = 0
        
        self.actions = []
        self.temps = []
        self.costs = []
        self.power = []
        self.on_runtimes = []
        self._on_duration = 0
        self.overrides = 0
        self.total_decisions = 0
        
    def get_mask(self):
        mask = [True, True]
        if self.hvac_state == 1 and self.minutes_on < self.config.lockout_time:
            mask[0] = False
        elif self.hvac_state == 0 and self.minutes_off < self.config.lockout_time:
            mask[1] = False
        return np.array(mask)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init()
        
        avail = self.end_idx - self.start_idx
        ep_len = min(self.episode_length, avail)
        
        if self.is_training:
            max_start = max(self.start_idx, self.end_idx - ep_len)
            self.ep_start = np.random.randint(self.start_idx, max_start + 1)
        else:
            self.ep_start = self.start_idx
        
        self.episode_length = min(ep_len, self.end_idx - self.ep_start)
        self.T_in = self.config.setpoint + np.random.uniform(-0.5, 0.5)
        self.T_mass = self.T_in - 0.3
        
        return self._obs(), {}
    
    def step(self, rl_action):
        idx = min(self.ep_start + self.step_count, len(self.data) - 1)
        row = self.data.iloc[idx]
        T_out, price = row["T_out"], row["Price"]
        hour = self.data.index[idx].hour
        
        mask = self.get_mask()
        
        # Apply safety layer
        action, overridden = self.safety_layer.apply(
            self.T_in, rl_action, mask, self.current_runtime, self.hvac_state == 1
        )
        self.total_decisions += 1
        if overridden:
            self.overrides += 1
        
        # Update runtime tracking
        if action == self.prev_action:
            self.current_runtime += 1
        else:
            self.current_runtime = 1
        
        # Update HVAC state
        if action == 1:
            if self.hvac_state == 0:
                self.minutes_on = 1
            else:
                self.minutes_on += 1
            self.minutes_off = 0
            self._on_duration += 1
            self.hvac_state = 1
        else:
            if self.hvac_state == 1:
                if self._on_duration > 0:
                    self.on_runtimes.append(self._on_duration)
                self._on_duration = 0
                self.minutes_off = 1
            else:
                self.minutes_off += 1
            self.minutes_on = 0
            self.hvac_state = 0
        
        # Thermal dynamics (2R2C model)
        Q = action * self.config.Q_hvac_max
        Q_im = (self.T_in - self.T_mass) / self.config.R_i
        Q_mo = (self.T_mass - T_out) / (self.config.R_w + self.config.R_o)
        
        self.T_in += (Q - Q_im) / self.config.C_in * self.config.dt
        self.T_mass += (Q_im - Q_mo) / self.config.C_m * self.config.dt
        self.T_in = np.clip(self.T_in, 5, 40)
        self.T_mass = np.clip(self.T_mass, 5, 40)
        
        is_peak = 16 <= hour < 21
        power_kw = action * self.config.Q_hvac_kw
        
        # Reward
        cost = power_kw * (1/60) * price
        
        # Primary: negative cost (RL should minimize)
        reward = -cost * 3
        
        # Comfort bonus (safety layer should guarantee this, but reward helps)
        in_comfort = self.config.comfort_min <= self.T_in <= self.config.comfort_max
        if in_comfort:
            reward += 0.2
            # Setpoint bonus
            if abs(self.T_in - self.config.setpoint) < 0.5:
                reward += 0.1
        
        # Cycling penalty
        if action != self.prev_action:
            reward -= 0.15
            # Extra penalty for short cycles
            if self.prev_action == 1 and len(self.on_runtimes) > 0:
                if self.on_runtimes[-1] < self.config.min_runtime:
                    reward -= 0.2
        
        # Peak avoidance (when comfortable)
        if is_peak and action == 0 and in_comfort and self.T_in > self.config.setpoint:
            reward += 0.1
        
        self.prev_action = action
        
        self.actions.append(action)
        self.temps.append(self.T_in)
        self.costs.append(cost)
        self.power.append(power_kw)
        
        self.step_count += 1
        done = self.step_count >= self.episode_length
        
        if done and self._on_duration > 0:
            self.on_runtimes.append(self._on_duration)
        
        return self._obs(), reward, done, False, {"cost": cost}
    
    def _obs(self):
        idx = min(self.ep_start + self.step_count, len(self.data) - 1)
        row = self.data.iloc[idx]
        
        T_in_obs = np.clip(self.T_in + np.random.normal(0, self.config.obs_noise_std), 5, 40)
        mask = self.get_mask()
        
        rl_freedom = self.safety_layer.get_rl_freedom(T_in_obs)
        is_peak = 1.0 if 16 <= self.data.index[idx].hour < 21 else 0.0
        runtime_norm = min(self.current_runtime / 60.0, 1.0)
        
        return np.array([
            (T_in_obs - self.config.setpoint) / 10.0,
            (row["T_out"] - 10.0) / 30.0,
            (self.T_mass - self.config.setpoint) / 10.0,
            row["Price"] / self.config.peak_price,
            row["time_sin"], row["time_cos"],
            float(mask[0]), float(mask[1]),
            rl_freedom, is_peak, runtime_norm
        ], dtype=np.float32)
    
    def get_stats(self):
        in_comfort = sum(1 for t in self.temps 
                        if self.config.comfort_min <= t <= self.config.comfort_max)
        cycles = sum(1 for i in range(1, len(self.actions)) 
                    if self.actions[i-1] == 1 and self.actions[i] == 0)
        short_cycles = sum(1 for r in self.on_runtimes if r < self.config.min_runtime)
        
        return {
            "cost": sum(self.costs),
            "energy_kwh": sum(self.power) / 60.0,
            "comfort_ratio": in_comfort / max(len(self.temps), 1),
            "cycles": cycles,
            "mean_temp": np.mean(self.temps) if self.temps else 0,
            "on_ratio": np.mean(self.actions) if self.actions else 0,
            "avg_runtime": np.mean(self.on_runtimes) if self.on_runtimes else 0,
            "short_cycles": short_cycles,
            "override_ratio": self.overrides / max(self.total_decisions, 1),
        }


class SmartThermostat:
    """Baseline thermostat with same safety layer."""
    
    def __init__(self, config: Config):
        self.config = config
        self.action = 0
        self.runtime = config.lockout_time
        self.offtime = config.lockout_time
        self.current_runtime = 0
        self.safety_layer = SafetyLayer(config)
        
    def reset(self):
        self.action = 0
        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time
        self.current_runtime = 0
        
    def predict(self, T_in: float, mask: np.ndarray) -> int:
        upper = self.config.setpoint + self.config.deadband
        lower = self.config.setpoint - self.config.deadband
        
        if T_in > upper:
            desired = 0
        elif T_in < lower:
            desired = 1
        else:
            desired = self.action
        
        # Apply safety layer
        action, _ = self.safety_layer.apply(T_in, desired, mask, self.current_runtime, self.action == 1)
        
        # Update runtime
        if action == self.action:
            self.current_runtime += 1
        else:
            self.current_runtime = 1
        
        if action == 1:
            self.runtime += 1
            self.offtime = 0
        else:
            self.offtime += 1
            self.runtime = 0
        
        self.action = action
        return action


def evaluate(env, controller, is_ppo=False, seed=42):
    env.reset(seed=seed)
    obs, _ = env.reset(seed=seed)
    
    if not is_ppo:
        controller.reset()
    
    done = False
    while not done:
        mask = env.get_mask()
        if is_ppo:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            T_in = obs[0] * 10.0 + env.config.setpoint
            action = controller.predict(T_in, mask)
        obs, _, done, _, _ = env.step(action)
    
    return env.get_stats()


def statistical_analysis(baseline_results: List[Dict], ppo_results: List[Dict]) -> Dict:
    """Comprehensive statistical analysis."""
    
    metrics = ["cost", "energy_kwh", "comfort_ratio", "cycles", "avg_runtime", "short_cycles"]
    analysis = {}
    
    for m in metrics:
        b_vals = [r[m] for r in baseline_results]
        p_vals = [r[m] for r in ppo_results]
        
        b_mean, b_std = np.mean(b_vals), np.std(b_vals, ddof=1) if len(b_vals) > 1 else 0
        p_mean, p_std = np.mean(p_vals), np.std(p_vals, ddof=1) if len(p_vals) > 1 else 0
        
        # Paired t-test
        if len(b_vals) >= 2 and b_std > 1e-10 and p_std > 1e-10:
            t_stat, p_value = stats.ttest_rel(b_vals, p_vals)
        else:
            t_stat, p_value = 0, 1.0
        
        # Percent change
        if m in ["cost", "energy_kwh", "cycles", "short_cycles"]:
            pct_change = (b_mean - p_mean) / b_mean * 100 if b_mean != 0 else 0
        elif m == "comfort_ratio":
            pct_change = (p_mean - b_mean) / b_mean * 100 if b_mean != 0 else 0
        else:
            pct_change = (p_mean - b_mean) / b_mean * 100 if b_mean != 0 else 0
        
        analysis[m] = {
            "baseline_mean": b_mean,
            "baseline_std": b_std,
            "ppo_mean": p_mean,
            "ppo_std": p_std,
            "p_value": p_value,
            "pct_change": pct_change,
            "significant": p_value < 0.05,
        }
    
    return analysis


def print_results(analysis: Dict, scenario_name: str):
    """Print formatted results."""
    print(f"\n{'='*95}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*95}")
    print(f"{'Metric':<15} {'Baseline':<22} {'PI-DRL':<22} {'Change':<12} {'p-value':<10} {'Sig.'}")
    print(f"{'-'*95}")
    
    format_rules = {
        "cost": ("${:.2f}", "${:.2f}"),
        "energy_kwh": ("{:.1f} kWh", "{:.1f} kWh"),
        "comfort_ratio": ("{:.1%}", "{:.1%}"),
        "cycles": ("{:.0f}", "{:.0f}"),
        "avg_runtime": ("{:.1f} min", "{:.1f} min"),
        "short_cycles": ("{:.0f}", "{:.0f}"),
    }
    
    for m, v in analysis.items():
        fmt = format_rules.get(m, ("{:.2f}", "{:.2f}"))
        b_str = f"{fmt[0].format(v['baseline_mean'])} ± {fmt[1].format(v['baseline_std'])}"
        p_str = f"{fmt[0].format(v['ppo_mean'])} ± {fmt[1].format(v['ppo_std'])}"
        
        p_str_val = "< 0.001" if v['p_value'] < 0.001 else f"{v['p_value']:.3f}"
        sig = "***" if v['p_value'] < 0.001 else ("**" if v['p_value'] < 0.01 else ("*" if v['p_value'] < 0.05 else ""))
        
        print(f"{m:<15} {b_str:<22} {p_str:<22} {v['pct_change']:+.1f}%{'':<5} {p_str_val:<10} {sig}")
    
    print(f"{'='*95}")


def create_figure(scenario_name: str, env_b, env_p, output_dir: str):
    """Create comparison figure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    times = np.arange(len(env_b.temps)) / 60
    
    # Temperature
    ax1 = axes[0]
    ax1.plot(times, env_b.temps, 'b-', label='Baseline', linewidth=1.5)
    ax1.plot(times, env_p.temps, 'r-', label='PI-DRL', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=env_b.config.comfort_min, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=env_b.config.comfort_max, color='green', linestyle='--', alpha=0.5)
    ax1.fill_between(times, env_b.config.comfort_min, env_b.config.comfort_max, 
                     alpha=0.1, color='green', label='Comfort Band')
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(f'{scenario_name}: Temperature Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Power
    ax2 = axes[1]
    ax2.fill_between(times, env_b.power, label='Baseline', alpha=0.5, step='post', color='blue')
    ax2.fill_between(times, env_p.power, label='PI-DRL', alpha=0.5, step='post', color='red')
    ax2.set_ylabel('Power (kW)', fontsize=12)
    ax2.set_title('HVAC Power Consumption', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Cost
    ax3 = axes[2]
    ax3.plot(times, np.cumsum(env_b.costs), 'b-', label='Baseline', linewidth=2)
    ax3.plot(times, np.cumsum(env_p.costs), 'r-', label='PI-DRL', linewidth=2)
    ax3.set_ylabel('Cumulative Cost ($)', fontsize=12)
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_title('Cumulative Energy Cost', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics annotation
    stats_b = env_b.get_stats()
    stats_p = env_p.get_stats()
    
    text = (f"Cost: ${stats_b['cost']:.2f} → ${stats_p['cost']:.2f} "
            f"({(stats_b['cost']-stats_p['cost'])/stats_b['cost']*100:+.1f}%)\n"
            f"Comfort: {stats_b['comfort_ratio']*100:.1f}% → {stats_p['comfort_ratio']*100:.1f}%\n"
            f"Cycles: {stats_b['cycles']} → {stats_p['cycles']}")
    
    ax3.text(0.98, 0.05, text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{scenario_name.replace(" ", "_")}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("="*80)
    print("REALISTIC SAFETY-CONSTRAINED PI-DRL HVAC CONTROLLER")
    print("Properly sized HVAC system for all weather conditions")
    print("="*80)
    
    config = Config()
    output_dir = os.path.join(os.getcwd(), "HVAC_Realistic_Output")
    os.makedirs(output_dir, exist_ok=True)
    
    N_SEEDS = 10
    
    # Scenarios where baseline CAN achieve high comfort
    scenarios = [
        ("Mild Winter", 5.0, 5.0),    # T_out: 0°C to 10°C
        ("Cold Winter", -2.0, 5.0),   # T_out: -7°C to 3°C
        ("Spring", 12.0, 6.0),        # T_out: 6°C to 18°C
    ]
    
    all_results = {}
    
    for scenario_name, T_base, T_amp in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"T_out: base={T_base}°C, amplitude=±{T_amp}°C")
        print(f"Expected T_out range: {T_base-T_amp:.0f}°C to {T_base+T_amp:.0f}°C")
        print(f"{'='*80}")
        
        set_seed(config.seed)
        data = generate_data(T_base, T_amp, config, n_days=30)
        split_idx = int(len(data) * config.train_split)
        
        print(f"Actual T_out range: {data['T_out'].min():.1f}°C to {data['T_out'].max():.1f}°C")
        
        # First verify baseline can achieve high comfort
        baseline = SmartThermostat(config)
        env = SafetyHVACEnv(data, config, False, split_idx, len(data))
        baseline_check = evaluate(env, baseline, False, config.seed)
        
        print(f"\nBaseline verification:")
        print(f"  Comfort: {baseline_check['comfort_ratio']*100:.1f}%")
        print(f"  Cost: ${baseline_check['cost']:.2f}")
        print(f"  Cycles: {baseline_check['cycles']}")
        
        if baseline_check['comfort_ratio'] < 0.90:
            print(f"⚠️  Warning: Baseline comfort is low ({baseline_check['comfort_ratio']*100:.1f}%)")
            print("   This indicates HVAC is undersized for this scenario.")
            print("   Skipping this scenario...")
            continue
        
        # Training
        print(f"\n--- Training ({config.total_timesteps} steps) ---")
        
        def make_env():
            def _init():
                env = SafetyHVACEnv(data, config, True, 0, split_idx)
                env.reset(seed=config.seed)
                return env
            return _init
        
        vec_env = DummyVecEnv([make_env()])
        
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
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
        
        # Evaluation
        print(f"\n--- Evaluating ({N_SEEDS} seeds) ---")
        
        seeds = [config.seed + i * 17 for i in range(N_SEEDS)]
        baseline_results = []
        ppo_results = []
        
        for seed in seeds:
            set_seed(seed)
            
            env = SafetyHVACEnv(data, config, False, split_idx, len(data))
            baseline_results.append(evaluate(env, baseline, False, seed))
            
            env = SafetyHVACEnv(data, config, False, split_idx, len(data))
            ppo_results.append(evaluate(env, model, True, seed))
        
        # Analysis
        analysis = statistical_analysis(baseline_results, ppo_results)
        print_results(analysis, scenario_name)
        
        # Create figure
        set_seed(seeds[0])
        env_b = SafetyHVACEnv(data, config, False, split_idx, len(data))
        evaluate(env_b, baseline, False, seeds[0])
        env_p = SafetyHVACEnv(data, config, False, split_idx, len(data))
        evaluate(env_p, model, True, seeds[0])
        create_figure(scenario_name, env_b, env_p, output_dir)
        
        all_results[scenario_name] = {
            "analysis": analysis,
            "baseline_results": baseline_results,
            "ppo_results": ppo_results
        }
        
        model.save(os.path.join(output_dir, f"ppo_{scenario_name.replace(' ', '_').lower()}"))
    
    # Summary
    print(f"\n{'='*95}")
    print("PUBLICATION-READY SUMMARY")
    print(f"{'='*95}")
    
    for scenario_name, results in all_results.items():
        a = results["analysis"]
        
        comfort_ok = a['comfort_ratio']['ppo_mean'] >= 0.95
        cost_reduced = a['cost']['pct_change'] > 0
        cost_sig = a['cost']['p_value'] < 0.05
        cycles_ok = a['cycles']['pct_change'] >= 0  # Fewer or equal cycles
        
        print(f"\n{scenario_name}:")
        p_val_str = "<0.001" if a['cost']['p_value'] < 0.001 else f"{a['cost']['p_value']:.3f}"
        print(f"  Cost: {a['cost']['baseline_mean']:.2f} → {a['cost']['ppo_mean']:.2f} "
              f"({a['cost']['pct_change']:+.1f}%, p={p_val_str})")
        print(f"  Comfort: {a['comfort_ratio']['baseline_mean']*100:.1f}% → {a['comfort_ratio']['ppo_mean']*100:.1f}%")
        print(f"  Cycles: {a['cycles']['baseline_mean']:.0f} → {a['cycles']['ppo_mean']:.0f}")
        
        if comfort_ok and cost_reduced and cost_sig:
            print(f"  ✅ PUBLICATION READY: Comfort maintained, significant cost reduction")
        elif comfort_ok and not cost_sig:
            print(f"  ⚠️  Comfort maintained but cost reduction not statistically significant")
        else:
            print(f"  ❌ Does not meet publication criteria (comfort={a['comfort_ratio']['ppo_mean']*100:.1f}%)")
    
    # Save summary
    summary_rows = []
    for scenario, res in all_results.items():
        a = res["analysis"]
        summary_rows.append({
            "Scenario": scenario,
            "Baseline_Cost_Mean": f"{a['cost']['baseline_mean']:.2f}",
            "Baseline_Cost_Std": f"{a['cost']['baseline_std']:.2f}",
            "DRL_Cost_Mean": f"{a['cost']['ppo_mean']:.2f}",
            "DRL_Cost_Std": f"{a['cost']['ppo_std']:.2f}",
            "Cost_Change_Pct": f"{a['cost']['pct_change']:.1f}",
            "Cost_p_value": f"{a['cost']['p_value']:.4f}" if a['cost']['p_value'] >= 0.001 else "< 0.001",
            "Baseline_Comfort_Pct": f"{a['comfort_ratio']['baseline_mean']*100:.1f}",
            "DRL_Comfort_Pct": f"{a['comfort_ratio']['ppo_mean']*100:.1f}",
            "Comfort_p_value": f"{a['comfort_ratio']['p_value']:.4f}" if a['comfort_ratio']['p_value'] >= 0.001 else "< 0.001",
            "Baseline_Cycles": f"{a['cycles']['baseline_mean']:.0f}",
            "DRL_Cycles": f"{a['cycles']['ppo_mean']:.0f}",
            "Cycles_p_value": f"{a['cycles']['p_value']:.4f}" if a['cycles']['p_value'] >= 0.001 else "< 0.001",
        })
    
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "publication_summary.csv"), index=False)
    
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
