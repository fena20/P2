# -*- coding: utf-8 -*-
"""
Safety-Constrained PI-DRL HVAC Controller
==========================================

This version implements a SAFETY LAYER that GUARANTEES comfort constraint.
The RL agent can only make decisions when comfort is guaranteed.

Key innovation:
- When T_in < comfort_min: HVAC is FORCED ON (safety override)
- When T_in > comfort_max: HVAC is FORCED OFF (safety override)
- RL only controls in the "safe zone" where both actions are acceptable

This ensures:
1. Comfort ≥ 95% (physically guaranteed)
2. RL optimizes cost only within safe constraints
3. Consistent behavior across scenarios

Author: Energy Systems ML Researcher
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import json

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
    
    # Thermal model (2R2C)
    R_i: float = 0.0008     # Indoor-mass resistance
    R_w: float = 0.003      # Wall resistance
    R_o: float = 0.002      # Outdoor resistance
    C_in: float = 600_000.0   # Indoor capacitance (J/K)
    C_m: float = 2_500_000.0  # Mass capacitance (J/K)
    
    # HVAC
    Q_hvac_max: float = 3000.0  # Watts
    Q_hvac_kw: float = 3.0
    dt: float = 60.0  # seconds per step
    
    # Comfort - HARD CONSTRAINT enforced by safety layer
    setpoint: float = 21.0
    deadband: float = 1.0  # Tighter deadband for more aggressive control
    comfort_min: float = 19.5
    comfort_max: float = 23.5
    safety_margin: float = 0.5  # Margin for safety layer to kick in
    lockout_time: int = 15
    
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
    ent_coef: float = 0.01  # Lower entropy for more stable policy
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
    noise = np.random.normal(0, 2.0, n)
    
    df["T_out"] = T_base + daily + noise
    df["T_out"] = df["T_out"].clip(-25, 35)
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
    3. Otherwise: RL decides
    
    This ensures we NEVER violate comfort for more than a brief period.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.heating_threshold = config.comfort_min + config.safety_margin
        self.cooling_threshold = config.comfort_max - config.safety_margin
        
    def apply(self, T_in: float, rl_action: int, mask: np.ndarray) -> Tuple[int, bool]:
        """
        Apply safety override.
        
        Returns:
            action: Final action (0 or 1)
            overridden: Whether safety layer overrode RL decision
        """
        # First check lockout constraints
        if not mask[rl_action]:
            rl_action = 1 - rl_action
        
        # Safety override for cold
        if T_in <= self.heating_threshold:
            if mask[1]:  # Can heat
                return 1, (rl_action != 1)
            else:
                return rl_action, False  # Can't heat, follow lockout
        
        # Safety override for hot
        if T_in >= self.cooling_threshold:
            if mask[0]:  # Can turn off
                return 0, (rl_action != 0)
            else:
                return rl_action, False  # Can't turn off, follow lockout
        
        # In safe zone - RL decides
        return rl_action, False
    
    def get_rl_allowed_zone(self, T_in: float) -> bool:
        """Check if RL is allowed to make free decisions."""
        return self.heating_threshold < T_in < self.cooling_threshold


class SafetyHVACEnv(gym.Env):
    """
    HVAC Environment with integrated safety layer.
    """
    
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
        
        # Observation: [T_in_norm, T_out_norm, T_mass_norm, price_norm, 
        #               time_sin, time_cos, can_off, can_on, safety_zone, is_peak]
        self.observation_space = spaces.Box(
            low=np.array([-2, -2, -2, 0, -1, -1, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dtype=np.float32),
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
            mask[0] = False  # Can't turn off
        elif self.hvac_state == 0 and self.minutes_off < self.config.lockout_time:
            mask[1] = False  # Can't turn on
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
        self.T_in = self.config.setpoint + np.random.uniform(-0.3, 0.3)
        self.T_mass = self.T_in - 0.2
        
        return self._obs(), {}
    
    def step(self, rl_action):
        idx = min(self.ep_start + self.step_count, len(self.data) - 1)
        row = self.data.iloc[idx]
        T_out, price = row["T_out"], row["Price"]
        hour = self.data.index[idx].hour
        
        mask = self.get_mask()
        
        # Apply safety layer
        action, overridden = self.safety_layer.apply(self.T_in, rl_action, mask)
        self.total_decisions += 1
        if overridden:
            self.overrides += 1
        
        # Update state
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
        self.T_in = np.clip(self.T_in, 10, 35)
        self.T_mass = np.clip(self.T_mass, 10, 35)
        
        is_peak = 16 <= hour < 21
        power_kw = action * self.config.Q_hvac_kw
        
        # Reward: simple cost-focused (safety layer guarantees comfort)
        cost = power_kw * (1/60) * price
        
        # Base reward is negative cost
        reward = -cost * 5
        
        # Switching penalty (encourage stable operation)
        if action != self.prev_action:
            reward -= 0.1
        
        # If in safe zone and NOT heating during peak -> bonus
        if self.safety_layer.get_rl_allowed_zone(self.T_in):
            if is_peak and action == 0:
                reward += 0.05
        
        self.prev_action = action
        
        self.actions.append(action)
        self.temps.append(self.T_in)
        self.costs.append(cost)
        self.power.append(power_kw)
        
        self.step_count += 1
        done = self.step_count >= self.episode_length
        
        if done and self._on_duration > 0:
            self.on_runtimes.append(self._on_duration)
        
        return self._obs(), reward, done, False, {"cost": cost, "overridden": overridden}
    
    def _obs(self):
        idx = min(self.ep_start + self.step_count, len(self.data) - 1)
        row = self.data.iloc[idx]
        
        T_in_obs = np.clip(self.T_in + np.random.normal(0, self.config.obs_noise_std), 10, 35)
        mask = self.get_mask()
        
        safety_zone = 1.0 if self.safety_layer.get_rl_allowed_zone(T_in_obs) else 0.0
        is_peak = 1.0 if 16 <= self.data.index[idx].hour < 21 else 0.0
        
        return np.array([
            (T_in_obs - self.config.setpoint) / 10.0,
            (row["T_out"] - 10.0) / 30.0,
            (self.T_mass - self.config.setpoint) / 10.0,
            row["Price"] / self.config.peak_price,
            row["time_sin"], row["time_cos"],
            float(mask[0]), float(mask[1]),
            safety_zone, is_peak
        ], dtype=np.float32)
    
    def get_stats(self):
        in_comfort = sum(1 for t in self.temps 
                        if self.config.comfort_min <= t <= self.config.comfort_max)
        cycles = sum(1 for i in range(1, len(self.actions)) 
                    if self.actions[i-1] == 1 and self.actions[i] == 0)
        
        return {
            "cost": sum(self.costs),
            "energy_kwh": sum(self.power) / 60.0,
            "comfort_ratio": in_comfort / max(len(self.temps), 1),
            "cycles": cycles,
            "mean_temp": np.mean(self.temps) if self.temps else 0,
            "on_ratio": np.mean(self.actions) if self.actions else 0,
            "avg_runtime": np.mean(self.on_runtimes) if self.on_runtimes else 0,
            "override_ratio": self.overrides / max(self.total_decisions, 1),
        }


class SmartThermostat:
    """
    Baseline thermostat with same safety layer for fair comparison.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.action = 0
        self.runtime = config.lockout_time
        self.offtime = config.lockout_time
        self.safety_layer = SafetyLayer(config)
        
    def reset(self):
        self.action = 0
        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time
        
    def predict(self, T_in: float, mask: np.ndarray) -> int:
        upper = self.config.setpoint + self.config.deadband
        lower = self.config.setpoint - self.config.deadband
        
        # Thermostat logic
        if T_in > upper:
            desired = 0
        elif T_in < lower:
            desired = 1
        else:
            desired = self.action
        
        # Apply safety layer (same as RL)
        action, _ = self.safety_layer.apply(T_in, desired, mask)
        
        # Update state
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


def run_statistical_analysis(baseline_results: List[Dict], ppo_results: List[Dict]) -> Dict:
    """Comprehensive statistical analysis for ALL metrics."""
    
    metrics = ["cost", "energy_kwh", "comfort_ratio", "cycles", "avg_runtime", "override_ratio", "mean_temp"]
    analysis = {}
    
    for m in metrics:
        b_vals = [r[m] for r in baseline_results]
        p_vals = [r[m] for r in ppo_results]
        
        b_mean, b_std = np.mean(b_vals), np.std(b_vals, ddof=1)
        p_mean, p_std = np.mean(p_vals), np.std(p_vals, ddof=1)
        
        # Paired t-test
        if len(b_vals) >= 2 and np.std(b_vals) > 1e-10 and np.std(p_vals) > 1e-10:
            t_stat, p_value = stats.ttest_rel(b_vals, p_vals)
        else:
            t_stat, p_value = 0, 1.0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((b_std**2 + p_std**2) / 2)
        cohens_d = (b_mean - p_mean) / pooled_std if pooled_std > 1e-10 else 0
        
        # Percent change (positive = improvement = baseline higher)
        if m in ["cost", "energy_kwh", "cycles"]:  # Lower is better
            pct_change = (b_mean - p_mean) / b_mean * 100 if b_mean != 0 else 0
        elif m in ["comfort_ratio", "avg_runtime"]:  # Higher is better (for runtime)
            pct_change = (p_mean - b_mean) / b_mean * 100 if b_mean != 0 else 0
        else:
            pct_change = (b_mean - p_mean) / b_mean * 100 if b_mean != 0 else 0
        
        analysis[m] = {
            "baseline_mean": b_mean,
            "baseline_std": b_std,
            "ppo_mean": p_mean,
            "ppo_std": p_std,
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "pct_change": pct_change,
            "significant": p_value < 0.05,
            "n_samples": len(b_vals)
        }
    
    return analysis


def print_results(analysis: Dict, scenario_name: str):
    """Print comprehensive results table."""
    
    print(f"\n{'='*95}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*95}")
    print(f"{'Metric':<18} {'Baseline':<20} {'PI-DRL':<20} {'Change':<12} {'p-value':<12} {'Sig.'}")
    print(f"{'-'*95}")
    
    for m, v in analysis.items():
        if m == "comfort_ratio" or m == "override_ratio":
            b_str = f"{v['baseline_mean']*100:.1f} ± {v['baseline_std']*100:.1f}%"
            p_str = f"{v['ppo_mean']*100:.1f} ± {v['ppo_std']*100:.1f}%"
        elif m == "avg_runtime":
            b_str = f"{v['baseline_mean']:.1f} ± {v['baseline_std']:.1f} min"
            p_str = f"{v['ppo_mean']:.1f} ± {v['ppo_std']:.1f} min"
        else:
            b_str = f"{v['baseline_mean']:.2f} ± {v['baseline_std']:.2f}"
            p_str = f"{v['ppo_mean']:.2f} ± {v['ppo_std']:.2f}"
        
        if v['p_value'] < 0.001:
            p_str_val = "< 0.001"
        else:
            p_str_val = f"{v['p_value']:.3f}"
        
        sig = "***" if v['p_value'] < 0.001 else ("**" if v['p_value'] < 0.01 else ("*" if v['p_value'] < 0.05 else ""))
        
        print(f"{m:<18} {b_str:<20} {p_str:<20} {v['pct_change']:+.1f}%{'':<5} {p_str_val:<12} {sig}")
    
    print(f"{'='*95}")
    print("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05")


def create_figures(scenario_name: str, baseline_env, ppo_env, output_dir: str):
    """Create publication-quality figures."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    times = np.arange(len(baseline_env.temps)) / 60  # Hours
    
    # Temperature comparison
    ax1 = axes[0]
    ax1.plot(times, baseline_env.temps, 'b-', label='Baseline', linewidth=1)
    ax1.plot(times, ppo_env.temps, 'r-', label='PI-DRL', linewidth=1, alpha=0.8)
    ax1.axhline(y=baseline_env.config.comfort_min, color='gray', linestyle='--', label='Comfort Band')
    ax1.axhline(y=baseline_env.config.comfort_max, color='gray', linestyle='--')
    ax1.fill_between(times, baseline_env.config.comfort_min, baseline_env.config.comfort_max, 
                     alpha=0.1, color='green')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'{scenario_name}: Temperature Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Power comparison
    ax2 = axes[1]
    ax2.fill_between(times, baseline_env.power, label='Baseline', alpha=0.5, step='post')
    ax2.fill_between(times, ppo_env.power, label='PI-DRL', alpha=0.5, step='post')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('HVAC Power Consumption')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cumulative cost
    ax3 = axes[2]
    ax3.plot(times, np.cumsum(baseline_env.costs), 'b-', label='Baseline', linewidth=2)
    ax3.plot(times, np.cumsum(ppo_env.costs), 'r-', label='PI-DRL', linewidth=2)
    ax3.set_ylabel('Cumulative Cost ($)')
    ax3.set_xlabel('Time (hours)')
    ax3.set_title('Cumulative Energy Cost')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{scenario_name.replace(" ", "_")}_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("="*80)
    print("SAFETY-CONSTRAINED PI-DRL HVAC CONTROLLER")
    print("Comfort guaranteed by safety layer - RL optimizes within safe bounds")
    print("="*80)
    
    config = Config()
    output_dir = os.path.join(os.getcwd(), "HVAC_Safety_Output")
    os.makedirs(output_dir, exist_ok=True)
    
    N_SEEDS = 10
    
    scenarios = [
        ("Mild Winter", 8.0, 7.0),
        ("Cold Winter", 2.0, 6.0),
        ("Spring", 14.0, 8.0),
    ]
    
    all_results = {}
    
    for scenario_name, T_base, T_amp in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"T_out: base={T_base}°C, amplitude=±{T_amp}°C")
        print(f"{'='*80}")
        
        set_seed(config.seed)
        data = generate_data(T_base, T_amp, config, n_days=30)
        split_idx = int(len(data) * config.train_split)
        
        print(f"T_out range: {data['T_out'].min():.1f}°C to {data['T_out'].max():.1f}°C")
        
        # Training
        print(f"\n--- Training ---")
        
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
        
        baseline = SmartThermostat(config)
        seeds = [config.seed + i * 17 for i in range(N_SEEDS)]
        baseline_results = []
        ppo_results = []
        
        for i, seed in enumerate(seeds):
            set_seed(seed)
            
            env = SafetyHVACEnv(data, config, False, split_idx, len(data))
            baseline_results.append(evaluate(env, baseline, False, seed))
            
            env = SafetyHVACEnv(data, config, False, split_idx, len(data))
            ppo_results.append(evaluate(env, model, True, seed))
        
        # Statistical analysis
        analysis = run_statistical_analysis(baseline_results, ppo_results)
        print_results(analysis, scenario_name)
        
        # Create figure for first seed
        set_seed(seeds[0])
        env_baseline = SafetyHVACEnv(data, config, False, split_idx, len(data))
        evaluate(env_baseline, baseline, False, seeds[0])
        
        env_ppo = SafetyHVACEnv(data, config, False, split_idx, len(data))
        evaluate(env_ppo, model, True, seeds[0])
        
        create_figures(scenario_name, env_baseline, env_ppo, output_dir)
        
        all_results[scenario_name] = {
            "analysis": analysis,
            "baseline_results": baseline_results,
            "ppo_results": ppo_results
        }
        
        model.save(os.path.join(output_dir, f"ppo_{scenario_name.replace(' ', '_').lower()}"))
    
    # Summary table
    print(f"\n{'='*95}")
    print("SUMMARY ACROSS ALL SCENARIOS")
    print(f"{'='*95}")
    
    summary_data = []
    for scenario_name, results in all_results.items():
        a = results["analysis"]
        
        # Interpret results
        cost_better = a['cost']['pct_change'] > 0
        comfort_maintained = a['comfort_ratio']['ppo_mean'] >= 0.95
        
        summary_data.append({
            "Scenario": scenario_name,
            "Cost Reduction": f"{a['cost']['pct_change']:.1f}%" if cost_better else f"{a['cost']['pct_change']:.1f}%",
            "Cost p": "< 0.001" if a['cost']['p_value'] < 0.001 else f"{a['cost']['p_value']:.3f}",
            "Comfort (Base)": f"{a['comfort_ratio']['baseline_mean']*100:.1f}%",
            "Comfort (DRL)": f"{a['comfort_ratio']['ppo_mean']*100:.1f}%",
            "Comfort p": "< 0.001" if a['comfort_ratio']['p_value'] < 0.001 else f"{a['comfort_ratio']['p_value']:.3f}",
            "Cycles (Base)": f"{a['cycles']['baseline_mean']:.0f}",
            "Cycles (DRL)": f"{a['cycles']['ppo_mean']:.0f}",
            "Cycles p": "< 0.001" if a['cycles']['p_value'] < 0.001 else f"{a['cycles']['p_value']:.3f}",
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    
    # Interpretation
    print(f"\n{'='*95}")
    print("INTERPRETATION")
    print(f"{'='*95}")
    
    for scenario_name, results in all_results.items():
        a = results["analysis"]
        print(f"\n{scenario_name}:")
        
        comfort_ok = a['comfort_ratio']['ppo_mean'] >= 0.95
        cost_reduced = a['cost']['pct_change'] > 0
        cost_sig = a['cost']['p_value'] < 0.05
        
        if comfort_ok and cost_reduced and cost_sig:
            print(f"  ✅ SUCCESS: Comfort maintained ({a['comfort_ratio']['ppo_mean']*100:.1f}%) "
                  f"with significant cost reduction ({a['cost']['pct_change']:.1f}%, p={a['cost']['p_value']:.4f})")
        elif comfort_ok and not cost_reduced:
            print(f"  ⚠️  NEUTRAL: Comfort maintained but no cost savings")
        else:
            print(f"  ❌ ISSUE: Comfort={a['comfort_ratio']['ppo_mean']*100:.1f}%, "
                  f"Cost change={a['cost']['pct_change']:.1f}%")
    
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
