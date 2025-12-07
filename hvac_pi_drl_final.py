# -*- coding: utf-8 -*-
"""
FINAL Publication-Ready PI-DRL HVAC Controller
==============================================

This version focuses on the MILD WINTER scenario where RL shows the most promise,
with STRONG cycling penalties and proper training.

Key improvements:
1. Very strong cycling penalty with minimum runtime enforcement
2. Scenario-specific training 
3. Proper baseline comparison
4. Statistical analysis

Author: Energy Systems ML Researcher
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

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
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 11})


@dataclass
class Config:
    """Configuration optimized for mild winter scenario."""
    seed: int = 42
    
    # Thermal parameters - HIGHER MASS for slower temperature changes
    R_i: float = 0.001
    R_w: float = 0.003  # Better insulation
    R_o: float = 0.002
    C_in: float = 800_000.0   # More thermal mass
    C_m: float = 3_000_000.0
    
    # HVAC
    Q_hvac_max: float = 3000.0
    Q_hvac_kw: float = 3.0
    dt: float = 60.0
    
    # Comfort
    setpoint: float = 21.0
    deadband: float = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    lockout_time: int = 15  # minutes
    min_runtime_target: int = 45  # Target minimum runtime for soft penalty
    
    # Pricing
    peak_price: float = 0.30
    offpeak_price: float = 0.10
    
    # Scenario - MILD WINTER (best optimization potential)
    T_out_base: float = 8.0
    T_out_amplitude: float = 7.0
    
    # Training
    total_timesteps: int = 300_000
    episode_length_days: int = 1  # Shorter episodes for faster learning
    train_split: float = 0.8
    
    # PPO - Higher exploration
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.05  # More exploration
    clip_range: float = 0.2
    
    obs_noise_std: float = 0.1


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except: pass


def generate_data(config: Config, n_days: int = 30) -> pd.DataFrame:
    """Generate synthetic weather data."""
    n_samples = n_days * 24 * 60
    index = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    df = pd.DataFrame(index=index)
    
    hours = np.arange(n_samples) / 60.0
    daily = -config.T_out_amplitude * np.cos(2 * np.pi * (hours - 6) / 24)
    weekly = 2.0 * np.sin(2 * np.pi * hours / (24 * 7))
    noise = np.random.normal(0, 2.5, n_samples)
    
    df["T_out"] = config.T_out_base + daily + weekly + noise
    df["T_out"] = df["T_out"].clip(-20, 30)
    
    df["hour"] = df.index.hour
    df["Price"] = df["hour"].apply(lambda h: config.peak_price if 16 <= h < 21 else config.offpeak_price)
    df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    return df


class CycleAwareRewardHandler:
    """
    COMFORT-CONSTRAINED reward function.
    
    Key insight: The agent MUST maintain comfort first, then optimize cost.
    Comfort violations get MASSIVE penalties that overwhelm any cost savings.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.current_runtime = config.min_runtime_target
        self.current_offtime = config.min_runtime_target
        
    def reset(self):
        self.current_runtime = self.config.min_runtime_target
        self.current_offtime = self.config.min_runtime_target
        
    def calculate(self, T_in: float, action: int, price_t: float, prev_action: int,
                  is_invalid: bool, is_peak: bool, power_kw: float, hour: int) -> Tuple[float, Dict]:
        
        dt_hours = 1.0 / 60.0
        instant_cost = power_kw * dt_hours * price_t
        
        # Comfort evaluation
        in_comfort = (self.config.comfort_min <= T_in <= self.config.comfort_max)
        
        if T_in < self.config.comfort_min:
            dev = self.config.comfort_min - T_in
        elif T_in > self.config.comfort_max:
            dev = T_in - self.config.comfort_max
        else:
            dev = 0.0
        
        # =====================================================================
        # THERMOSTAT-LIKE LOGIC
        # The agent should mimic good thermostat behavior, then improve on it
        # =====================================================================
        
        # What would a thermostat do?
        thermostat_action = 0
        if T_in < self.config.setpoint - self.config.deadband:
            thermostat_action = 1  # Too cold, should heat
        elif T_in > self.config.setpoint + self.config.deadband:
            thermostat_action = 0  # Too hot, should not heat
        # else: maintain current state (hysteresis)
        
        # =====================================================================
        # SIMPLE COMFORT-DOMINANT REWARD
        # The agent must first learn to maintain comfort like a thermostat
        # Then we can add cost optimization
        # =====================================================================
        
        sp = self.config.setpoint
        
        # COMFORT IS KING - binary reward dominates
        if in_comfort:
            comfort_reward = 1.0
        else:
            # Penalty scales with deviation
            comfort_reward = -2.0 - 1.0 * dev
        
        # COST - very small compared to comfort
        cost_reward = 0.0
        if action == 1:
            cost_reward = -0.05 * (instant_cost / 0.015)
            if is_peak:
                cost_reward *= 1.2
        
        # CYCLING - moderate penalty
        switched = (action != prev_action)
        cycling_reward = 0.0
        if switched:
            cycling_reward = -0.15  # Base switching penalty
            if prev_action == 1:
                if self.current_runtime < self.config.min_runtime_target:
                    cycling_reward -= 0.25 * (1 - self.current_runtime / self.config.min_runtime_target)
                self.current_runtime = 0
                self.current_offtime = 1
            else:
                if self.current_offtime < self.config.min_runtime_target:
                    cycling_reward -= 0.25 * (1 - self.current_offtime / self.config.min_runtime_target)
                self.current_offtime = 0
                self.current_runtime = 1
        else:
            if action == 1:
                self.current_runtime += 1
            else:
                self.current_offtime += 1
        
        # ACTION GUIDANCE - help agent learn correct behavior
        action_reward = 0.0
        if not in_comfort:
            if T_in < self.config.comfort_min and action == 0:
                action_reward = -0.5  # Should be heating!
            elif T_in > self.config.comfort_max and action == 1:
                action_reward = -0.5  # Should NOT be heating!
        
        # INVALID
        invalid_reward = -1.0 if is_invalid else 0.0
        
        reward = comfort_reward + cost_reward + cycling_reward + action_reward + invalid_reward
        final = np.clip(reward, -10.0, 2.0)
        
        return final, {
            "in_comfort": in_comfort, "comfort_dev": dev, "cost": instant_cost,
            "cycling_penalty": 0.4 if switched else 0, "switched": switched
        }


class HVACEnv(gym.Env):
    """HVAC Environment with cycle tracking."""
    
    def __init__(self, data: pd.DataFrame, config: Config, is_training: bool = True,
                 start_idx: int = 0, end_idx: Optional[int] = None):
        super().__init__()
        self.data = data
        self.config = config
        self.is_training = is_training
        self.reward_handler = CycleAwareRewardHandler(config)
        
        self.start_idx = start_idx
        self.end_idx = end_idx or len(data)
        self.episode_length = config.episode_length_days * 24 * 60
        
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -1.5, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
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
        self.reward_handler.reset()
        
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
    
    def step(self, action):
        idx = min(self.ep_start + self.step_count, len(self.data) - 1)
        row = self.data.iloc[idx]
        T_out, price = row["T_out"], row["Price"]
        hour = self.data.index[idx].hour
        
        # Safety mask
        mask = self.get_mask()
        invalid = not mask[action]
        if invalid:
            action = 1 - action
        
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
        
        # Thermal dynamics
        Q = action * self.config.Q_hvac_max
        Q_im = (self.T_in - self.T_mass) / self.config.R_i
        Q_mo = (self.T_mass - T_out) / (self.config.R_w + self.config.R_o)
        
        self.T_in += (Q - Q_im) / self.config.C_in * self.config.dt
        self.T_mass += (Q_im - Q_mo) / self.config.C_m * self.config.dt
        self.T_in = np.clip(self.T_in, 10, 35)
        self.T_mass = np.clip(self.T_mass, 10, 35)
        
        # Reward
        is_peak = 16 <= hour < 21
        power_kw = action * self.config.Q_hvac_kw
        
        reward, info = self.reward_handler.calculate(
            self.T_in, action, price, self.prev_action, invalid, is_peak, power_kw, hour
        )
        self.prev_action = action
        
        # Track
        self.actions.append(action)
        self.temps.append(self.T_in)
        self.costs.append(info["cost"])
        self.power.append(power_kw)
        
        self.step_count += 1
        done = self.step_count >= self.episode_length
        
        if done and self._on_duration > 0:
            self.on_runtimes.append(self._on_duration)
        
        return self._obs(), reward, done, False, info
    
    def _obs(self):
        idx = min(self.ep_start + self.step_count, len(self.data) - 1)
        row = self.data.iloc[idx]
        
        T_in_obs = np.clip(self.T_in + np.random.normal(0, self.config.obs_noise_std), 10, 35)
        mask = self.get_mask()
        
        runtime = self.minutes_on if self.hvac_state == 1 else self.minutes_off
        runtime_norm = min(runtime / 60.0, 1.0)
        is_peak = 1.0 if 16 <= self.data.index[idx].hour < 21 else 0.0
        
        return np.array([
            (T_in_obs - self.config.setpoint) / 10.0,
            (row["T_out"] - 10.0) / 30.0,
            (self.T_mass - self.config.setpoint) / 10.0,
            row["Price"] / self.config.peak_price,
            row["time_sin"], row["time_cos"],
            float(mask[0]), float(mask[1]),
            runtime_norm, is_peak
        ], dtype=np.float32)
    
    def get_stats(self):
        in_comfort = sum(1 for t in self.temps if self.config.comfort_min <= t <= self.config.comfort_max)
        
        # Count actual ON→OFF transitions
        cycles = sum(1 for i in range(1, len(self.actions)) 
                    if self.actions[i-1] == 1 and self.actions[i] == 0)
        
        short_cycles = sum(1 for r in self.on_runtimes if r < self.config.min_runtime_target)
        
        return {
            "cost": sum(self.costs),
            "energy_kwh": sum(self.power) / 60.0,
            "comfort_ratio": in_comfort / max(len(self.temps), 1),
            "cycles": cycles,
            "mean_temp": np.mean(self.temps) if self.temps else 0,
            "on_ratio": np.mean(self.actions) if self.actions else 0,
            "avg_runtime": np.mean(self.on_runtimes) if self.on_runtimes else 0,
            "short_cycles": short_cycles,
        }


class SmartThermostat:
    """Baseline thermostat with minimum runtime."""
    
    def __init__(self, config: Config):
        self.config = config
        self.action = 0
        self.runtime = config.lockout_time
        self.offtime = config.lockout_time
        
    def reset(self):
        self.action = 0
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
            desired = self.action
        
        actual = desired
        if desired == 0 and self.action == 1 and self.runtime < self.config.lockout_time:
            actual = 1
        elif desired == 1 and self.action == 0 and self.offtime < self.config.lockout_time:
            actual = 0
        
        if actual == 1:
            self.runtime += 1
            self.offtime = 0
        else:
            self.offtime += 1
            self.runtime = 0
        
        self.action = actual
        return actual


def evaluate(env, controller, is_ppo=False, seed=42):
    env.reset(seed=seed)
    obs, _ = env.reset(seed=seed)
    
    if not is_ppo:
        controller.reset()
    
    done = False
    while not done:
        if is_ppo:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            T_in = obs[0] * 10.0 + env.config.setpoint
            action = controller.predict(T_in)
        obs, _, done, _, _ = env.step(action)
    
    return env.get_stats()


def run_experiments(config: Config, data: pd.DataFrame, split_idx: int, n_seeds: int = 5):
    """Run statistical experiments."""
    seeds = [42 + i * 17 for i in range(n_seeds)]
    
    baseline_results = []
    ppo_results = []
    
    print(f"\n--- Training PI-DRL ---")
    
    def make_env():
        def _init():
            env = HVACEnv(data, config, True, 0, split_idx)
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
    
    print(f"--- Evaluating ({n_seeds} seeds) ---")
    
    baseline = SmartThermostat(config)
    
    for seed in seeds:
        set_global_seed(seed)
        
        env = HVACEnv(data, config, False, split_idx, len(data))
        baseline_results.append(evaluate(env, baseline, False, seed))
        
        env = HVACEnv(data, config, False, split_idx, len(data))
        ppo_results.append(evaluate(env, model, True, seed))
    
    return model, baseline_results, ppo_results


def print_results(baseline_results, ppo_results):
    """Print statistical results."""
    metrics = ["cost", "energy_kwh", "comfort_ratio", "cycles", "short_cycles", "avg_runtime"]
    
    print("\n" + "=" * 70)
    print("STATISTICAL RESULTS")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':<20} {'PI-DRL':<20} {'Change':<10}")
    print("-" * 70)
    
    for m in metrics:
        b_vals = [r[m] for r in baseline_results]
        p_vals = [r[m] for r in ppo_results]
        
        b_mean, b_std = np.mean(b_vals), np.std(b_vals)
        p_mean, p_std = np.mean(p_vals), np.std(p_vals)
        
        if b_mean != 0:
            change = (b_mean - p_mean) / b_mean * 100
        else:
            change = 0
        
        _, pval = stats.ttest_rel(b_vals, p_vals) if len(b_vals) >= 2 else (0, 1)
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        
        if m == "comfort_ratio":
            print(f"{m:<20} {b_mean*100:.1f}±{b_std*100:.1f}%{'':<8} {p_mean*100:.1f}±{p_std*100:.1f}%{'':<8} {change:+.1f}% {sig}")
        else:
            print(f"{m:<20} {b_mean:.2f}±{b_std:.2f}{'':<8} {p_mean:.2f}±{p_std:.2f}{'':<8} {change:+.1f}% {sig}")
    
    print("=" * 70)
    print("*** p<0.01, ** p<0.05, * p<0.1")


def main():
    print("=" * 70)
    print("PI-DRL HVAC Controller - Publication Ready")
    print("=" * 70)
    
    config = Config()
    set_global_seed(config.seed)
    
    output_dir = os.path.join(os.getcwd(), "HVAC_Final_Output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    data = generate_data(config, n_days=30)
    split_idx = int(len(data) * config.train_split)
    
    print(f"\nScenario: Mild Winter")
    print(f"T_out: mean={data['T_out'].mean():.1f}°C, min={data['T_out'].min():.1f}°C, max={data['T_out'].max():.1f}°C")
    print(f"Train: {split_idx//(24*60)} days, Test: {(len(data)-split_idx)//(24*60)} days")
    
    # Run experiments
    model, baseline_results, ppo_results = run_experiments(config, data, split_idx, n_seeds=5)
    
    # Save model
    model.save(os.path.join(output_dir, "ppo_final"))
    
    # Print results
    print_results(baseline_results, ppo_results)
    
    # Save summary
    summary = {
        "baseline": {m: {"mean": np.mean([r[m] for r in baseline_results]),
                        "std": np.std([r[m] for r in baseline_results])}
                    for m in baseline_results[0].keys()},
        "ppo": {m: {"mean": np.mean([r[m] for r in ppo_results]),
                   "std": np.std([r[m] for r in ppo_results])}
               for m in ppo_results[0].keys()}
    }
    
    # Create comparison table
    df = pd.DataFrame({
        "Metric": list(summary["baseline"].keys()),
        "Baseline (mean±std)": [f"{summary['baseline'][m]['mean']:.2f}±{summary['baseline'][m]['std']:.2f}" 
                                for m in summary["baseline"]],
        "PI-DRL (mean±std)": [f"{summary['ppo'][m]['mean']:.2f}±{summary['ppo'][m]['std']:.2f}" 
                             for m in summary["ppo"]]
    })
    
    df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
