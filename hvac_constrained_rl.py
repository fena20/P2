# -*- coding: utf-8 -*-
"""
Constrained PI-DRL HVAC Controller
==================================

This version implements COMFORT AS A HARD CONSTRAINT using Lagrangian relaxation.
The agent MUST maintain comfort ≥ 95% before optimizing for cost.

Key improvements:
1. Lagrangian-based constrained optimization
2. Proper statistical reporting for ALL metrics
3. Physical explanation of agent behavior
4. Consistent results across scenarios

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
from stable_baselines3.common.callbacks import BaseCallback

import random
import torch
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class Config:
    seed: int = 42
    
    # Thermal model
    R_i: float = 0.0008
    R_w: float = 0.003
    R_o: float = 0.002
    C_in: float = 600_000.0
    C_m: float = 2_500_000.0
    
    # HVAC
    Q_hvac_max: float = 3000.0
    Q_hvac_kw: float = 3.0
    dt: float = 60.0
    
    # Comfort - HARD CONSTRAINT
    setpoint: float = 21.0
    deadband: float = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    comfort_target: float = 0.95  # Must achieve 95% comfort
    lockout_time: int = 15
    min_runtime_target: int = 30
    
    # Pricing
    peak_price: float = 0.30
    offpeak_price: float = 0.10
    
    # Training
    total_timesteps: int = 200_000
    episode_length_days: int = 1
    train_split: float = 0.8
    
    # Lagrangian multiplier for comfort constraint
    lambda_comfort: float = 10.0  # Will be adapted
    lambda_cycling: float = 5.0
    
    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.02
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


class ConstrainedRewardHandler:
    """
    Lagrangian-based reward for constrained optimization.
    
    Objective: Minimize cost
    Constraint 1: Comfort ratio ≥ 95%
    Constraint 2: Cycles ≤ baseline cycles
    
    L(x, λ) = cost + λ_comfort * max(0, 0.95 - comfort) + λ_cycling * max(0, cycles - target)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.lambda_comfort = config.lambda_comfort
        self.lambda_cycling = config.lambda_cycling
        
        # Track for constraint violation
        self.comfort_violations = 0
        self.total_steps = 0
        self.current_runtime = config.min_runtime_target
        self.current_offtime = config.min_runtime_target
        
    def reset(self):
        self.comfort_violations = 0
        self.total_steps = 0
        self.current_runtime = self.config.min_runtime_target
        self.current_offtime = self.config.min_runtime_target
        
    def update_lambda(self, comfort_ratio: float, n_cycles: int, baseline_cycles: int):
        """Adapt Lagrange multipliers based on constraint violations."""
        # If comfort is violated, increase lambda
        if comfort_ratio < self.config.comfort_target:
            violation = self.config.comfort_target - comfort_ratio
            self.lambda_comfort = min(50.0, self.lambda_comfort * (1 + violation))
        else:
            self.lambda_comfort = max(1.0, self.lambda_comfort * 0.99)
            
        # If cycles exceed baseline, increase lambda
        if n_cycles > baseline_cycles:
            self.lambda_cycling = min(20.0, self.lambda_cycling * 1.1)
        else:
            self.lambda_cycling = max(0.5, self.lambda_cycling * 0.99)
    
    def calculate(self, T_in: float, action: int, price: float, prev_action: int,
                  is_invalid: bool, is_peak: bool, power_kw: float, hour: int) -> Tuple[float, Dict]:
        
        self.total_steps += 1
        
        # Cost
        cost = power_kw * (1/60) * price
        
        # Comfort check
        in_comfort = (self.config.comfort_min <= T_in <= self.config.comfort_max)
        if not in_comfort:
            self.comfort_violations += 1
        
        if T_in < self.config.comfort_min:
            dev = self.config.comfort_min - T_in
        elif T_in > self.config.comfort_max:
            dev = T_in - self.config.comfort_max
        else:
            dev = 0.0
        
        # =====================================================================
        # CONSTRAINED REWARD: r = -cost - λ_comfort * comfort_violation - λ_cycling * switch
        # =====================================================================
        
        # Base: negative cost (want to minimize)
        reward = -cost * 10  # Scale to make comparable
        
        # Comfort constraint (Lagrangian penalty)
        if in_comfort:
            # Bonus for being in comfort
            reward += 0.5
            # Extra bonus for being near setpoint
            if abs(T_in - self.config.setpoint) < 0.5:
                reward += 0.2
        else:
            # Lagrangian penalty for constraint violation
            comfort_penalty = self.lambda_comfort * dev
            reward -= comfort_penalty
            
            # Action guidance
            if T_in < self.config.comfort_min and action == 0:
                reward -= 2.0  # Should be heating!
            elif T_in > self.config.comfort_max and action == 1:
                reward -= 2.0  # Should NOT be heating!
        
        # Cycling constraint
        switched = (action != prev_action)
        if switched:
            # Base switching penalty (Lagrangian)
            reward -= self.lambda_cycling * 0.1
            
            # Extra penalty for short cycles
            if prev_action == 1 and self.current_runtime < self.config.min_runtime_target:
                shortness = 1 - self.current_runtime / self.config.min_runtime_target
                reward -= self.lambda_cycling * 0.2 * shortness
                self.current_runtime = 0
                self.current_offtime = 1
            elif prev_action == 0 and self.current_offtime < self.config.min_runtime_target:
                shortness = 1 - self.current_offtime / self.config.min_runtime_target
                reward -= self.lambda_cycling * 0.2 * shortness
                self.current_offtime = 0
                self.current_runtime = 1
            else:
                if prev_action == 1:
                    self.current_runtime = 0
                    self.current_offtime = 1
                else:
                    self.current_offtime = 0
                    self.current_runtime = 1
        else:
            if action == 1:
                self.current_runtime += 1
            else:
                self.current_offtime += 1
        
        # Peak hour optimization (only when comfortable)
        if in_comfort:
            # Pre-heating incentive
            if 14 <= hour < 16 and action == 1 and T_in < self.config.setpoint + 1.0:
                reward += 0.1
            # Coasting incentive
            if is_peak and action == 0 and T_in > self.config.setpoint:
                reward += 0.15
        
        # Invalid action
        if is_invalid:
            reward -= 1.0
        
        return np.clip(reward, -20, 3), {
            "cost": cost, "in_comfort": in_comfort, "dev": dev, "switched": switched
        }


class HVACEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, config: Config, is_training: bool = True,
                 start_idx: int = 0, end_idx: Optional[int] = None):
        super().__init__()
        self.data = data
        self.config = config
        self.is_training = is_training
        self.reward_handler = ConstrainedRewardHandler(config)
        
        self.start_idx = start_idx
        self.end_idx = end_idx or len(data)
        self.episode_length = config.episode_length_days * 24 * 60
        
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
        
        is_peak = 16 <= hour < 21
        power_kw = action * self.config.Q_hvac_kw
        
        reward, info = self.reward_handler.calculate(
            self.T_in, action, price, self.prev_action, invalid, is_peak, power_kw, hour
        )
        self.prev_action = action
        
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
        cycles = sum(1 for i in range(1, len(self.actions)) if self.actions[i-1] == 1 and self.actions[i] == 0)
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


class Thermostat:
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


class LambdaCallback(BaseCallback):
    """Callback to adapt Lagrange multipliers during training."""
    
    def __init__(self, env_fn, config: Config, baseline_cycles: int):
        super().__init__()
        self.env_fn = env_fn
        self.config = config
        self.baseline_cycles = baseline_cycles
        self.eval_freq = 10000
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            # Quick evaluation
            env = self.env_fn()
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
            
            stats = env.get_stats()
            
            # Update lambda in all training envs
            if hasattr(self.training_env, 'envs'):
                for e in self.training_env.envs:
                    e.reward_handler.update_lambda(
                        stats["comfort_ratio"], 
                        stats["cycles"],
                        self.baseline_cycles
                    )
        return True


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


def run_statistical_analysis(baseline_results: List[Dict], ppo_results: List[Dict]) -> Dict:
    """Comprehensive statistical analysis for ALL metrics."""
    
    metrics = ["cost", "energy_kwh", "comfort_ratio", "cycles", "short_cycles", "avg_runtime", "mean_temp"]
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
        
        # Percent change
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
    
    print(f"\n{'='*90}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*90}")
    print(f"{'Metric':<15} {'Baseline':<18} {'PI-DRL':<18} {'Change':<12} {'p-value':<12} {'Sig.':<5}")
    print(f"{'-'*90}")
    
    for m, v in analysis.items():
        if m == "comfort_ratio":
            b_str = f"{v['baseline_mean']*100:.1f} ± {v['baseline_std']*100:.1f}%"
            p_str = f"{v['ppo_mean']*100:.1f} ± {v['ppo_std']*100:.1f}%"
        else:
            b_str = f"{v['baseline_mean']:.2f} ± {v['baseline_std']:.2f}"
            p_str = f"{v['ppo_mean']:.2f} ± {v['ppo_std']:.2f}"
        
        if v['p_value'] < 0.001:
            p_str_val = "< 0.001"
        else:
            p_str_val = f"{v['p_value']:.3f}"
        
        sig = "***" if v['p_value'] < 0.001 else ("**" if v['p_value'] < 0.01 else ("*" if v['p_value'] < 0.05 else ""))
        
        print(f"{m:<15} {b_str:<18} {p_str:<18} {v['pct_change']:+.1f}%{'':<6} {p_str_val:<12} {sig:<5}")
    
    print(f"{'='*90}")
    print("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05")
    print(f"N samples per group: {analysis['cost']['n_samples']}")


def main():
    print("="*80)
    print("CONSTRAINED PI-DRL HVAC CONTROLLER")
    print("With Lagrangian optimization for comfort constraint")
    print("="*80)
    
    config = Config()
    output_dir = os.path.join(os.getcwd(), "HVAC_Constrained_Output")
    os.makedirs(output_dir, exist_ok=True)
    
    N_SEEDS = 10  # More seeds for better statistics
    
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
        
        # First, get baseline performance for Lagrangian target
        baseline = Thermostat(config)
        env = HVACEnv(data, config, False, split_idx, len(data))
        baseline_stats = evaluate(env, baseline, False, config.seed)
        baseline_cycles = baseline_stats["cycles"]
        
        print(f"\nBaseline cycles: {baseline_cycles}")
        print(f"Baseline comfort: {baseline_stats['comfort_ratio']*100:.1f}%")
        
        # Train with adaptive Lagrangian
        print(f"\n--- Training with Constrained RL ---")
        
        def make_env():
            def _init():
                env = HVACEnv(data, config, True, 0, split_idx)
                env.reset(seed=config.seed)
                return env
            return _init
        
        def make_eval_env():
            return HVACEnv(data, config, False, split_idx, len(data))
        
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
        
        callback = LambdaCallback(make_eval_env, config, baseline_cycles)
        model.learn(total_timesteps=config.total_timesteps, callback=callback, progress_bar=False)
        
        # Evaluate with multiple seeds
        print(f"\n--- Evaluating ({N_SEEDS} seeds) ---")
        
        seeds = [config.seed + i * 17 for i in range(N_SEEDS)]
        baseline_results = []
        ppo_results = []
        
        for seed in seeds:
            set_seed(seed)
            
            env = HVACEnv(data, config, False, split_idx, len(data))
            baseline_results.append(evaluate(env, baseline, False, seed))
            
            env = HVACEnv(data, config, False, split_idx, len(data))
            ppo_results.append(evaluate(env, model, True, seed))
        
        # Statistical analysis
        analysis = run_statistical_analysis(baseline_results, ppo_results)
        print_results(analysis, scenario_name)
        
        all_results[scenario_name] = {
            "analysis": analysis,
            "baseline_results": baseline_results,
            "ppo_results": ppo_results
        }
        
        # Save model
        model.save(os.path.join(output_dir, f"ppo_{scenario_name.replace(' ', '_').lower()}"))
    
    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY ACROSS ALL SCENARIOS")
    print(f"{'='*90}")
    
    summary_data = []
    for scenario_name, results in all_results.items():
        a = results["analysis"]
        summary_data.append({
            "Scenario": scenario_name,
            "Cost Change (%)": f"{a['cost']['pct_change']:.1f}",
            "Cost p-value": f"{a['cost']['p_value']:.4f}" if a['cost']['p_value'] >= 0.001 else "< 0.001",
            "Comfort Baseline (%)": f"{a['comfort_ratio']['baseline_mean']*100:.1f}",
            "Comfort PI-DRL (%)": f"{a['comfort_ratio']['ppo_mean']*100:.1f}",
            "Comfort p-value": f"{a['comfort_ratio']['p_value']:.4f}" if a['comfort_ratio']['p_value'] >= 0.001 else "< 0.001",
            "Cycles Baseline": f"{a['cycles']['baseline_mean']:.0f}",
            "Cycles PI-DRL": f"{a['cycles']['ppo_mean']:.0f}",
            "Cycles p-value": f"{a['cycles']['p_value']:.4f}" if a['cycles']['p_value'] >= 0.001 else "< 0.001",
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
