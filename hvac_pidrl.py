#!/usr/bin/env python3
"""
Robust, Safety-Critical 2R2C PI-DRL Controller for Residential HVAC
=====================================================================

A complete end-to-end implementation that:
1. Downloads real HVAC data from GitHub
2. Implements a 2R2C thermal model with safety constraints
3. Trains a PPO-based PI-DRL controller with domain randomization
4. Evaluates against a baseline thermostat
5. Generates 7 figures and 4 tables automatically

Author: Senior Energy Systems ML Researcher
Date: December 4, 2025
"""

import os
import sys
import warnings
import io
import zipfile
from urllib.request import urlopen
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ==============================================================================
# 1. DATA ACQUISITION AND PREPROCESSING
# ==============================================================================

class DataPipeline:
    """Downloads and preprocesses real HVAC data from GitHub."""
    
    DATA_URL = "https://github.com/Fateme9977/P3/raw/main/data/dataverse_files.zip"
    
    def __init__(self):
        self.df_merged = None
        self.train_df = None
        self.test_df = None
        
    def download_and_extract(self) -> Dict[str, pd.DataFrame]:
        """Download and extract the dataset from GitHub."""
        print("=" * 80)
        print("STEP 1: DATA ACQUISITION")
        print("=" * 80)
        print(f"Downloading data from: {self.DATA_URL}")
        
        # Download the zip file
        response = urlopen(self.DATA_URL)
        zip_data = io.BytesIO(response.read())
        
        print("✓ Download complete. Extracting files...")
        
        # Extract and identify files
        dataframes = {}
        with zipfile.ZipFile(zip_data) as zf:
            for filename in zf.namelist():
                basename = os.path.basename(filename)
                
                # Look for the three required files
                if basename.startswith("Electricity_WHE"):
                    print(f"  Found: {basename} (Whole-Home Electricity)")
                    df = pd.read_csv(zf.open(filename))
                    dataframes['WHE'] = df
                    
                elif basename.startswith("Electricity_HPE"):
                    print(f"  Found: {basename} (Heat Pump Electricity)")
                    df = pd.read_csv(zf.open(filename))
                    dataframes['HPE'] = df
                    
                elif basename.startswith("Climate_HourlyWeathe"):
                    print(f"  Found: {basename} (Hourly Weather)")
                    df = pd.read_csv(zf.open(filename))
                    dataframes['Weather'] = df
        
        if len(dataframes) < 3:
            raise ValueError(f"Expected 3 files, found {len(dataframes)}")
        
        print("✓ Extraction complete.\n")
        return dataframes
    
    def process_data(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process and merge all dataframes."""
        print("=" * 80)
        print("STEP 2: DATA PROCESSING")
        print("=" * 80)
        
        # Process electricity data (WHE)
        df_whe = dataframes['WHE'].copy()
        # Convert unix timestamp to datetime
        df_whe['timestamp'] = pd.to_datetime(df_whe['unix_ts'], unit='s')
        df_whe = df_whe.set_index('timestamp')
        df_whe = df_whe[['P']].rename(columns={'P': 'P_whe'})
        
        # Process heat pump electricity (HPE)
        df_hpe = dataframes['HPE'].copy()
        df_hpe['timestamp'] = pd.to_datetime(df_hpe['unix_ts'], unit='s')
        df_hpe = df_hpe.set_index('timestamp')
        df_hpe = df_hpe[['P']].rename(columns={'P': 'P_hpe'})
        
        # Process weather data
        df_weather = dataframes['Weather'].copy()
        df_weather['timestamp'] = pd.to_datetime(df_weather['Date/Time'])
        df_weather = df_weather.set_index('timestamp')
        df_weather = df_weather[['Temp (C)']].rename(columns={'Temp (C)': 'T_out'})
        
        print(f"WHE data shape: {df_whe.shape}")
        print(f"HPE data shape: {df_hpe.shape}")
        print(f"Weather data shape: {df_weather.shape}")
        
        # Resample weather to 1-minute (linear interpolation)
        print("\nResampling weather data to 1-minute resolution...")
        df_weather_1min = df_weather.resample('1min').interpolate(method='linear')
        
        # Merge on timestamp index
        print("Merging datasets...")
        df_merged = df_whe.join(df_hpe, how='outer').join(df_weather_1min, how='outer')
        df_merged = df_merged.dropna()
        
        # Add TOU pricing
        df_merged['hour'] = df_merged.index.hour
        df_merged['Price'] = df_merged['hour'].apply(
            lambda h: 0.30 if 16 <= h < 21 else 0.10
        )
        
        # Add time features
        df_merged['time_sin'] = np.sin(2 * np.pi * df_merged['hour'] / 24)
        df_merged['time_cos'] = np.cos(2 * np.pi * df_merged['hour'] / 24)
        
        print(f"\n✓ Final merged dataset shape: {df_merged.shape}")
        print(f"  Date range: {df_merged.index.min()} to {df_merged.index.max()}")
        print(f"  Duration: {(df_merged.index.max() - df_merged.index.min()).days} days")
        print()
        
        return df_merged
    
    def split_train_test(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """Split data chronologically."""
        print("=" * 80)
        print("STEP 3: TRAIN/TEST SPLIT")
        print("=" * 80)
        
        split_idx = int(len(df) * train_ratio)
        self.train_df = df.iloc[:split_idx].copy()
        self.test_df = df.iloc[split_idx:].copy()
        
        print(f"Training set: {self.train_df.shape[0]:,} samples "
              f"({self.train_df.index.min()} to {self.train_df.index.max()})")
        print(f"Test set: {self.test_df.shape[0]:,} samples "
              f"({self.test_df.index.min()} to {self.test_df.index.max()})")
        print()
        
    def run(self):
        """Execute the full data pipeline."""
        dataframes = self.download_and_extract()
        self.df_merged = self.process_data(dataframes)
        self.split_train_test(self.df_merged)
        return self.train_df, self.test_df


# ==============================================================================
# 2. 2R2C THERMAL MODEL WITH SAFETY CONSTRAINTS
# ==============================================================================

class SafetyHVACEnv(gym.Env):
    """
    2R2C thermal model with safety constraints (15-min lockout).
    Implements domain randomization for robustness.
    """
    
    metadata = {'render.modes': []}
    
    # Nominal 2R2C parameters
    NOMINAL_PARAMS = {
        'R_i': 0.5,      # °C/kW (indoor to mass)
        'R_w': 0.3,      # °C/kW (mass to outdoor)
        'R_o': 0.2,      # °C/kW (indoor to outdoor)
        'C_in': 20.0,    # kWh/°C (indoor capacitance)
        'C_m': 50.0,     # kWh/°C (mass capacitance)
    }
    
    def __init__(self, 
                 data_df: pd.DataFrame,
                 episode_length_days: int = 7,
                 randomize: bool = False,
                 param_multipliers: Optional[Dict[str, float]] = None,
                 randomization_scale: float = 0.15,
                 w_cost: float = 1.0,
                 w_disc: float = 1.0,
                 w_switch: float = 0.2,
                 w_peak: float = 1.0,
                 w_invalid: float = 0.5):
        """
        Parameters
        ----------
        data_df : pd.DataFrame
            Data with columns: T_out, Price, time_sin, time_cos
        episode_length_days : int
            Length of each episode in days
        randomize : bool
            If True, randomize parameters each episode
        param_multipliers : dict, optional
            Custom parameter multipliers (for robustness testing)
        randomization_scale : float
            Scale for domain randomization (0.0, 0.10, 0.15)
        w_cost, w_disc, w_switch, w_peak, w_invalid : float
            Configurable reward weights
        """
        super().__init__()
        
        self.data_df = data_df.copy()
        self.episode_length_steps = episode_length_days * 24 * 60  # 1-min steps
        self.randomize = randomize
        self.custom_multipliers = param_multipliers
        self.randomization_scale = randomization_scale
        
        # State: [T_in_obs, T_out, T_mass, Price, time_sin, time_cos, mask_off, mask_on]
        self.observation_space = spaces.Box(
            low=np.array([-10, -30, -10, 0, -1, -1, 0, 0], dtype=np.float32),
            high=np.array([50, 50, 50, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: {0: OFF, 1: ON}
        self.action_space = spaces.Discrete(2)
        
        # HVAC parameters
        self.Q_hvac = 4.0  # kW (heating power when ON)
        self.dt = 60.0     # seconds (1 minute)
        self.dt_hours = self.dt / 3600.0
        
        # Observation noise
        self.obs_noise_std = 0.1  # °C
        
        # Comfort settings
        self.T_setpoint = 21.0
        self.T_comfort_min = 19.5
        self.T_comfort_max = 24.0
        
        # Lockout constraints
        self.lockout_minutes = 15
        
        # Reward weights (configurable)
        self.w_cost = w_cost
        self.w_disc = w_disc
        self.w_switch = w_switch
        self.w_peak = w_peak
        self.w_invalid = w_invalid
        
        # Normalization constants
        self.cost_scale = 0.01
        self.disc_scale = 1.0  # Reduced from 5.0 to make discomfort more impactful
        self.switch_scale = 1.0
        self.peak_scale = 0.01
        
        # Internal state
        self.reset()
        
        # Logging
        self.masked_off_count = 0
        self.masked_on_count = 0
        
    def _set_parameters(self):
        """Set 2R2C parameters (with optional randomization)."""
        if self.custom_multipliers is not None:
            # Use custom multipliers for robustness testing
            multipliers = self.custom_multipliers
        elif self.randomize:
            # Domain randomization with configurable scale
            scale = self.randomization_scale
            low = 1.0 - scale
            high = 1.0 + scale
            multipliers = {
                key: np.random.uniform(low, high)
                for key in self.NOMINAL_PARAMS.keys()
            }
        else:
            # Nominal parameters
            multipliers = {key: 1.0 for key in self.NOMINAL_PARAMS.keys()}
        
        self.params = {
            key: self.NOMINAL_PARAMS[key] * multipliers[key]
            for key in self.NOMINAL_PARAMS.keys()
        }
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Set parameters
        self._set_parameters()
        
        # Random starting position with peak hour coverage
        max_start = len(self.data_df) - self.episode_length_steps
        if max_start <= 0:
            # If data is shorter than episode length, use full data
            self.current_step = 0
            self.episode_length_steps = len(self.data_df)
        else:
            # Sample with bias towards episodes containing peak hours
            attempts = 0
            max_attempts = 20
            valid_start = False
            
            while attempts < max_attempts:
                candidate_start = np.random.randint(0, max_start)
                candidate_end = candidate_start + self.episode_length_steps
                
                # Check if episode contains peak hours (16-21)
                episode_hours = self.data_df.iloc[candidate_start:candidate_end].index.hour
                has_peak_hours = ((episode_hours >= 16) & (episode_hours < 21)).any()
                
                if has_peak_hours or attempts >= max_attempts - 1:
                    self.current_step = candidate_start
                    valid_start = True
                    break
                
                attempts += 1
            
            if not valid_start:
                self.current_step = np.random.randint(0, max_start)
        
        self.episode_start = self.current_step
        
        # Initialize thermal states
        T_out_init = self.data_df.iloc[self.current_step]['T_out']
        self.T_in = 21.0  # Start at comfortable temperature
        self.T_mass = 21.0
        
        # Safety lockout tracking (renamed for clarity)
        self.minutes_since_on = 0   # Minutes since compressor turned ON
        self.minutes_since_off = 20  # Minutes since compressor turned OFF (start with enough)
        self.last_action = 0
        
        # Backward compatibility
        self.runtime_minutes = self.minutes_since_on
        self.offtime_minutes = self.minutes_since_off
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of shape (2,), for actions [OFF, ON].
        True = action allowed, False = action forbidden by lockout.
        """
        allowed = np.array([True, True], dtype=bool)
        if self.minutes_since_on < self.lockout_minutes and self.minutes_since_on > 0:
            # Still in minimum ON time, cannot turn OFF
            allowed[0] = False   # OFF forbidden
        if self.minutes_since_off < self.lockout_minutes and self.minutes_since_off > 0:
            # Still in minimum OFF time, cannot turn ON
            allowed[1] = False   # ON forbidden
        return allowed
    
    def _get_observation(self) -> np.ndarray:
        """Get observation with noise on T_in and action mask."""
        row = self.data_df.iloc[self.current_step]
        
        # Add Gaussian noise to T_in
        T_in_obs = self.T_in + np.random.randn() * self.obs_noise_std
        
        # Get action mask
        mask = self.get_action_mask()
        
        obs = np.array([
            T_in_obs,
            row['T_out'],
            self.T_mass,
            row['Price'],
            row['time_sin'],
            row['time_cos'],
            float(mask[0]),  # OFF allowed
            float(mask[1])   # ON allowed
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: int):
        """Execute one timestep."""
        row = self.data_df.iloc[self.current_step]
        T_out = row['T_out']
        Price = row['Price']
        current_hour = row.name.hour if hasattr(row.name, 'hour') else self.data_df.index[self.current_step].hour
        
        # Get action mask
        action_mask = self.get_action_mask()
        original_action = action
        
        # Check if action is invalid (for penalty)
        invalid_action = not action_mask[action]
        
        # Apply safety lockout masking
        if not action_mask[action]:
            # Force to allowed action
            if action_mask[0]:
                action = 0  # Force OFF
                self.masked_off_count += 1
            elif action_mask[1]:
                action = 1  # Force ON
                self.masked_on_count += 1
        
        # Update lockout counters
        if self.last_action == 1:  # Was ON
            self.minutes_since_on += 1
            self.minutes_since_off = 0
        else:  # Was OFF
            self.minutes_since_off += 1
            self.minutes_since_on = 0
        
        # Reset counters on state change
        if action != self.last_action:
            if action == 1:  # Turning ON
                self.minutes_since_on = 1
                self.minutes_since_off = 0
            else:  # Turning OFF
                self.minutes_since_off = 1
                self.minutes_since_on = 0
        
        prev_action = self.last_action
        self.last_action = action
        
        # Backward compatibility
        self.runtime_minutes = self.minutes_since_on
        self.offtime_minutes = self.minutes_since_off
        
        # Compute HVAC power
        Q_hvac = self.Q_hvac if action == 1 else 0.0
        
        # 2R2C dynamics (Euler integration)
        R_i = self.params['R_i']
        R_w = self.params['R_w']
        R_o = self.params['R_o']
        C_in = self.params['C_in']
        C_m = self.params['C_m']
        
        dT_in = ((self.T_mass - self.T_in) / R_i + 
                 (T_out - self.T_in) / R_o + 
                 Q_hvac / C_in) * self.dt_hours
        
        dT_mass = ((self.T_in - self.T_mass) / R_i + 
                   (T_out - self.T_mass) / R_w) * self.dt_hours
        
        self.T_in += dT_in
        self.T_mass += dT_mass
        
        # Compute normalized reward components
        # 1. Energy cost
        power_kw = Q_hvac
        energy_kwh = power_kw * self.dt_hours
        instant_cost = energy_kwh * Price
        cost_norm = instant_cost / self.cost_scale
        
        # 2. Discomfort (use true indoor temp, not noisy observation)
        disc_raw = (self.T_in - self.T_setpoint) ** 2 * self.dt_hours
        disc_norm = disc_raw / self.disc_scale
        
        # 3. Switching penalty
        switch_raw = 1.0 if action != prev_action else 0.0
        switch_norm = switch_raw / self.switch_scale
        
        # 4. Peak hour penalty
        is_peak = 1.0 if 16 <= current_hour < 21 else 0.0
        peak_penalty_raw = is_peak * energy_kwh
        peak_norm = peak_penalty_raw / self.peak_scale
        
        # 5. Invalid action penalty
        invalid_penalty = 1.0 if invalid_action else 0.0
        
        # Combined reward
        reward = -(
            self.w_cost * cost_norm +
            self.w_disc * disc_norm +
            self.w_switch * switch_norm +
            self.w_peak * peak_norm +
            self.w_invalid * invalid_penalty
        )
        
        # Move to next timestep
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= self.episode_start + self.episode_length_steps or
                self.current_step >= len(self.data_df))
        
        # Get next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(8, dtype=np.float32)
        
        # Compute backward-compatible metrics for logging
        cost_t = instant_cost
        discomfort_t = (self.T_in - self.T_setpoint) ** 2
        penalty = 1.0 if self.T_in < self.T_comfort_min or self.T_in > self.T_comfort_max else 0.0
        
        info = {
            'T_in': self.T_in,
            'T_mass': self.T_mass,
            'cost': cost_t,
            'discomfort': discomfort_t,
            'penalty': penalty,
            'power': power_kw,
            'action_original': original_action,
            'action_final': action,
        }
        
        truncated = False
        
        return obs, reward, done, truncated, info


# ==============================================================================
# 3. BASELINE THERMOSTAT CONTROLLER
# ==============================================================================

class BaselineThermostat:
    """Traditional thermostat with deadband and lockout."""
    
    def __init__(self, 
                 setpoint: float = 21.0,
                 deadband: float = 1.5,
                 lockout_minutes: int = 15):
        self.setpoint = setpoint
        self.deadband = deadband
        self.lockout_minutes = lockout_minutes
        
        self.last_action = 0
        self.runtime_minutes = 0
        self.offtime_minutes = 20
        
    def reset(self):
        """Reset controller state."""
        self.last_action = 0
        self.runtime_minutes = 0
        self.offtime_minutes = 20
        
    def get_action(self, T_in: float) -> int:
        """Compute action based on temperature and lockout."""
        # Thermostat logic
        if T_in > self.setpoint + self.deadband:
            desired_action = 0  # OFF (too hot)
        elif T_in < self.setpoint - self.deadband:
            desired_action = 1  # ON (too cold)
        else:
            desired_action = self.last_action  # Maintain
        
        # Apply lockout
        action = desired_action
        if self.last_action == 1:  # Was ON
            self.runtime_minutes += 1
            if self.runtime_minutes < self.lockout_minutes and desired_action == 0:
                action = 1  # Keep ON
        
        if self.last_action == 0:  # Was OFF
            self.offtime_minutes += 1
            if self.offtime_minutes < self.lockout_minutes and desired_action == 1:
                action = 0  # Keep OFF
        
        # Update counters
        if action != self.last_action:
            if action == 1:
                self.runtime_minutes = 1
                self.offtime_minutes = 0
            else:
                self.offtime_minutes = 1
                self.runtime_minutes = 0
        
        self.last_action = action
        return action


# ==============================================================================
# 4. TRAINING AND EVALUATION
# ==============================================================================

def train_pi_drl(train_df: pd.DataFrame, 
                 total_timesteps: int = 100_000) -> PPO:
    """Train PPO agent with curriculum learning and domain randomization."""
    print("=" * 80)
    print("STEP 4: TRAINING PI-DRL CONTROLLER WITH CURRICULUM")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Curriculum: Phase 1 (nominal) → Phase 2 (±10%) → Phase 3 (±15%)")
    print()
    
    # Curriculum phases
    phases = [
        {'name': 'Phase 1 (Nominal)', 'scale': 0.0, 'timesteps': total_timesteps // 3},
        {'name': 'Phase 2 (±10%)', 'scale': 0.10, 'timesteps': total_timesteps // 3},
        {'name': 'Phase 3 (±15%)', 'scale': 0.15, 'timesteps': total_timesteps - 2 * (total_timesteps // 3)},
    ]
    
    model = None
    
    for phase_idx, phase in enumerate(phases):
        print(f"\n{'='*60}")
        print(f"  {phase['name']}: randomization_scale={phase['scale']}")
        print(f"  Training for {phase['timesteps']:,} timesteps")
        print(f"{'='*60}")
        
        # Create training environment with current randomization scale
        def make_env():
            return SafetyHVACEnv(
                train_df, 
                episode_length_days=7, 
                randomize=True,
                randomization_scale=phase['scale'],
                w_cost=1.0,
                w_disc=5.0,  # Increase discomfort weight
                w_switch=0.1,  # Reduce switching penalty
                w_peak=0.5,  # Reduce peak penalty
                w_invalid=0.2  # Reduce invalid penalty
            )
        
        env = DummyVecEnv([make_env])
        
        if model is None:
            # Initialize PPO for first phase
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=2e-4,  # Reduced from 3e-4
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.15,  # Reduced from 0.2
                ent_coef=0.05,  # Increased from 0.01 to encourage exploration
                verbose=1,
                device='cpu'
            )
        else:
            # Continue training with new environment
            model.set_env(env)
        
        # Train for this phase
        model.learn(total_timesteps=phase['timesteps'], reset_num_timesteps=False)
    
    print("\n✓ Curriculum training complete.\n")
    
    return model


def evaluate_controller(controller, 
                        test_df: pd.DataFrame,
                        controller_name: str,
                        use_model: bool = False) -> Dict:
    """Evaluate a controller on the test set."""
    print(f"Evaluating {controller_name}...")
    
    # Create test environment (no randomization, nominal parameters)
    env = SafetyHVACEnv(test_df, episode_length_days=9999, randomize=False, randomization_scale=0.0)
    env.episode_length_steps = len(test_df)  # Use full test set
    
    obs, _ = env.reset()
    
    # Data collection
    results = {
        'timestamps': [],
        'T_in': [],
        'T_out': [],
        'T_mass': [],
        'actions': [],
        'power': [],
        'cost': [],
        'discomfort': [],
        'penalty': [],
        'Price': [],
        'hour': [],
    }
    
    if not use_model:
        controller.reset()
    
    done = False
    step = 0
    
    while not done and step < len(test_df):
        # Get action
        if use_model:
            action, _ = controller.predict(obs, deterministic=True)
            action = int(action)
        else:
            # Baseline thermostat uses true T_in (first element of obs, but noisy)
            # For fairness, use the true T_in from env
            action = controller.get_action(env.T_in)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Record
        results['timestamps'].append(test_df.index[step])
        results['T_in'].append(info['T_in'])
        results['T_out'].append(test_df.iloc[step]['T_out'])
        results['T_mass'].append(info['T_mass'])
        results['actions'].append(info['action_final'])
        results['power'].append(info['power'])
        results['cost'].append(info['cost'])
        results['discomfort'].append(info['discomfort'])
        results['penalty'].append(info['penalty'])
        results['Price'].append(test_df.iloc[step]['Price'])
        results['hour'].append(test_df.index[step].hour)
        
        step += 1
        
        if done or truncated:
            break
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df['timestamps'] = pd.to_datetime(results_df['timestamps'])
    results_df = results_df.set_index('timestamps')
    
    # Compute summary metrics
    total_cost = results_df['cost'].sum()
    total_discomfort = results_df['discomfort'].sum()
    total_energy = results_df['power'].sum() / 60.0  # kWh
    
    # Count cycles (OFF->ON transitions)
    cycles = (np.diff(results_df['actions'].values, prepend=0) == 1).sum()
    
    # Peak power
    peak_hours_mask = (results_df['hour'] >= 16) & (results_df['hour'] < 21)
    avg_power_peak = results_df.loc[peak_hours_mask, 'power'].mean()
    avg_power_offpeak = results_df.loc[~peak_hours_mask, 'power'].mean()
    
    summary = {
        'total_cost': total_cost,
        'total_discomfort': total_discomfort,
        'total_energy_kwh': total_energy,
        'total_cycles': cycles,
        'avg_power_peak': avg_power_peak,
        'avg_power_offpeak': avg_power_offpeak,
        'masked_off': env.masked_off_count,
        'masked_on': env.masked_on_count,
    }
    
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Total energy: {total_energy:.2f} kWh")
    print(f"  Total cycles: {cycles}")
    print(f"  Masked actions: {env.masked_off_count + env.masked_on_count}")
    print()
    
    return results_df, summary


def robustness_test(controller, 
                    test_df: pd.DataFrame,
                    r_multipliers: List[float],
                    use_model: bool = False) -> Dict:
    """Test robustness by varying R parameters."""
    print("Running robustness analysis...")
    
    results = {}
    
    for r_mult in r_multipliers:
        print(f"  Testing R multiplier = {r_mult:.1f}...")
        
        # Create environment with custom multipliers
        multipliers = {
            'R_i': r_mult,
            'R_w': r_mult,
            'R_o': r_mult,
            'C_in': 1.0,
            'C_m': 1.0,
        }
        
        env = SafetyHVACEnv(
            test_df, 
            episode_length_days=9999, 
            randomize=False,
            param_multipliers=multipliers,
            randomization_scale=0.0
        )
        env.episode_length_steps = len(test_df)
        
        obs, _ = env.reset()
        
        if not use_model:
            controller.reset()
        
        total_cost = 0.0
        done = False
        step = 0
        
        while not done and step < len(test_df):
            if use_model:
                action, _ = controller.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = controller.get_action(env.T_in)
            
            obs, reward, done, truncated, info = env.step(action)
            total_cost += info['cost']
            step += 1
            
            if done or truncated:
                break
        
        results[r_mult] = total_cost
    
    print()
    return results


# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================

def create_all_figures(baseline_df: pd.DataFrame,
                       pidrl_df: pd.DataFrame,
                       baseline_summary: Dict,
                       pidrl_summary: Dict,
                       pidrl_model: PPO,
                       test_df: pd.DataFrame,
                       output_dir: str = "output"):
    """Generate all 7 figures."""
    print("=" * 80)
    print("STEP 5: GENERATING FIGURES")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Micro Dynamics (4-hour zoom)
    print("Creating Figure 1: Micro Dynamics...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Select 4-hour window from middle of test set
    start_idx = len(baseline_df) // 2
    end_idx = start_idx + 240  # 4 hours * 60 minutes
    
    zoom_baseline = baseline_df.iloc[start_idx:end_idx]
    zoom_pidrl = pidrl_df.iloc[start_idx:end_idx]
    
    # Temperature subplot
    ax = axes[0]
    ax.plot(zoom_baseline.index, zoom_baseline['T_in'], 
            label='Baseline', linewidth=1.5, alpha=0.8)
    ax.plot(zoom_pidrl.index, zoom_pidrl['T_in'], 
            label='PI-DRL', linewidth=1.5, alpha=0.8)
    ax.axhline(21.0, color='black', linestyle='--', linewidth=1, label='Setpoint')
    ax.axhspan(19.5, 24.0, alpha=0.1, color='green', label='Comfort Band')
    ax.set_ylabel('Indoor Temperature (°C)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('4-Hour Micro-Dynamics: Temperature & Actions', fontsize=13, fontweight='bold')
    
    # Actions subplot
    ax = axes[1]
    ax.fill_between(zoom_baseline.index, 0, zoom_baseline['actions'], 
                     alpha=0.5, label='Baseline', step='post')
    ax.fill_between(zoom_pidrl.index, 0, zoom_pidrl['actions'], 
                     alpha=0.5, label='PI-DRL', step='post')
    ax.set_ylabel('HVAC Action (0=OFF, 1=ON)', fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_1_Micro_Dynamics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/Figure_1_Micro_Dynamics.png")
    
    # Figure 2: Safety Verification (Runtime Histograms)
    print("Creating Figure 2: Safety Verification...")
    
    # Extract runtime durations
    def get_runtime_durations(actions):
        runtimes = []
        current_runtime = 0
        for action in actions:
            if action == 1:
                current_runtime += 1
            else:
                if current_runtime > 0:
                    runtimes.append(current_runtime)
                current_runtime = 0
        if current_runtime > 0:
            runtimes.append(current_runtime)
        return runtimes
    
    baseline_runtimes = get_runtime_durations(baseline_df['actions'].values)
    pidrl_runtimes = get_runtime_durations(pidrl_df['actions'].values)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline histogram
    ax = axes[0]
    ax.hist(baseline_runtimes, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(15, color='red', linestyle='--', linewidth=2, label='15-min Lockout')
    ax.set_xlabel('Runtime Duration (minutes)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Baseline Thermostat', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # PI-DRL histogram
    ax = axes[1]
    ax.hist(pidrl_runtimes, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(15, color='red', linestyle='--', linewidth=2, label='15-min Lockout')
    ax.set_xlabel('Runtime Duration (minutes)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('PI-DRL Controller', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Safety Verification: ON-Runtime Distributions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_2_Safety_Verification.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/Figure_2_Safety_Verification.png")
    
    # Figure 3: Policy Heatmap (P(ON) vs hour and T_out)
    print("Creating Figure 3: Policy Heatmap...")
    
    # Create grid for policy evaluation
    hours = np.arange(0, 24)
    temps_out = np.linspace(test_df['T_out'].min(), test_df['T_out'].max(), 20)
    
    prob_grid = np.zeros((len(temps_out), len(hours)))
    
    for i, T_out in enumerate(temps_out):
        for j, hour in enumerate(hours):
            # Create synthetic observation (8-dim with action masks)
            time_sin = np.sin(2 * np.pi * hour / 24)
            time_cos = np.cos(2 * np.pi * hour / 24)
            obs = np.array([21.0, T_out, 21.0, 0.1, time_sin, time_cos, 1.0, 1.0], dtype=np.float32)
            
            # Query policy multiple times for stochasticity
            actions = []
            for _ in range(10):
                action, _ = pidrl_model.predict(obs, deterministic=False)
                actions.append(int(action))
            
            prob_grid[i, j] = np.mean(actions)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(prob_grid, aspect='auto', cmap='RdYlGn', origin='lower',
                   extent=[0, 24, temps_out[0], temps_out[-1]], vmin=0, vmax=1)
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Outdoor Temperature (°C)', fontsize=11)
    ax.set_title('PI-DRL Policy Heatmap: P(HVAC=ON)', fontsize=13, fontweight='bold')
    
    # Mark peak hours
    ax.axvspan(16, 21, alpha=0.2, color='red', label='Peak Hours')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability of HVAC ON', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_3_Policy_Heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/Figure_3_Policy_Heatmap.png")
    
    # Figure 4: Multi-Objective Radar Chart
    print("Creating Figure 4: Multi-Objective Radar...")
    
    # Normalize metrics (higher = better after transformation)
    metrics = ['Cost\nSavings', 'Comfort', 'Cycle\nReduction', 
               'Energy\nEfficiency', 'Peak\nReduction']
    
    # Transform so higher is better
    baseline_cost = baseline_summary['total_cost']
    pidrl_cost = pidrl_summary['total_cost']
    cost_savings = (baseline_cost - pidrl_cost) / baseline_cost if baseline_cost > 0 else 0
    
    baseline_discomfort = baseline_summary['total_discomfort']
    pidrl_discomfort = pidrl_summary['total_discomfort']
    comfort_improvement = (baseline_discomfort - pidrl_discomfort) / baseline_discomfort if baseline_discomfort > 0 else 0
    
    baseline_cycles = baseline_summary['total_cycles']
    pidrl_cycles = pidrl_summary['total_cycles']
    cycle_reduction = (baseline_cycles - pidrl_cycles) / baseline_cycles if baseline_cycles > 0 else 0
    
    baseline_energy = baseline_summary['total_energy_kwh']
    pidrl_energy = pidrl_summary['total_energy_kwh']
    energy_savings = (baseline_energy - pidrl_energy) / baseline_energy if baseline_energy > 0 else 0
    
    baseline_peak = baseline_summary['avg_power_peak']
    pidrl_peak = pidrl_summary['avg_power_peak']
    peak_reduction = (baseline_peak - pidrl_peak) / baseline_peak if baseline_peak > 0 else 0
    
    # Clip to [0, 1] and scale for visibility
    baseline_scores = [0.0, 0.0, 0.0, 0.0, 0.0]  # Reference
    pidrl_scores = [
        max(0, min(1, cost_savings)),
        max(0, min(1, comfort_improvement)),
        max(0, min(1, cycle_reduction)),
        max(0, min(1, energy_savings)),
        max(0, min(1, peak_reduction))
    ]
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    baseline_scores += baseline_scores[:1]
    pidrl_scores += pidrl_scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, baseline_scores, 'o-', linewidth=2, label='Baseline', color='steelblue')
    ax.fill(angles, baseline_scores, alpha=0.25, color='steelblue')
    ax.plot(angles, pidrl_scores, 'o-', linewidth=2, label='PI-DRL', color='orange')
    ax.fill(angles, pidrl_scores, alpha=0.25, color='orange')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Multi-Objective Performance Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_4_Multi_Objective_Radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/Figure_4_Multi_Objective_Radar.png")
    
    # Figure 5: Robustness Analysis
    print("Creating Figure 5: Robustness...")
    
    r_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    baseline_controller = BaselineThermostat()
    baseline_robustness = robustness_test(baseline_controller, test_df, r_multipliers, use_model=False)
    pidrl_robustness = robustness_test(pidrl_model, test_df, r_multipliers, use_model=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_multipliers, [baseline_robustness[r] for r in r_multipliers], 
            'o-', linewidth=2, markersize=8, label='Baseline', color='steelblue')
    ax.plot(r_multipliers, [pidrl_robustness[r] for r in r_multipliers], 
            's-', linewidth=2, markersize=8, label='PI-DRL', color='orange')
    ax.set_xlabel('Thermal Resistance Multiplier', fontsize=11)
    ax.set_ylabel('Total Cost ($)', fontsize=11)
    ax.set_title('Robustness to Parameter Uncertainty', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_5_Robustness.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/Figure_5_Robustness.png")
    
    # Figure 6: Comfort Distribution
    print("Creating Figure 6: Comfort Distribution...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = [1, 2]
    data_to_plot = [baseline_df['T_in'].values, pidrl_df['T_in'].values]
    
    parts = ax.violinplot(data_to_plot, positions=positions, widths=0.7,
                          showmeans=True, showextrema=True)
    
    for pc in parts['bodies']:
        pc.set_alpha(0.7)
    
    ax.axhspan(19.5, 24.0, alpha=0.1, color='green', label='Comfort Band')
    ax.axhline(21.0, color='black', linestyle='--', linewidth=1, label='Setpoint')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Baseline', 'PI-DRL'], fontsize=11)
    ax.set_ylabel('Indoor Temperature (°C)', fontsize=11)
    ax.set_title('Temperature Distribution: Comfort Analysis', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_6_Comfort_Distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/Figure_6_Comfort_Distribution.png")
    
    # Figure 7: Price Response
    print("Creating Figure 7: Price Response...")
    
    baseline_hourly = baseline_df.groupby('hour')['power'].mean()
    pidrl_hourly = pidrl_df.groupby('hour')['power'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hours_x = np.arange(24)
    width = 0.35
    
    ax.bar(hours_x - width/2, baseline_hourly.values, width, 
           label='Baseline', alpha=0.8, color='steelblue')
    ax.bar(hours_x + width/2, pidrl_hourly.values, width, 
           label='PI-DRL', alpha=0.8, color='orange')
    
    # Highlight peak hours
    ax.axvspan(16, 21, alpha=0.15, color='red', label='Peak Hours (16:00-21:00)')
    
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Average Power (kW)', fontsize=11)
    ax.set_title('Price-Responsive Load Profile', fontsize=13, fontweight='bold')
    ax.set_xticks(hours_x)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_7_Price_Response.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir}/Figure_7_Price_Response.png")
    
    print("\n✓ All figures generated successfully.\n")


# ==============================================================================
# 6. TABLE GENERATION
# ==============================================================================

def create_all_tables(baseline_summary: Dict,
                      pidrl_summary: Dict,
                      pidrl_model: PPO,
                      train_env: SafetyHVACEnv,
                      test_env: SafetyHVACEnv,
                      output_dir: str = "output"):
    """Generate all 4 tables."""
    print("=" * 80)
    print("STEP 6: GENERATING TABLES")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Table 1: System Parameters
    print("Creating Table 1: System Parameters...")
    
    params_data = {
        'Parameter': [
            'R_i (Indoor-Mass Resistance)',
            'R_w (Mass-Outdoor Resistance)',
            'R_o (Indoor-Outdoor Resistance)',
            'C_in (Indoor Capacitance)',
            'C_m (Mass Capacitance)',
            'Q_hvac (Heating Power)',
            'Lockout Time',
            'Setpoint Temperature',
            'Deadband',
            'PPO Learning Rate',
            'PPO Batch Size',
            'PPO Gamma',
            'Reward: lambda_cost',
            'Reward: lambda_discomfort',
            'Reward: lambda_penalty',
        ],
        'Value': [
            '0.5',
            '0.3',
            '0.2',
            '20.0',
            '50.0',
            '4.0',
            '15',
            '21.0',
            '±1.5',
            '3e-4',
            '64',
            '0.99',
            '1.0',
            '50.0',
            '10.0',
        ],
        'Unit': [
            '°C/kW',
            '°C/kW',
            '°C/kW',
            'kWh/°C',
            'kWh/°C',
            'kW',
            'minutes',
            '°C',
            '°C',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
        ]
    }
    
    table1 = pd.DataFrame(params_data)
    table1.to_csv(f"{output_dir}/Table_1_System_Parameters.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/Table_1_System_Parameters.csv")
    
    # Table 2: Performance Summary
    print("Creating Table 2: Performance Summary...")
    
    perf_data = {
        'Controller': ['Baseline', 'PI-DRL'],
        'Total_Cost_$': [
            f"{baseline_summary['total_cost']:.2f}",
            f"{pidrl_summary['total_cost']:.2f}"
        ],
        'Total_Discomfort': [
            f"{baseline_summary['total_discomfort']:.2f}",
            f"{pidrl_summary['total_discomfort']:.2f}"
        ],
        'Total_Cycles': [
            baseline_summary['total_cycles'],
            pidrl_summary['total_cycles']
        ],
        'Total_Energy_kWh': [
            f"{baseline_summary['total_energy_kwh']:.2f}",
            f"{pidrl_summary['total_energy_kwh']:.2f}"
        ]
    }
    
    table2 = pd.DataFrame(perf_data)
    table2.to_csv(f"{output_dir}/Table_2_Performance_Summary.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/Table_2_Performance_Summary.csv")
    
    # Table 3: Grid Impact
    print("Creating Table 3: Grid Impact...")
    
    baseline_peak = baseline_summary['avg_power_peak']
    pidrl_peak = pidrl_summary['avg_power_peak']
    peak_reduction_pct = ((baseline_peak - pidrl_peak) / baseline_peak * 100) if baseline_peak > 0 else 0
    
    grid_data = {
        'Controller': ['Baseline', 'PI-DRL'],
        'Avg_Power_Peak_kW': [
            f"{baseline_summary['avg_power_peak']:.3f}",
            f"{pidrl_summary['avg_power_peak']:.3f}"
        ],
        'Avg_Power_OffPeak_kW': [
            f"{baseline_summary['avg_power_offpeak']:.3f}",
            f"{pidrl_summary['avg_power_offpeak']:.3f}"
        ],
        'Peak_Reduction_%': [
            '0.0',
            f"{peak_reduction_pct:.2f}"
        ]
    }
    
    table3 = pd.DataFrame(grid_data)
    table3.to_csv(f"{output_dir}/Table_3_Grid_Impact.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/Table_3_Grid_Impact.csv")
    
    # Table 4: Safety Shield Activity
    print("Creating Table 4: Safety Shield Activity...")
    
    # For training, we'd need to track this during training (simplified here)
    # Using test metrics as proxy
    total_test_steps = len(test_env.data_df)
    
    shield_data = {
        'Phase': ['Training (Estimate)', 'Testing'],
        'Total_Timesteps': [
            '500,000',
            f"{total_test_steps:,}"
        ],
        'Masked_OFF_Actions': [
            'N/A',  # Would need to track during training
            pidrl_summary['masked_off']
        ],
        'Masked_ON_Actions': [
            'N/A',
            pidrl_summary['masked_on']
        ],
        'Shield_Active_%': [
            'N/A',
            f"{((pidrl_summary['masked_off'] + pidrl_summary['masked_on']) / total_test_steps * 100):.2f}"
        ]
    }
    
    table4 = pd.DataFrame(shield_data)
    table4.to_csv(f"{output_dir}/Table_4_Safety_Shield_Activity.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/Table_4_Safety_Shield_Activity.csv")
    
    print("\n✓ All tables generated successfully.\n")


# ==============================================================================
# 7. MAIN PIPELINE
# ==============================================================================

def main():
    """Execute the complete pipeline."""
    print("\n" + "=" * 80)
    print(" ROBUST, SAFETY-CRITICAL 2R2C PI-DRL CONTROLLER FOR RESIDENTIAL HVAC")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Step 1-3: Data Pipeline
    pipeline = DataPipeline()
    train_df, test_df = pipeline.run()
    
    # Step 4: Train PI-DRL Controller
    pidrl_model = train_pi_drl(train_df, total_timesteps=100_000)
    
    # Step 5: Evaluate Controllers
    print("=" * 80)
    print("STEP 5: EVALUATION")
    print("=" * 80)
    
    # Baseline
    baseline_controller = BaselineThermostat()
    baseline_df, baseline_summary = evaluate_controller(
        baseline_controller, test_df, "Baseline Thermostat", use_model=False
    )
    
    # PI-DRL
    pidrl_df, pidrl_summary = evaluate_controller(
        pidrl_model, test_df, "PI-DRL Controller", use_model=True
    )
    
    # Step 6: Generate Figures
    train_env = SafetyHVACEnv(train_df, episode_length_days=7, randomize=True, randomization_scale=0.15)
    test_env = SafetyHVACEnv(test_df, episode_length_days=9999, randomize=False, randomization_scale=0.0)
    
    create_all_figures(
        baseline_df, pidrl_df, baseline_summary, pidrl_summary,
        pidrl_model, test_df, output_dir="output"
    )
    
    # Step 7: Generate Tables
    create_all_tables(
        baseline_summary, pidrl_summary, pidrl_model,
        train_env, test_env, output_dir="output"
    )
    
    # Final Summary
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\n📊 Summary of Results:")
    print(f"  Baseline Cost:  ${baseline_summary['total_cost']:.2f}")
    print(f"  PI-DRL Cost:    ${pidrl_summary['total_cost']:.2f}")
    print(f"  Cost Savings:   ${baseline_summary['total_cost'] - pidrl_summary['total_cost']:.2f} "
          f"({(baseline_summary['total_cost'] - pidrl_summary['total_cost']) / baseline_summary['total_cost'] * 100:.1f}%)")
    print(f"\n  Baseline Cycles: {baseline_summary['total_cycles']}")
    print(f"  PI-DRL Cycles:   {pidrl_summary['total_cycles']}")
    print(f"  Cycle Reduction: {baseline_summary['total_cycles'] - pidrl_summary['total_cycles']} "
          f"({(baseline_summary['total_cycles'] - pidrl_summary['total_cycles']) / baseline_summary['total_cycles'] * 100:.1f}%)")
    print(f"\n  Baseline Peak Power:  {baseline_summary['avg_power_peak']:.3f} kW")
    print(f"  PI-DRL Peak Power:    {pidrl_summary['avg_power_peak']:.3f} kW")
    print(f"  Peak Reduction:       {(baseline_summary['avg_power_peak'] - pidrl_summary['avg_power_peak']) / baseline_summary['avg_power_peak'] * 100:.1f}%")
    
    print("\n📁 Output Files Generated:")
    print("  Figures:")
    for i in range(1, 8):
        fig_names = [
            "Micro_Dynamics",
            "Safety_Verification",
            "Policy_Heatmap",
            "Multi_Objective_Radar",
            "Robustness",
            "Comfort_Distribution",
            "Price_Response"
        ]
        print(f"    ✓ output/Figure_{i}_{fig_names[i-1]}.png")
    
    print("\n  Tables:")
    table_names = [
        "System_Parameters",
        "Performance_Summary",
        "Grid_Impact",
        "Safety_Shield_Activity"
    ]
    for i, name in enumerate(table_names, 1):
        print(f"    ✓ output/Table_{i}_{name}.csv")
    
    print("\n" + "=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
