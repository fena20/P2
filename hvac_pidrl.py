#!/usr/bin/env python3
"""
Robust, Safety-Critical 2R2C PI-DRL Controller for Residential HVAC
================================================================================
A complete, execution-ready implementation that:
1. Downloads and processes real HVAC data from GitHub
2. Implements a 2R2C thermal model with domain randomization
3. Trains a PPO-based PI-DRL controller with safety constraints
4. Compares against a baseline thermostat
5. Generates 7 publication-quality figures and 4 tables

Author: Energy Systems ML Researcher
Date: 2025
"""

import os
import io
import zipfile
import requests
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings('ignore')

# Set publication-quality plotting style
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

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """Configuration parameters for the PI-DRL HVAC system."""
    # Data
    data_url: str = "https://github.com/Fateme9977/P3/raw/main/data/dataverse_files.zip"
    
    # 2R2C Nominal Parameters (realistic residential building values)
    # Typical residential R-values: 0.001-0.01 K/W for building envelope
    # Typical C-values: 1e6-1e7 J/K for building thermal mass
    # These are tuned so 4kW heating can maintain ~21°C when T_out ~ 5-10°C
    R_i: float = 0.0005  # Thermal resistance indoor-mass (K/W) - fast heat transfer
    R_w: float = 0.003  # Thermal resistance wall (K/W) - good insulation
    R_o: float = 0.002  # Thermal resistance outdoor (K/W)
    C_in: float = 500000.0  # Thermal capacitance indoor air (J/K) ~0.5 MJ/K
    C_m: float = 2000000.0  # Thermal capacitance building mass (J/K) ~2 MJ/K
    
    # HVAC
    Q_hvac_max: float = 4000.0  # Heating power (W) = 4 kW
    dt: float = 60.0  # Time step (seconds)
    
    # Thermostat
    setpoint: float = 21.0  # Setpoint temperature (°C)
    deadband: float = 1.5  # Deadband (°C)
    comfort_min: float = 19.5  # Comfort band minimum (°C)
    comfort_max: float = 24.0  # Comfort band maximum (°C)
    lockout_time: int = 15  # Minimum runtime/offtime (minutes)
    
    # TOU Pricing
    peak_price: float = 0.30  # $/kWh during peak (16:00-21:00)
    offpeak_price: float = 0.10  # $/kWh off-peak
    
    # Domain Randomization
    param_variation: float = 0.15  # ±15% variation for R and C parameters
    
    # Training
    total_timesteps: int = 200000  # Training steps
    episode_length_days: int = 1  # 1-day episodes
    train_split: float = 0.8
    
    # Reward Weights (comfort-first approach)
    lambda_cost: float = 0.5  # Only applies during peak when comfortable
    lambda_discomfort: float = 1.0  # Not used in new reward
    lambda_penalty: float = 1.0  # Not used in new reward
    
    # PPO Hyperparameters
    learning_rate: float = 3e-4  # Standard PPO learning rate
    gamma: float = 0.99  # Standard discount
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01  # Standard entropy
    clip_range: float = 0.2
    
    # Observation Noise
    obs_noise_std: float = 0.1  # Gaussian noise std for T_in observation


# =============================================================================
# DATA DOWNLOAD AND PROCESSING
# =============================================================================
def download_and_process_data(config: Config) -> pd.DataFrame:
    """
    Download the dataset from GitHub and process it into a unified DataFrame.
    
    Returns:
        DataFrame with columns: T_out, WHE, HPE, Price, hour, time_sin, time_cos
    """
    print("="*80)
    print("PHASE 1: Data Download and Processing")
    print("="*80)
    
    print(f"Downloading data from: {config.data_url}")
    
    try:
        response = requests.get(config.data_url, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download data: {e}")
        print("Generating synthetic data as fallback...")
        return generate_synthetic_data(config)
    
    # Load zip file in memory
    zip_buffer = io.BytesIO(response.content)
    
    # Initialize data containers
    df_whe = None
    df_hpe = None
    df_weather = None
    
    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Found {len(file_list)} files in archive")
        
        for filename in file_list:
            basename = os.path.basename(filename)
            
            # Skip directories and hidden files
            if not basename or basename.startswith('.'):
                continue
            
            # Identify files by their basename prefix
            if basename.startswith("Electricity_WHE"):
                print(f"  Loading: {basename}")
                with zip_ref.open(filename) as f:
                    df_whe = pd.read_csv(f)
                    
            elif basename.startswith("Electricity_HPE"):
                print(f"  Loading: {basename}")
                with zip_ref.open(filename) as f:
                    df_hpe = pd.read_csv(f)
                    
            elif basename.startswith("Climate_HourlyWeathe"):
                print(f"  Loading: {basename}")
                with zip_ref.open(filename) as f:
                    df_weather = pd.read_csv(f)
    
    # Check if we got all required files
    if df_whe is None or df_hpe is None or df_weather is None:
        print("Warning: Could not find all required files. Generating synthetic data...")
        return generate_synthetic_data(config)
    
    print(f"\nLoaded data shapes:")
    print(f"  WHE: {df_whe.shape}")
    print(f"  HPE: {df_hpe.shape}")
    print(f"  Weather: {df_weather.shape}")
    
    # Process timestamps for each dataframe
    # WHE and HPE have 'unix_ts' column (Unix timestamp)
    # Weather has 'Date/Time' column
    
    # Process electricity data (unix timestamp in seconds)
    if 'unix_ts' in df_whe.columns:
        df_whe['datetime'] = pd.to_datetime(df_whe['unix_ts'], unit='s')
        df_whe = df_whe.set_index('datetime').sort_index()
    else:
        df_whe = process_timestamp(df_whe, 'WHE')
    
    if 'unix_ts' in df_hpe.columns:
        df_hpe['datetime'] = pd.to_datetime(df_hpe['unix_ts'], unit='s')
        df_hpe = df_hpe.set_index('datetime').sort_index()
    else:
        df_hpe = process_timestamp(df_hpe, 'HPE')
    
    # Process weather data
    if 'Date/Time' in df_weather.columns:
        df_weather['datetime'] = pd.to_datetime(df_weather['Date/Time'])
        df_weather = df_weather.set_index('datetime').sort_index()
    else:
        df_weather = process_timestamp(df_weather, 'Weather')
    
    # Get the temperature column - 'Temp (C)' is the column name
    temp_col = None
    for col in df_weather.columns:
        if col == 'Temp (C)' or 'temp' in col.lower():
            temp_col = col
            break
    
    if temp_col is None:
        # Use first column that contains numeric temperature-like data
        for col in df_weather.columns:
            if df_weather[col].dtype in [np.float64, np.int64]:
                # Check if values are in reasonable temperature range
                if df_weather[col].mean() > -50 and df_weather[col].mean() < 50:
                    temp_col = col
                    break
    
    if temp_col is None:
        print("Warning: No temperature column found. Using synthetic weather.")
        return generate_synthetic_data(config)
    
    print(f"Using temperature column: '{temp_col}'")
    
    # Limit data size for faster processing (use 30 days = 43200 minutes)
    max_samples = 43200  # 30 days at 1-min resolution
    
    # Resample weather to 1-minute resolution using interpolation
    print("\nResampling weather data to 1-minute resolution...")
    df_weather_temp = df_weather[[temp_col]].copy()
    df_weather_temp = df_weather_temp.resample('1min').interpolate(method='time')
    df_weather_temp = df_weather_temp.ffill().bfill()  # Fill any remaining NaNs
    
    # Get power column - 'P' is real power in these datasets
    power_col = 'P' if 'P' in df_whe.columns else None
    if power_col is None:
        power_col = get_power_column(df_whe, 'WHE')
    
    # Align all dataframes to 1-minute resolution
    print("Aligning all data to common time index...")
    
    # Find common time range
    start_time = max(df_whe.index.min(), df_hpe.index.min(), df_weather_temp.index.min())
    end_time = min(df_whe.index.max(), df_hpe.index.max(), df_weather_temp.index.max())
    
    print(f"Common time range: {start_time} to {end_time}")
    
    # Create common index (limit to max_samples for faster processing)
    common_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    if len(common_index) > max_samples:
        common_index = common_index[:max_samples]
        print(f"Limiting to {max_samples} samples ({max_samples/(60*24):.1f} days)")
    
    # Reindex all dataframes
    df_whe_aligned = df_whe.reindex(common_index, method='nearest')
    df_hpe_aligned = df_hpe.reindex(common_index, method='nearest')
    df_weather_aligned = df_weather_temp.reindex(common_index, method='nearest')
    
    # Merge into single DataFrame
    df = pd.DataFrame(index=common_index)
    df['T_out'] = df_weather_aligned[temp_col].values
    
    # Get power values (convert from W to kW if needed)
    if power_col and power_col in df_whe_aligned.columns:
        whe_values = df_whe_aligned[power_col].values
        # Convert W to kW if values seem too large
        if np.nanmean(whe_values) > 100:
            whe_values = whe_values / 1000.0
        df['WHE'] = whe_values
    else:
        df['WHE'] = 0
    
    if power_col and power_col in df_hpe_aligned.columns:
        hpe_values = df_hpe_aligned[power_col].values
        if np.nanmean(hpe_values) > 100:
            hpe_values = hpe_values / 1000.0
        df['HPE'] = hpe_values
    else:
        df['HPE'] = 0
    
    # Create TOU pricing
    df['hour'] = df.index.hour
    df['Price'] = df['hour'].apply(
        lambda h: config.peak_price if 16 <= h < 21 else config.offpeak_price
    )
    
    # Create time features
    df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Handle missing values
    df = df.ffill().bfill()
    df = df.dropna()
    
    # Clean outdoor temperature (sanity checks)
    df['T_out'] = df['T_out'].clip(-40, 50)  # Reasonable outdoor temp range
    
    if len(df) == 0:
        print("Warning: No valid data after processing. Using synthetic data.")
        return generate_synthetic_data(config)
    
    print(f"\nFinal merged dataset: {len(df)} samples ({len(df)/(60*24):.1f} days)")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print(f"T_out range: {df['T_out'].min():.1f}°C to {df['T_out'].max():.1f}°C")
    
    return df


def process_timestamp(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Convert timestamp column to datetime index."""
    # Try common timestamp column names
    ts_cols = ['datetime', 'timestamp', 'time', 'date_time', 'Date', 'Timestamp', 'Date/Time', 'unix_ts']
    ts_col = None
    
    for col in ts_cols:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        # Try first column
        ts_col = df.columns[0]
    
    try:
        if ts_col == 'unix_ts':
            df['datetime'] = pd.to_datetime(df[ts_col], unit='s')
            df = df.set_index('datetime')
        else:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col)
        df = df.sort_index()
    except Exception as e:
        print(f"Warning: Could not parse timestamp for {name}: {e}")
        # Create synthetic timestamps
        df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')
    
    return df


def get_power_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find the power/energy column in a dataframe."""
    # First try exact match for 'P' (real power)
    if 'P' in df.columns:
        return 'P'
    
    for col in df.columns:
        if any(x in col.lower() for x in ['power', 'energy', 'watt', 'kwh', 'wh']):
            return col
    # Return first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    return None


def generate_synthetic_data(config: Config, n_days: int = 60) -> pd.DataFrame:
    """Generate synthetic data as fallback when real data is unavailable."""
    print("Generating synthetic weather and energy data...")
    
    n_samples = n_days * 24 * 60  # 1-minute resolution
    timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='1min')
    
    # Generate outdoor temperature (seasonal + daily variation)
    day_of_year = np.array([(t - timestamps[0]).days for t in timestamps])
    hour_of_day = np.array([t.hour + t.minute/60.0 for t in timestamps])
    
    # Base temperature: varies seasonally
    seasonal_temp = 5 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    T_out = seasonal_temp + daily_variation + np.random.normal(0, 2, n_samples)
    T_out = np.clip(T_out, -15, 35)
    
    # Generate energy consumption (correlated with outdoor temp)
    WHE = np.random.uniform(0.1, 0.5, n_samples)  # Background load
    HPE = np.maximum(0, (20 - T_out) * 0.3) * np.random.uniform(0.8, 1.2, n_samples)
    
    df = pd.DataFrame({
        'T_out': T_out,
        'WHE': WHE,
        'HPE': HPE,
    }, index=timestamps)
    
    # Add time features
    df['hour'] = df.index.hour
    df['Price'] = df['hour'].apply(
        lambda h: config.peak_price if 16 <= h < 21 else config.offpeak_price
    )
    df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    print(f"Generated {len(df)} samples ({n_days} days)")
    return df


# =============================================================================
# 2R2C THERMAL MODEL ENVIRONMENT WITH SAFETY LAYER
# =============================================================================
class SafetyHVACEnv(gym.Env):
    """
    2R2C Physics-Informed HVAC Environment with Safety Constraints.
    
    State: [T_in_obs, T_out, T_mass, Price, time_sin, time_cos]
    Action: Discrete {0: OFF, 1: ON}
    
    Features:
    - 2R2C thermal dynamics (indoor air + building mass)
    - Domain randomization for training robustness
    - Safety layer with 15-minute lockout
    - Gaussian observation noise on T_in
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        is_training: bool = True,
        use_domain_randomization: bool = True,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None
    ):
        super().__init__()
        
        self.data = data
        self.config = config
        self.is_training = is_training
        self.use_domain_randomization = use_domain_randomization and is_training
        
        # Data range
        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx = end_idx if end_idx is not None else len(data)
        self.episode_length = config.episode_length_days * 24 * 60  # in minutes
        
        # State space: [T_in_norm, T_out_norm, T_mass_norm, Price_norm, time_sin, time_cos]
        # All normalized to roughly [-1, 1] or [0, 1] range
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -1.5, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 1.5, 1.5, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 0 = OFF, 1 = ON
        self.action_space = spaces.Discrete(2)
        
        # Initialize state
        self.T_in = config.setpoint
        self.T_mass = config.setpoint
        self.current_step = 0
        self.episode_start_idx = 0
        
        # Safety counters
        self.runtime = config.lockout_time  # Minutes since last ON (start unlocked)
        self.offtime = config.lockout_time  # Minutes since last OFF (start unlocked)
        self.last_action = 0
        
        # Mask event counters
        self.masked_off_count = 0
        self.masked_on_count = 0
        
        # Episode statistics
        self.episode_actions = []
        self.episode_temps = []
        self.episode_costs = []
        self.episode_power = []
        self.on_runtimes = []  # Track duration of each ON period
        
        # Current parameters (will be set in reset)
        self.R_i = config.R_i
        self.R_w = config.R_w
        self.R_o = config.R_o
        self.C_in = config.C_in
        self.C_m = config.C_m
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        # Domain randomization (training only)
        if self.use_domain_randomization:
            variation = self.config.param_variation
            self.R_i = self.config.R_i * np.random.uniform(1 - variation, 1 + variation)
            self.R_w = self.config.R_w * np.random.uniform(1 - variation, 1 + variation)
            self.R_o = self.config.R_o * np.random.uniform(1 - variation, 1 + variation)
            self.C_in = self.config.C_in * np.random.uniform(1 - variation, 1 + variation)
            self.C_m = self.config.C_m * np.random.uniform(1 - variation, 1 + variation)
        else:
            # Use nominal parameters
            self.R_i = self.config.R_i
            self.R_w = self.config.R_w
            self.R_o = self.config.R_o
            self.C_in = self.config.C_in
            self.C_m = self.config.C_m
        
        # Ensure we have valid data range
        data_len = len(self.data)
        self.start_idx = max(0, min(self.start_idx, data_len - 1))
        self.end_idx = max(self.start_idx + 1, min(self.end_idx, data_len))
        
        # Random episode start (training) or sequential (testing)
        if self.is_training:
            available_range = self.end_idx - self.start_idx
            ep_len = min(self.episode_length, available_range)
            max_start = max(self.start_idx, self.end_idx - ep_len)
            if max_start <= self.start_idx:
                self.episode_start_idx = self.start_idx
            else:
                self.episode_start_idx = np.random.randint(self.start_idx, max_start)
            self.episode_length = min(ep_len, self.end_idx - self.episode_start_idx)
        else:
            self.episode_start_idx = self.start_idx
            self.episode_length = self.end_idx - self.start_idx
        
        # Ensure episode_length is at least 1
        self.episode_length = max(1, self.episode_length)
        
        self.current_step = 0
        
        # Initialize temperatures - safely access data
        safe_idx = min(self.episode_start_idx, len(self.data) - 1)
        T_out = self.data.iloc[safe_idx]['T_out']
        self.T_in = self.config.setpoint + np.random.uniform(-1, 1)
        self.T_mass = self.T_in - 0.5  # Mass is slightly cooler initially
        
        # Reset safety counters (unlocked state)
        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time
        self.last_action = 0
        
        # Reset statistics
        self.masked_off_count = 0
        self.masked_on_count = 0
        self.episode_actions = []
        self.episode_temps = []
        self.episode_costs = []
        self.episode_power = []
        self.on_runtimes = []
        self._current_on_duration = 0
        
        observation = self._get_observation()
        info = {'T_in_true': self.T_in, 'T_out': T_out}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep with safety layer enforcement."""
        
        # Get current data
        data_idx = min(self.episode_start_idx + self.current_step, len(self.data) - 1)
        row = self.data.iloc[data_idx]
        T_out = row['T_out']
        price = row['Price']
        time_sin = row['time_sin']
        time_cos = row['time_cos']
        
        # =====================================================================
        # SAFETY LAYER: 15-minute lockout enforcement
        # =====================================================================
        original_action = action
        masked = False
        
        if action == 0 and self.last_action == 1:
            # Trying to turn OFF while ON
            if self.runtime < self.config.lockout_time:
                # Cannot turn OFF yet - force ON
                action = 1
                self.masked_off_count += 1
                masked = True
        elif action == 1 and self.last_action == 0:
            # Trying to turn ON while OFF
            if self.offtime < self.config.lockout_time:
                # Cannot turn ON yet - force OFF
                action = 0
                self.masked_on_count += 1
                masked = True
        
        # Update counters
        if action == 1:
            self.runtime += 1
            self.offtime = 0
            self._current_on_duration += 1
        else:
            self.offtime += 1
            if self.last_action == 1 and self._current_on_duration > 0:
                # Just turned OFF - record the ON duration
                self.on_runtimes.append(self._current_on_duration)
                self._current_on_duration = 0
            self.runtime = 0
        
        self.last_action = action
        
        # =====================================================================
        # 2R2C THERMAL DYNAMICS (Euler integration)
        # =====================================================================
        # Heat input from HVAC
        Q_hvac = action * self.config.Q_hvac_max  # W
        
        # Heat flows (W)
        # Q_im: heat flow from indoor air to mass
        Q_im = (self.T_in - self.T_mass) / self.R_i
        
        # Q_mo: heat flow from mass to outdoor (through wall)
        Q_mo = (self.T_mass - T_out) / (self.R_w + self.R_o)
        
        # Temperature updates
        # Indoor air: gains heat from HVAC, loses to mass
        dT_in = (Q_hvac - Q_im) / self.C_in * self.config.dt
        self.T_in += dT_in
        
        # Building mass: gains from indoor air, loses to outdoor
        dT_mass = (Q_im - Q_mo) / self.C_m * self.config.dt
        self.T_mass += dT_mass
        
        # Clip temperatures to reasonable range
        self.T_in = np.clip(self.T_in, 10.0, 35.0)
        self.T_mass = np.clip(self.T_mass, 10.0, 35.0)
        
        # =====================================================================
        # REWARD CALCULATION (Multi-objective: comfort, cost, and stability)
        # =====================================================================
        dt_hours = 1.0 / 60.0  # 1 minute in hours
        power_kw = action * (self.config.Q_hvac_max / 1000.0)  # Convert W to kW
        
        # Cost component
        cost_t = power_kw * price * dt_hours
        
        # Comfort-based reward
        temp_error = abs(self.T_in - self.config.setpoint)
        discomfort_t = temp_error ** 2  # Keep for statistics
        
        # Determine comfort status
        in_comfort = self.config.comfort_min <= self.T_in <= self.config.comfort_max
        
        if in_comfort:
            # In comfort band: positive reward, better when closer to setpoint
            comfort_reward = 1.0 - (temp_error / 3.0)
        else:
            # Outside comfort band: strong penalty
            if self.T_in < self.config.comfort_min:
                violation = self.config.comfort_min - self.T_in
            else:
                violation = self.T_in - self.config.comfort_max
            comfort_reward = -3.0 * (1.0 + violation)
        
        # Cost penalty (always applies, stronger during peak)
        data_idx = min(self.episode_start_idx + self.current_step, len(self.data) - 1)
        hour = self.data.index[data_idx].hour
        is_peak = 16 <= hour < 21
        
        if is_peak:
            cost_penalty = 0.5 * cost_t  # Stronger peak penalty
        else:
            cost_penalty = 0.1 * cost_t  # Small off-peak penalty
        
        # Action switching penalty (discourages unnecessary cycling)
        switching_penalty = 0.0
        if len(self.episode_actions) > 0 and action != self.episode_actions[-1]:
            switching_penalty = 0.1  # Small penalty for each switch
        
        # Total reward
        reward = comfort_reward - cost_penalty - switching_penalty
        
        # Store episode data
        self.episode_actions.append(action)
        self.episode_temps.append(self.T_in)
        self.episode_costs.append(cost_t)
        self.episode_power.append(power_kw)
        
        # Update step
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # If episode ends and we're still ON, record that duration
        if terminated and self._current_on_duration > 0:
            self.on_runtimes.append(self._current_on_duration)
        
        observation = self._get_observation()
        
        info = {
            'T_in_true': self.T_in,
            'T_out': T_out,
            'T_mass': self.T_mass,
            'action': action,
            'original_action': original_action,
            'masked': masked,
            'power_kw': power_kw,
            'cost': cost_t,
            'discomfort': discomfort_t,
            'price': price,
            'runtime': self.runtime,
            'offtime': self.offtime,
            'masked_off_total': self.masked_off_count,
            'masked_on_total': self.masked_on_count
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get normalized observation with noise on T_in."""
        data_idx = min(self.episode_start_idx + self.current_step, len(self.data) - 1)
        row = self.data.iloc[data_idx]
        
        # Add Gaussian noise to T_in observation
        T_in_obs = self.T_in + np.random.normal(0, self.config.obs_noise_std)
        T_in_obs = np.clip(T_in_obs, 10.0, 35.0)
        
        # Normalize observations for better learning
        # T_in: normalize around setpoint (21°C ± 10°C) -> roughly [-1, 1]
        T_in_norm = (T_in_obs - self.config.setpoint) / 10.0
        
        # T_out: normalize (-20 to 40°C) -> roughly [-1, 1]  
        T_out_norm = (row['T_out'] - 10.0) / 30.0
        
        # T_mass: same as T_in
        T_mass_norm = (self.T_mass - self.config.setpoint) / 10.0
        
        # Price: already in a reasonable range, normalize 0-1
        price_norm = row['Price'] / self.config.peak_price
        
        return np.array([
            T_in_norm,
            T_out_norm,
            T_mass_norm,
            price_norm,
            row['time_sin'],
            row['time_cos']
        ], dtype=np.float32)
    
    def get_statistics(self) -> Dict:
        """Get episode statistics."""
        return {
            'total_cost': sum(self.episode_costs),
            'total_discomfort': sum((t - self.config.setpoint)**2 for t in self.episode_temps) / 60.0,
            'total_energy_kwh': sum(self.episode_power) / 60.0,  # Convert minute-kW to kWh
            'n_cycles': max(0, len(self.on_runtimes) - 1),  # Number of complete ON/OFF cycles
            'masked_off': self.masked_off_count,
            'masked_on': self.masked_on_count,
            'on_runtimes': self.on_runtimes.copy(),
            'temps': self.episode_temps.copy(),
            'actions': self.episode_actions.copy(),
            'power': self.episode_power.copy()
        }


# =============================================================================
# BASELINE THERMOSTAT CONTROLLER
# =============================================================================
class BaselineThermostat:
    """
    Simple thermostat with deadband and 15-minute lockout.
    
    Setpoint: 21°C, Deadband: ±1.5°C
    Lockout: 15 minutes minimum ON/OFF time
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.current_action = 0
        self.runtime = config.lockout_time  # Start unlocked
        self.offtime = config.lockout_time
        
        self.masked_off_count = 0
        self.masked_on_count = 0
        
    def reset(self):
        """Reset the thermostat state."""
        self.current_action = 0
        self.runtime = self.config.lockout_time
        self.offtime = self.config.lockout_time
        self.masked_off_count = 0
        self.masked_on_count = 0
        
    def predict(self, T_in: float) -> int:
        """
        Predict action based on temperature.
        
        Returns:
            Action (0 or 1) after applying lockout constraints
        """
        upper_bound = self.config.setpoint + self.config.deadband  # 22.5
        lower_bound = self.config.setpoint - self.config.deadband  # 19.5
        
        # Determine desired action based on temperature
        if T_in > upper_bound:
            desired_action = 0  # Too hot, want OFF
        elif T_in < lower_bound:
            desired_action = 1  # Too cold, want ON
        else:
            desired_action = self.current_action  # In deadband, keep current
        
        # Apply lockout constraints
        actual_action = desired_action
        
        if desired_action == 0 and self.current_action == 1:
            # Trying to turn OFF
            if self.runtime < self.config.lockout_time:
                actual_action = 1  # Must stay ON
                self.masked_off_count += 1
        elif desired_action == 1 and self.current_action == 0:
            # Trying to turn ON
            if self.offtime < self.config.lockout_time:
                actual_action = 0  # Must stay OFF
                self.masked_on_count += 1
        
        # Update counters
        if actual_action == 1:
            self.runtime += 1
            self.offtime = 0
        else:
            self.offtime += 1
            self.runtime = 0
        
        self.current_action = actual_action
        return actual_action


# =============================================================================
# TRAINING CALLBACK
# =============================================================================
class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self):
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"  Timesteps: {self.num_timesteps}")


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================
def evaluate_controller(
    env: SafetyHVACEnv,
    controller: Any,
    is_ppo: bool = False,
    config: Config = None
) -> Dict:
    """
    Evaluate a controller on the environment.
    
    Args:
        env: The HVAC environment
        controller: Either a PPO model or BaselineThermostat
        is_ppo: Whether the controller is a PPO model
        config: Configuration object
        
    Returns:
        Dictionary with evaluation results
    """
    obs, info = env.reset()
    done = False
    
    # Track hourly power
    hourly_power = {h: [] for h in range(24)}
    
    # For baseline thermostat
    if not is_ppo:
        controller.reset()
    
    while not done:
        if is_ppo:
            action, _ = controller.predict(obs, deterministic=True)
        else:
            # Baseline uses true T_in from environment (not normalized obs)
            # The normalized T_in_obs is: (T_in - setpoint) / 10.0
            # So: T_in = obs[0] * 10.0 + setpoint
            T_in = obs[0] * 10.0 + env.config.setpoint
            action = controller.predict(T_in)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track hourly power
        data_idx = min(env.episode_start_idx + env.current_step - 1, len(env.data) - 1)
        hour = env.data.index[data_idx].hour
        hourly_power[hour].append(info['power_kw'])
    
    # Get statistics
    stats = env.get_statistics()
    
    # Add mask counts from environment
    stats['masked_off'] = env.masked_off_count
    stats['masked_on'] = env.masked_on_count
    
    # Add hourly power averages
    stats['hourly_power'] = {h: np.mean(p) if p else 0 for h, p in hourly_power.items()}
    
    # Peak vs off-peak power
    peak_powers = [stats['hourly_power'][h] for h in range(16, 21)]
    offpeak_powers = [stats['hourly_power'][h] for h in range(24) if h < 16 or h >= 21]
    
    stats['avg_peak_power'] = np.mean(peak_powers) if peak_powers else 0
    stats['avg_offpeak_power'] = np.mean(offpeak_powers) if offpeak_powers else 0
    
    return stats


def run_robustness_test(
    data: pd.DataFrame,
    model: PPO,
    baseline: BaselineThermostat,
    config: Config,
    test_start_idx: int,
    test_end_idx: int
) -> Dict:
    """
    Run robustness test with varying R multipliers.
    
    Returns results for R multipliers: [0.8, 0.9, 1.0, 1.1, 1.2]
    """
    r_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
    results = {'r_multipliers': r_multipliers, 'ppo_costs': [], 'baseline_costs': []}
    
    for r_mult in r_multipliers:
        # Create modified config
        test_config = Config()
        test_config.R_i = config.R_i * r_mult
        test_config.R_w = config.R_w * r_mult
        test_config.R_o = config.R_o * r_mult
        
        # Test environment
        test_env = SafetyHVACEnv(
            data=data,
            config=test_config,
            is_training=False,
            use_domain_randomization=False,
            start_idx=test_start_idx,
            end_idx=test_end_idx
        )
        
        # Evaluate PPO
        ppo_stats = evaluate_controller(test_env, model, is_ppo=True, config=test_config)
        results['ppo_costs'].append(ppo_stats['total_cost'])
        
        # Evaluate baseline
        test_env.reset()
        baseline_stats = evaluate_controller(test_env, baseline, is_ppo=False, config=test_config)
        results['baseline_costs'].append(baseline_stats['total_cost'])
    
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
    """Generate all 7 figures."""
    
    print("\n" + "="*80)
    print("PHASE 4: Generating Figures")
    print("="*80)
    
    # Figure 1: Micro Dynamics (4-hour zoom)
    generate_figure1_micro_dynamics(baseline_stats, ppo_stats, config, output_dir)
    
    # Figure 2: Safety Verification
    generate_figure2_safety_verification(baseline_stats, ppo_stats, config, output_dir)
    
    # Figure 3: Policy Heatmap
    generate_figure3_policy_heatmap(model, config, output_dir)
    
    # Figure 4: Multi-Objective Radar
    generate_figure4_radar(baseline_stats, ppo_stats, config, output_dir)
    
    # Figure 5: Robustness
    generate_figure5_robustness(robustness_results, output_dir)
    
    # Figure 6: Comfort Distribution
    generate_figure6_comfort_distribution(baseline_stats, ppo_stats, config, output_dir)
    
    # Figure 7: Price Response
    generate_figure7_price_response(baseline_stats, ppo_stats, config, output_dir)
    
    print("All figures generated!")


def generate_figure1_micro_dynamics(baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str):
    """Figure 1: 4-hour zoom of temperature and actions."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Get 4-hour window (240 minutes)
    n_samples = min(240, len(baseline_stats['temps']), len(ppo_stats['temps']))
    time_hours = np.arange(n_samples) / 60.0
    
    baseline_temps = baseline_stats['temps'][:n_samples]
    baseline_actions = baseline_stats['actions'][:n_samples]
    ppo_temps = ppo_stats['temps'][:n_samples]
    ppo_actions = ppo_stats['actions'][:n_samples]
    
    # Top plot: Temperatures
    ax1 = axes[0]
    ax1.plot(time_hours, baseline_temps, 'r-', label='Baseline', linewidth=2, alpha=0.8)
    ax1.plot(time_hours, ppo_temps, 'b-', label='PI-DRL', linewidth=2, alpha=0.8)
    ax1.axhline(y=config.setpoint, color='k', linestyle='--', label='Setpoint (21°C)', linewidth=1.5)
    ax1.axhspan(config.comfort_min, config.comfort_max, alpha=0.2, color='green', label='Comfort Band')
    ax1.set_ylabel('Indoor Temperature (°C)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim([18, 26])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Temperature Trajectories (4-Hour Window)', fontweight='bold')
    
    # Bottom plot: Actions
    ax2 = axes[1]
    ax2.step(time_hours, baseline_actions, 'r-', where='post', label='Baseline', linewidth=2, alpha=0.7)
    ax2.step(time_hours, [a + 0.05 for a in ppo_actions], 'b-', where='post', label='PI-DRL', linewidth=2, alpha=0.7)
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


def generate_figure2_safety_verification(baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str):
    """Figure 2: Histograms of ON-runtimes showing safety compliance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline runtimes
    ax1 = axes[0]
    baseline_runtimes = baseline_stats.get('on_runtimes', [])
    if len(baseline_runtimes) > 0:
        bins = np.arange(0, max(baseline_runtimes) + 5, 5)
        ax1.hist(baseline_runtimes, bins=bins, color='red', alpha=0.7, edgecolor='black')
    ax1.axvline(x=config.lockout_time, color='black', linestyle='--', linewidth=2, label=f'Lockout ({config.lockout_time} min)')
    ax1.set_xlabel('ON Runtime (minutes)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Baseline Thermostat', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Count violations
    baseline_violations = sum(1 for r in baseline_runtimes if r < config.lockout_time)
    ax1.text(0.95, 0.95, f'Violations < 15 min: {baseline_violations}',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # PI-DRL runtimes
    ax2 = axes[1]
    ppo_runtimes = ppo_stats.get('on_runtimes', [])
    if len(ppo_runtimes) > 0:
        bins = np.arange(0, max(ppo_runtimes) + 5, 5)
        ax2.hist(ppo_runtimes, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=config.lockout_time, color='black', linestyle='--', linewidth=2, label=f'Lockout ({config.lockout_time} min)')
    ax2.set_xlabel('ON Runtime (minutes)', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('PI-DRL Agent', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Count violations (should be 0 due to safety layer)
    ppo_violations = sum(1 for r in ppo_runtimes if r < config.lockout_time)
    ax2.text(0.95, 0.95, f'Violations < 15 min: {ppo_violations}',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if ppo_violations == 0 else 'wheat', alpha=0.8))
    
    plt.suptitle('Safety Verification: ON-Runtime Distributions', fontweight='bold', fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_2_Safety_Verification.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure3_policy_heatmap(model: PPO, config: Config, output_dir: str):
    """Figure 3: Policy heatmap showing P(ON) vs hour and T_out."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    hours = np.arange(24)
    temps = np.linspace(-10, 30, 20)
    
    prob_on = np.zeros((len(temps), len(hours)))
    
    for i, T_out in enumerate(temps):
        for j, hour in enumerate(hours):
            # Create normalized observation (matching environment normalization)
            T_in = config.setpoint
            T_mass = config.setpoint
            price = config.peak_price if 16 <= hour < 21 else config.offpeak_price
            time_sin = np.sin(2 * np.pi * hour / 24)
            time_cos = np.cos(2 * np.pi * hour / 24)
            
            # Normalize (same as in _get_observation)
            T_in_norm = (T_in - config.setpoint) / 10.0
            T_out_norm = (T_out - 10.0) / 30.0
            T_mass_norm = (T_mass - config.setpoint) / 10.0
            price_norm = price / config.peak_price
            
            obs = np.array([T_in_norm, T_out_norm, T_mass_norm, price_norm, time_sin, time_cos], dtype=np.float32)
            
            # Sample multiple actions to estimate probability
            actions = []
            for _ in range(20):
                action, _ = model.predict(obs, deterministic=False)
                actions.append(action)
            prob_on[i, j] = np.mean(actions)
    
    # Create heatmap
    im = ax.imshow(prob_on, aspect='auto', origin='lower', cmap='RdYlBu_r', 
                   extent=[0, 24, temps[0], temps[-1]], vmin=0, vmax=1)
    
    # Highlight peak hours
    ax.axvspan(16, 21, alpha=0.2, color='red', label='Peak Hours (16:00-21:00)')
    
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Outdoor Temperature (°C)', fontweight='bold')
    ax.set_title('PI-DRL Policy: Probability of Heating ON', fontweight='bold', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(ON)', fontweight='bold', rotation=270, labelpad=15)
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_3_Policy_Heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure4_radar(baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str):
    """Figure 4: Multi-objective radar chart."""
    categories = ['Total Cost', 'Discomfort', 'Peak Load', 'Cycles', 'Energy']
    
    # Get values (normalize so baseline = 1)
    baseline_values = [
        baseline_stats['total_cost'],
        baseline_stats['total_discomfort'],
        baseline_stats.get('avg_peak_power', 1) * 10,  # Scale for visibility
        max(baseline_stats.get('n_cycles', 1), 1),
        baseline_stats['total_energy_kwh']
    ]
    
    ppo_values = [
        ppo_stats['total_cost'],
        ppo_stats['total_discomfort'],
        ppo_stats.get('avg_peak_power', 1) * 10,
        max(ppo_stats.get('n_cycles', 1), 1),
        ppo_stats['total_energy_kwh']
    ]
    
    # Normalize (lower is better, so invert for radar)
    # Radar shows "performance" where higher is better
    baseline_normalized = []
    ppo_normalized = []
    
    for b, p in zip(baseline_values, ppo_values):
        max_val = max(b, p, 0.001)
        # Invert: (1 - relative) so lower original = higher on radar
        baseline_normalized.append(1 - b / (max_val * 1.2))
        ppo_normalized.append(1 - p / (max_val * 1.2))
    
    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
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
    ax.set_title('Multi-Objective Performance\n(Higher = Better)', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_4_Multi_Objective_Radar.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure5_robustness(robustness_results: Dict, output_dir: str):
    """Figure 5: Robustness analysis across R multipliers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    r_mults = robustness_results['r_multipliers']
    ppo_costs = robustness_results['ppo_costs']
    baseline_costs = robustness_results['baseline_costs']
    
    ax.plot(r_mults, baseline_costs, 'r-o', linewidth=2, markersize=10, label='Baseline')
    ax.plot(r_mults, ppo_costs, 'b-s', linewidth=2, markersize=10, label='PI-DRL')
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Nominal R')
    
    ax.set_xlabel('R Multiplier', fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontweight='bold')
    ax.set_title('Robustness Analysis: Cost vs. Thermal Resistance Variation', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_5_Robustness.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure6_comfort_distribution(baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str):
    """Figure 6: Violin/box plots of indoor temperature distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_temps = baseline_stats['temps']
    ppo_temps = ppo_stats['temps']
    
    # Create data for violin plot
    data = [baseline_temps, ppo_temps]
    positions = [1, 2]
    
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ['red', 'blue']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    
    # Add comfort band
    ax.axhspan(config.comfort_min, config.comfort_max, alpha=0.2, color='green', label='Comfort Band')
    ax.axhline(y=config.setpoint, color='black', linestyle='--', linewidth=1.5, label=f'Setpoint ({config.setpoint}°C)')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', 'PI-DRL'], fontweight='bold')
    ax.set_ylabel('Indoor Temperature (°C)', fontweight='bold')
    ax.set_title('Temperature Distribution Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f'Baseline: μ={np.mean(baseline_temps):.1f}°C, σ={np.std(baseline_temps):.2f}°C\n'
    stats_text += f'PI-DRL: μ={np.mean(ppo_temps):.1f}°C, σ={np.std(ppo_temps):.2f}°C'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_6_Comfort_Distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def generate_figure7_price_response(baseline_stats: Dict, ppo_stats: Dict, config: Config, output_dir: str):
    """Figure 7: Average power vs hour-of-day with peak highlighting."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    hours = list(range(24))
    
    baseline_hourly = [baseline_stats['hourly_power'].get(h, 0) for h in hours]
    ppo_hourly = [ppo_stats['hourly_power'].get(h, 0) for h in hours]
    
    width = 0.35
    x = np.array(hours)
    
    bars1 = ax.bar(x - width/2, baseline_hourly, width, label='Baseline', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, ppo_hourly, width, label='PI-DRL', color='blue', alpha=0.7)
    
    # Highlight peak hours
    ax.axvspan(15.5, 20.5, alpha=0.2, color='orange', label='Peak Hours (16:00-21:00)')
    
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Average Power (kW)', fontweight='bold')
    ax.set_title('Demand Response: Power Consumption by Hour', fontweight='bold', fontsize=14)
    ax.set_xticks(hours)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Calculate peak reduction
    peak_hours = range(16, 21)
    baseline_peak = np.mean([baseline_stats['hourly_power'].get(h, 0) for h in peak_hours])
    ppo_peak = np.mean([ppo_stats['hourly_power'].get(h, 0) for h in peak_hours])
    
    if baseline_peak > 0:
        reduction = (baseline_peak - ppo_peak) / baseline_peak * 100
        ax.text(0.98, 0.98, f'Peak Power Reduction: {reduction:.1f}%',
                transform=ax.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
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
    """Generate all 4 tables."""
    
    print("\n" + "="*80)
    print("PHASE 5: Generating Tables")
    print("="*80)
    
    # Table 1: System Parameters
    table1 = pd.DataFrame({
        'Parameter': [
            'R_i (K/W)', 'R_w (K/W)', 'R_o (K/W)',
            'C_in (J/K)', 'C_m (J/K)',
            'Lockout Time (min)', 'Setpoint (°C)', 'Deadband (°C)',
            'Q_HVAC (kW)', 'dt (s)',
            'PPO Learning Rate', 'PPO Gamma', 'PPO n_steps', 'PPO batch_size',
            'λ_cost', 'λ_discomfort', 'λ_penalty'
        ],
        'Value': [
            f'{config.R_i:.4f}', f'{config.R_w:.4f}', f'{config.R_o:.4f}',
            f'{config.C_in:.0f}', f'{config.C_m:.0f}',
            f'{config.lockout_time}', f'{config.setpoint}', f'{config.deadband}',
            f'{config.Q_hvac_max/1000:.1f}', f'{config.dt:.0f}',
            f'{config.learning_rate}', f'{config.gamma}', f'{config.n_steps}', f'{config.batch_size}',
            f'{config.lambda_cost}', f'{config.lambda_discomfort}', f'{config.lambda_penalty}'
        ]
    })
    filepath = os.path.join(output_dir, 'Table_1_System_Parameters.csv')
    table1.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    
    # Table 2: Performance Summary
    table2 = pd.DataFrame({
        'Metric': ['Total Cost ($)', 'Total Discomfort (°C²·h)', 'Total Cycles', 'Total Energy (kWh)'],
        'Baseline': [
            f"{baseline_stats['total_cost']:.2f}",
            f"{baseline_stats['total_discomfort']:.2f}",
            f"{baseline_stats['n_cycles']}",
            f"{baseline_stats['total_energy_kwh']:.2f}"
        ],
        'PI-DRL': [
            f"{ppo_stats['total_cost']:.2f}",
            f"{ppo_stats['total_discomfort']:.2f}",
            f"{ppo_stats['n_cycles']}",
            f"{ppo_stats['total_energy_kwh']:.2f}"
        ],
        'Improvement (%)': [
            f"{(1 - ppo_stats['total_cost'] / max(baseline_stats['total_cost'], 0.001)) * 100:.1f}",
            f"{(1 - ppo_stats['total_discomfort'] / max(baseline_stats['total_discomfort'], 0.001)) * 100:.1f}",
            f"{(1 - ppo_stats['n_cycles'] / max(baseline_stats['n_cycles'], 1)) * 100:.1f}",
            f"{(1 - ppo_stats['total_energy_kwh'] / max(baseline_stats['total_energy_kwh'], 0.001)) * 100:.1f}"
        ]
    })
    filepath = os.path.join(output_dir, 'Table_2_Performance_Summary.csv')
    table2.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    
    # Table 3: Grid Impact
    baseline_peak = baseline_stats.get('avg_peak_power', 0)
    baseline_offpeak = baseline_stats.get('avg_offpeak_power', 0)
    ppo_peak = ppo_stats.get('avg_peak_power', 0)
    ppo_offpeak = ppo_stats.get('avg_offpeak_power', 0)
    
    peak_reduction = (baseline_peak - ppo_peak) / max(baseline_peak, 0.001) * 100
    
    table3 = pd.DataFrame({
        'Metric': ['Avg Peak Power (kW)', 'Avg Off-Peak Power (kW)', 'Peak Reduction (%)'],
        'Baseline': [f"{baseline_peak:.3f}", f"{baseline_offpeak:.3f}", '-'],
        'PI-DRL': [f"{ppo_peak:.3f}", f"{ppo_offpeak:.3f}", f"{peak_reduction:.1f}"]
    })
    filepath = os.path.join(output_dir, 'Table_3_Grid_Impact.csv')
    table3.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    
    # Table 4: Safety Shield Activity
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


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    """Run the complete PI-DRL HVAC pipeline."""
    
    print("="*80)
    print("ROBUST, SAFETY-CRITICAL 2R2C PI-DRL CONTROLLER")
    print("FOR RESIDENTIAL HVAC")
    print("="*80)
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    
    # Initialize configuration
    config = Config()
    
    # =========================================================================
    # PHASE 1: Data Download and Processing
    # =========================================================================
    data = download_and_process_data(config)
    
    # Split data chronologically: 80% train, 20% test
    split_idx = int(len(data) * config.train_split)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} samples ({len(train_data)/(60*24):.1f} days)")
    print(f"  Testing: {len(test_data)} samples ({len(test_data)/(60*24):.1f} days)")
    
    # =========================================================================
    # PHASE 2: Environment Setup
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Environment Setup")
    print("="*80)
    
    # Training environment (with domain randomization)
    train_env = SafetyHVACEnv(
        data=data,
        config=config,
        is_training=True,
        use_domain_randomization=True,
        start_idx=0,
        end_idx=split_idx
    )
    
    # Test environment (nominal parameters)
    test_env = SafetyHVACEnv(
        data=data,
        config=config,
        is_training=False,
        use_domain_randomization=False,
        start_idx=split_idx,
        end_idx=len(data)
    )
    
    print("Environments created successfully!")
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")
    
    # =========================================================================
    # PHASE 3: Training
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Training PPO Agent")
    print("="*80)
    
    # Wrap environment for stable-baselines3
    def make_env():
        return SafetyHVACEnv(
            data=data,
            config=config,
            is_training=True,
            use_domain_randomization=True,
            start_idx=0,
            end_idx=split_idx
        )
    
    vec_env = DummyVecEnv([make_env])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        verbose=1,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    )
    
    print(f"\nTraining for {config.total_timesteps:,} timesteps...")
    print("This may take several minutes...")
    
    # Training callback to track mask events
    class MaskTrackingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.masked_off_total = 0
            self.masked_on_total = 0
            self.total_steps = 0
            
        def _on_step(self):
            self.total_steps += 1
            # Get mask info from environment
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                if hasattr(env, 'masked_off_count'):
                    self.masked_off_total = env.masked_off_count
                    self.masked_on_total = env.masked_on_count
            return True
    
    mask_callback = MaskTrackingCallback()
    
    # Train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=mask_callback,
        progress_bar=True
    )
    
    # Save model
    model_path = os.path.join(output_dir, "ppo_hvac_model")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Training mask statistics
    train_mask_stats = {
        'total_steps': mask_callback.total_steps,
        'masked_off': mask_callback.masked_off_total,
        'masked_on': mask_callback.masked_on_total
    }
    
    # =========================================================================
    # PHASE 4: Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Evaluation")
    print("="*80)
    
    # Initialize baseline controller
    baseline = BaselineThermostat(config)
    
    # Evaluate on test set
    print("\nEvaluating Baseline Thermostat...")
    test_env.reset()
    baseline_stats = evaluate_controller(test_env, baseline, is_ppo=False, config=config)
    
    print("Evaluating PI-DRL Agent...")
    test_env.reset()
    ppo_stats = evaluate_controller(test_env, model, is_ppo=True, config=config)
    
    # Test mask statistics (from PI-DRL evaluation)
    test_mask_stats = {
        'total_steps': len(ppo_stats.get('actions', [])),
        'masked_off': ppo_stats.get('masked_off', 0),
        'masked_on': ppo_stats.get('masked_on', 0)
    }
    
    print(f"\nPI-DRL Safety Shield: {test_mask_stats['masked_off']} masked OFF, {test_mask_stats['masked_on']} masked ON")
    
    # Print comparison
    print("\n" + "-"*50)
    print("PERFORMANCE COMPARISON")
    print("-"*50)
    print(f"{'Metric':<25} {'Baseline':>12} {'PI-DRL':>12} {'Improve':>10}")
    print("-"*50)
    
    metrics = [
        ('Total Cost ($)', 'total_cost'),
        ('Total Discomfort (°C²·h)', 'total_discomfort'),
        ('Total Energy (kWh)', 'total_energy_kwh'),
        ('Cycles', 'n_cycles'),
        ('Avg Peak Power (kW)', 'avg_peak_power'),
    ]
    
    for label, key in metrics:
        b = baseline_stats.get(key, 0)
        p = ppo_stats.get(key, 0)
        if b > 0:
            improve = (1 - p / b) * 100
            print(f"{label:<25} {b:>12.2f} {p:>12.2f} {improve:>9.1f}%")
        else:
            print(f"{label:<25} {b:>12.2f} {p:>12.2f} {'N/A':>10}")
    
    print("-"*50)
    
    # Robustness test
    print("\nRunning robustness analysis...")
    robustness_results = run_robustness_test(
        data=data,
        model=model,
        baseline=baseline,
        config=config,
        test_start_idx=split_idx,
        test_end_idx=min(split_idx + 7*24*60, len(data))  # 7 days for robustness test
    )
    
    # =========================================================================
    # PHASE 5: Generate Outputs
    # =========================================================================
    
    # Generate all figures
    generate_all_figures(
        baseline_stats=baseline_stats,
        ppo_stats=ppo_stats,
        robustness_results=robustness_results,
        model=model,
        data=data,
        config=config,
        output_dir=output_dir
    )
    
    # Generate all tables
    generate_all_tables(
        baseline_stats=baseline_stats,
        ppo_stats=ppo_stats,
        robustness_results=robustness_results,
        config=config,
        train_mask_stats=train_mask_stats,
        test_mask_stats=test_mask_stats,
        output_dir=output_dir
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    print(f"\nGenerated outputs in '{output_dir}/':")
    print("\nFigures (7 PNG files):")
    for i in range(1, 8):
        fig_names = [
            'Micro_Dynamics', 'Safety_Verification', 'Policy_Heatmap',
            'Multi_Objective_Radar', 'Robustness', 'Comfort_Distribution', 'Price_Response'
        ]
        print(f"  - Figure_{i}_{fig_names[i-1]}.png")
    
    print("\nTables (4 CSV files):")
    for i, name in enumerate(['System_Parameters', 'Performance_Summary', 
                              'Grid_Impact', 'Safety_Shield_Activity'], 1):
        print(f"  - Table_{i}_{name}.csv")
    
    print("\nModel:")
    print(f"  - ppo_hvac_model.zip")
    
    # Verify policy is not collapsed
    print("\n" + "-"*50)
    print("POLICY VERIFICATION")
    print("-"*50)
    
    action_ratio = np.mean(ppo_stats['actions'])
    temp_in_band = sum(1 for t in ppo_stats['temps'] 
                       if config.comfort_min <= t <= config.comfort_max) / len(ppo_stats['temps']) * 100
    
    print(f"Action ON ratio: {action_ratio:.2%}")
    print(f"Time in comfort band: {temp_in_band:.1f}%")
    
    if action_ratio < 0.05:
        print("⚠️  Warning: Policy may be collapsed to 'always OFF'")
    elif action_ratio > 0.95:
        print("⚠️  Warning: Policy may be collapsed to 'always ON'")
    else:
        print("✓ Policy appears to be properly learned (not collapsed)")
    
    if temp_in_band < 80:
        print("⚠️  Warning: Temperature frequently outside comfort band")
    else:
        print("✓ Temperature well-maintained within comfort band")
    
    print("\n" + "="*80)
    print("SUCCESS! All outputs generated.")
    print("="*80)


if __name__ == "__main__":
    main()
