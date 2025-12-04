"""
Robust, Safety-Critical 2R2C PI-DRL Controller for Residential HVAC
Complete execution-ready script that runs full pipeline end-to-end.
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# ============================================================================
# DATA DOWNLOAD AND PROCESSING
# ============================================================================

def download_and_process_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download dataset from GitHub and process into training/test sets.
    
    Returns:
        train_df: Training DataFrame (80% of data)
        test_df: Test DataFrame (20% of data)
    """
    print("="*80)
    print("STEP 1: Downloading and Processing Data")
    print("="*80)
    
    url = "https://github.com/Fateme9977/P3/raw/main/data/dataverse_files.zip"
    print(f"Downloading from: {url}")
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    print("Unzipping archive...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Found {len(file_list)} files in archive")
        
        # Find required files
        whe_file = None
        hpe_file = None
        weather_file = None
        
        for fname in file_list:
            basename = os.path.basename(fname)
            if basename.startswith("Electricity_WHE"):
                whe_file = fname
            elif basename.startswith("Electricity_HPE"):
                hpe_file = fname
            elif basename.startswith("Climate_HourlyWeathe"):
                weather_file = fname
        
        if not whe_file or not hpe_file or not weather_file:
            raise ValueError(f"Missing required files. Found: WHE={whe_file}, HPE={hpe_file}, Weather={weather_file}")
        
        print(f"Loading: {whe_file}, {hpe_file}, {weather_file}")
        
        # Read files
        whe_df = pd.read_csv(zip_ref.open(whe_file))
        hpe_df = pd.read_csv(zip_ref.open(hpe_file))
        weather_df = pd.read_csv(zip_ref.open(weather_file))
    
    print("Processing timestamps...")
    # Try to identify timestamp column and process each dataframe
    processed_dfs = {}
    for df, name in [(whe_df, "WHE"), (hpe_df, "HPE"), (weather_df, "Weather")]:
        # Look for common timestamp column names
        ts_cols = [c for c in df.columns if any(x in c.lower() for x in ['time', 'date', 'timestamp', 'datetime'])]
        if ts_cols:
            ts_col = ts_cols[0]
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            df = df.set_index(ts_col)
            print(f"  {name}: Using '{ts_col}' as timestamp")
        else:
            # Create synthetic timestamps if none found
            print(f"  {name}: No timestamp column found, creating synthetic timestamps")
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='1H')
        processed_dfs[name] = df
    
    whe_df = processed_dfs['WHE']
    hpe_df = processed_dfs['HPE']
    weather_df = processed_dfs['Weather']
    
    # Extract temperature from weather data (assume first numeric column or named column)
    temp_cols = [c for c in weather_df.columns if any(x in c.lower() for x in ['temp', 'temperature', 't_out', 'outdoor'])]
    if temp_cols:
        T_out_col = temp_cols[0]
    else:
        # Use first numeric column
        T_out_col = weather_df.select_dtypes(include=[np.number]).columns[0]
    
    print(f"Using '{T_out_col}' as outdoor temperature")
    weather_df['T_out'] = pd.to_numeric(weather_df[T_out_col], errors='coerce')
    
    # Resample weather to 1-minute resolution using interpolation
    print("Resampling weather data to 1-minute resolution...")
    if not isinstance(weather_df.index, pd.DatetimeIndex):
        print("  Warning: Weather data index is not datetime, creating synthetic index")
        weather_df.index = pd.date_range(start='2020-01-01', periods=len(weather_df), freq='1H')
    
    weather_1min = weather_df['T_out'].resample('1T').interpolate(method='time')
    
    # Align all dataframes to 1-minute resolution
    # Use the longest time range
    start_time = min(whe_df.index.min(), hpe_df.index.min(), weather_1min.index.min())
    end_time = max(whe_df.index.max(), hpe_df.index.max(), weather_1min.index.max())
    time_index = pd.date_range(start=start_time, end=end_time, freq='1T')
    
    # Create merged dataframe
    print("Merging dataframes...")
    merged_df = pd.DataFrame(index=time_index)
    merged_df['T_out'] = weather_1min.reindex(time_index, method='nearest')
    
    # Forward fill missing values (using fillna with method parameter for compatibility)
    try:
        merged_df['T_out'] = merged_df['T_out'].fillna(method='ffill').fillna(method='bfill')
    except TypeError:
        # Newer pandas versions use different syntax
        merged_df['T_out'] = merged_df['T_out'].ffill().bfill()
    
    # Add time features
    merged_df['hour'] = merged_df.index.hour + merged_df.index.minute / 60.0
    merged_df['time_sin'] = np.sin(2 * np.pi * merged_df['hour'] / 24.0)
    merged_df['time_cos'] = np.cos(2 * np.pi * merged_df['hour'] / 24.0)
    
    # Create TOU price
    merged_df['Price'] = 0.10  # Off-peak default
    peak_mask = (merged_df.index.hour >= 16) & (merged_df.index.hour < 21)
    merged_df.loc[peak_mask, 'Price'] = 0.30  # Peak hours 16:00-21:00
    
    # Split chronologically: 80% train, 20% test
    split_idx = int(len(merged_df) * 0.8)
    train_df = merged_df.iloc[:split_idx].copy()
    test_df = merged_df.iloc[split_idx:].copy()
    
    print(f"Training data: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Test data: {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
    print("="*80)
    
    return train_df, test_df


# ============================================================================
# 2R2C THERMAL MODEL ENVIRONMENT
# ============================================================================

class SafetyHVACEnv(gym.Env):
    """
    2R2C Thermal Model Environment with Safety Constraints.
    
    State: [T_in_obs, T_out, T_mass, Price, time_sin, time_cos]
    Action: Discrete {0: OFF, 1: ON}
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        nominal_params: Dict[str, float] = None,
        use_domain_randomization: bool = False,
        episode_length_days: int = 7,
        seed: Optional[int] = None
    ):
        """
        Initialize environment.
        
        Args:
            data_df: DataFrame with T_out, Price, time_sin, time_cos columns
            nominal_params: Nominal 2R2C parameters
            use_domain_randomization: Whether to randomize parameters each episode
            episode_length_days: Episode length in days
            seed: Random seed
        """
        super().__init__()
        
        self.data_df = data_df.reset_index(drop=True)
        self.use_domain_randomization = use_domain_randomization
        self.episode_length_days = episode_length_days
        self.episode_length_steps = episode_length_days * 24 * 60  # minutes
        
        # Nominal 2R2C parameters (K/kW, kWh/K)
        if nominal_params is None:
            self.nominal_params = {
                'R_i': 0.05,   # Indoor resistance (K/kW)
                'R_w': 0.02,   # Wall resistance (K/kW)
                'R_o': 0.03,   # Outdoor resistance (K/kW)
                'C_in': 0.3,   # Indoor capacitance (kWh/K)
                'C_m': 0.5     # Mass capacitance (kWh/K)
            }
        else:
            self.nominal_params = nominal_params
        
        # Current parameters (will be randomized if use_domain_randomization=True)
        self.params = self.nominal_params.copy()
        
        # HVAC parameters
        self.Q_hvac_nominal = 4.0  # kW when ON
        self.dt = 60.0 / 3600.0  # 60 seconds in hours
        
        # State space: [T_in_obs, T_out, T_mass, Price, time_sin, time_cos]
        self.observation_space = spaces.Box(
            low=np.array([15.0, -10.0, 15.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([30.0, 40.0, 30.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: Discrete {0: OFF, 1: ON}
        self.action_space = spaces.Discrete(2)
        
        # State variables
        self.current_step = 0
        self.start_idx = 0
        self.T_in_true = 21.0  # True indoor temperature
        self.T_mass = 21.0     # Mass temperature
        self.current_action = 0
        
        # Safety constraints (15-minute lockout)
        self.min_runtime = 15  # minutes
        self.min_offtime = 15  # minutes
        self.runtime_counter = 0  # minutes since last ON
        self.offtime_counter = 0  # minutes since last OFF
        
        # Statistics tracking
        self.masked_off_count = 0
        self.masked_on_count = 0
        self.total_steps = 0
        
        # Reward weights
        self.lambda_cost = 1.0
        self.lambda_discomfort = 50.0
        self.lambda_penalty = 10.0
        
        # Episode statistics
        self.episode_data = {
            'T_in': [],
            'T_out': [],
            'T_mass': [],
            'actions': [],
            'actions_raw': [],  # Before safety masking
            'power': [],
            'cost': [],
            'discomfort': [],
            'masked_off': [],
            'masked_on': []
        }
        
        # Random seed
        self.np_random = np.random.RandomState(seed)
        self.seed = seed
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Domain randomization: sample parameters
        if self.use_domain_randomization:
            for key in self.params:
                multiplier = self.np_random.uniform(0.85, 1.15)
                self.params[key] = self.nominal_params[key] * multiplier
        
        # Random start point (7-day window)
        max_start = max(0, len(self.data_df) - self.episode_length_steps)
        self.start_idx = self.np_random.randint(0, max_start + 1)
        self.current_step = 0
        
        # Initial temperatures
        self.T_in_true = 21.0 + self.np_random.uniform(-1.0, 1.0)
        self.T_mass = self.T_in_true + self.np_random.uniform(-0.5, 0.5)
        
        # Reset action and safety counters
        self.current_action = 0
        self.runtime_counter = 0
        self.offtime_counter = 15  # Start with offtime satisfied
        
        # Reset statistics
        self.masked_off_count = 0
        self.masked_on_count = 0
        self.total_steps = 0
        self.episode_data = {k: [] for k in self.episode_data.keys()}
        
        observation = self._get_observation()
        info = {'T_in_true': self.T_in_true, 'T_mass': self.T_mass}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Get current data point
        data_idx = self.start_idx + self.current_step
        if data_idx >= len(self.data_df):
            # Episode truncated
            observation = self._get_observation()
            return observation, 0.0, False, True, {}
        
        row = self.data_df.iloc[data_idx]
        T_out = float(row['T_out'])
        Price = float(row['Price'])
        time_sin = float(row['time_sin'])
        time_cos = float(row['time_cos'])
        
        # Safety layer: enforce 15-minute lockout
        action_raw = action
        masked_off = False
        masked_on = False
        
        if action == 0:  # Policy wants OFF
            if self.runtime_counter < self.min_runtime:
                action = 1  # Force ON
                masked_off = True
                self.masked_off_count += 1
        elif action == 1:  # Policy wants ON
            if self.offtime_counter < self.min_offtime:
                action = 0  # Force OFF
                masked_on = True
                self.masked_on_count += 1
        
        # Update runtime/offtime counters
        if action == 1:
            self.runtime_counter += 1
            self.offtime_counter = 0
        else:
            self.runtime_counter = 0
            self.offtime_counter += 1
        
        self.current_action = action
        
        # HVAC heat input
        if action == 1:
            Q_hvac = self.Q_hvac_nominal  # kW
        else:
            Q_hvac = 0.0
        
        # 2R2C Thermal Dynamics (Euler integration)
        # Simplified 2R2C model:
        # T_in: indoor air temperature
        # T_mass: thermal mass temperature
        # R_i: resistance between indoor and mass
        # R_w: resistance between mass and outdoor
        # R_o: resistance between indoor and outdoor (parallel path)
        # C_in: indoor air capacitance
        # C_m: mass capacitance
        
        R_i, R_w, R_o, C_in, C_m = (
            self.params['R_i'], self.params['R_w'], self.params['R_o'],
            self.params['C_in'], self.params['C_m']
        )
        
        # Heat flows
        Q_in_mass = (self.T_mass - self.T_in_true) / R_i  # kW
        Q_mass_out = (T_out - self.T_mass) / R_w  # kW
        Q_in_out = (T_out - self.T_in_true) / R_o  # kW (parallel path)
        
        # Temperature updates
        dT_in = (Q_hvac + Q_in_mass + Q_in_out) / C_in * self.dt
        dT_mass = (Q_mass_out - Q_in_mass) / C_m * self.dt
        
        self.T_in_true += dT_in
        self.T_mass += dT_mass
        
        # Add observation noise to T_in
        T_in_obs = self.T_in_true + self.np_random.normal(0, 0.1)
        
        # Calculate reward components
        dt_hours = self.dt
        power_kw = Q_hvac
        cost_t = power_kw * Price * dt_hours
        
        # Discomfort (squared deviation from setpoint 21°C)
        discomfort_t = (self.T_in_true - 21.0) ** 2
        
        # Comfort band penalty [19.5, 24.0]°C
        if self.T_in_true < 19.5 or self.T_in_true > 24.0:
            penalty = 1.0
        else:
            penalty = 0.0
        
        # Total reward (negative because we minimize cost/discomfort)
        reward = -(
            self.lambda_cost * cost_t +
            self.lambda_discomfort * discomfort_t +
            self.lambda_penalty * penalty
        )
        
        # Store episode data
        self.episode_data['T_in'].append(self.T_in_true)
        self.episode_data['T_out'].append(T_out)
        self.episode_data['T_mass'].append(self.T_mass)
        self.episode_data['actions'].append(action)
        self.episode_data['actions_raw'].append(action_raw)
        self.episode_data['power'].append(power_kw)
        self.episode_data['cost'].append(cost_t)
        self.episode_data['discomfort'].append(discomfort_t)
        self.episode_data['masked_off'].append(1 if masked_off else 0)
        self.episode_data['masked_on'].append(1 if masked_on else 0)
        
        # Update step
        self.current_step += 1
        self.total_steps += 1
        
        # Check termination
        terminated = self.current_step >= self.episode_length_steps
        truncated = False
        
        observation = self._get_observation()
        info = {
            'T_in_true': self.T_in_true,
            'T_mass': self.T_mass,
            'action': action,
            'action_raw': action_raw,
            'cost': cost_t,
            'discomfort': discomfort_t,
            'masked_off': masked_off,
            'masked_on': masked_on
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        data_idx = self.start_idx + self.current_step
        if data_idx >= len(self.data_df):
            data_idx = len(self.data_df) - 1
        
        row = self.data_df.iloc[data_idx]
        T_out = float(row['T_out'])
        Price = float(row['Price'])
        time_sin = float(row['time_sin'])
        time_cos = float(row['time_cos'])
        
        # Add noise to T_in for observation
        T_in_obs = self.T_in_true + self.np_random.normal(0, 0.1)
        
        return np.array([
            T_in_obs,
            T_out,
            self.T_mass,
            Price,
            time_sin,
            time_cos
        ], dtype=np.float32)
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get episode statistics."""
        if len(self.episode_data['actions']) == 0:
            return {}
        
        actions = np.array(self.episode_data['actions'])
        cycles = np.sum(np.diff(actions) != 0)
        
        return {
            'total_cost': np.sum(self.episode_data['cost']),
            'total_discomfort': np.sum(self.episode_data['discomfort']),
            'total_energy': np.sum(self.episode_data['power']) * self.dt,
            'total_cycles': cycles,
            'masked_off_count': self.masked_off_count,
            'masked_on_count': self.masked_on_count,
            'T_in_mean': np.mean(self.episode_data['T_in']),
            'T_in_std': np.std(self.episode_data['T_in']),
            'episode_data': self.episode_data.copy()
        }


# ============================================================================
# BASELINE THERMOSTAT CONTROLLER
# ============================================================================

class BaselineThermostat:
    """Baseline thermostat with setpoint and deadband."""
    
    def __init__(self, setpoint: float = 21.0, deadband: float = 1.5):
        self.setpoint = setpoint
        self.deadband = deadband
        self.current_action = 0
        self.runtime_counter = 0
        self.offtime_counter = 15  # Start satisfied
        self.min_runtime = 15
        self.min_offtime = 15
    
    def predict(self, observation: np.ndarray) -> int:
        """Predict action based on observation."""
        T_in_obs = observation[0]  # Already in °C
        
        # Thermostat logic
        if T_in_obs > self.setpoint + self.deadband:  # > 22.5°C
            desired_action = 0  # OFF
        elif T_in_obs < self.setpoint - self.deadband:  # < 19.5°C
            desired_action = 1  # ON
        else:
            desired_action = self.current_action  # Keep previous
        
        # Apply 15-minute lockout
        if desired_action == 0:  # Want OFF
            if self.runtime_counter < self.min_runtime:
                desired_action = 1  # Force ON
        elif desired_action == 1:  # Want ON
            if self.offtime_counter < self.min_offtime:
                desired_action = 0  # Force OFF
        
        # Update counters
        if desired_action == 1:
            self.runtime_counter += 1
            self.offtime_counter = 0
        else:
            self.runtime_counter = 0
            self.offtime_counter += 1
        
        self.current_action = desired_action
        return desired_action
    
    def reset(self):
        """Reset controller state."""
        self.current_action = 0
        self.runtime_counter = 0
        self.offtime_counter = 15


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_pidrl_agent(env: SafetyHVACEnv, total_timesteps: int = 200000) -> PPO:
    """Train PPO agent on the environment."""
    print("\n" + "="*80)
    print("STEP 2: Training PI-DRL Agent (PPO)")
    print("="*80)
    
    # Create vectorized environment
    def make_env():
        return env
    
    vec_env = DummyVecEnv([make_env])
    
    # PPO hyperparameters
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    print("Training complete!")
    print("="*80)
    
    return model


def evaluate_controller(env: SafetyHVACEnv, controller: Any, n_episodes: int = 5) -> Dict[str, Any]:
    """Evaluate a controller on the environment."""
    all_results = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        if hasattr(controller, 'reset'):
            controller.reset()
        done = False
        
        while not done:
            if hasattr(controller, 'predict'):
                action = controller.predict(obs)
            else:
                action, _ = controller.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        stats = env.get_episode_statistics()
        all_results.append(stats)
    
    # Aggregate results
    aggregated = {
        'total_cost': np.mean([r['total_cost'] for r in all_results]),
        'total_discomfort': np.mean([r['total_discomfort'] for r in all_results]),
        'total_energy': np.mean([r['total_energy'] for r in all_results]),
        'total_cycles': np.mean([r['total_cycles'] for r in all_results]),
        'masked_off_count': np.sum([r['masked_off_count'] for r in all_results]),
        'masked_on_count': np.sum([r['masked_on_count'] for r in all_results]),
        'T_in_mean': np.mean([r['T_in_mean'] for r in all_results]),
        'T_in_std': np.std([r['T_in_mean'] for r in all_results]),
        'all_episodes': all_results
    }
    
    return aggregated


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_figure_1_micro_dynamics(
    baseline_data: Dict[str, Any],
    pidrl_data: Dict[str, Any],
    output_dir: str
):
    """Figure 1: 4-hour zoom of micro dynamics."""
    print("Generating Figure_1_Micro_Dynamics.png...")
    
    # Get first episode data
    baseline_ep = baseline_data['all_episodes'][0]
    pidrl_ep = pidrl_data['all_episodes'][0]
    
    # Extract 4-hour window (240 minutes)
    window = 240
    baseline_ep_data = baseline_ep['episode_data']
    pidrl_ep_data = pidrl_ep['episode_data']
    
    # Use shorter length if needed
    min_len = min(len(baseline_ep_data['T_in']), len(pidrl_ep_data['T_in']), window)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    time_min = np.arange(min_len)
    
    # Top plot: Temperature
    ax1 = axes[0]
    ax1.plot(time_min, baseline_ep_data['T_in'][:min_len], 'b-', label='Baseline', linewidth=1.5, alpha=0.7)
    ax1.plot(time_min, pidrl_ep_data['T_in'][:min_len], 'r-', label='PI-DRL', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=21.0, color='g', linestyle='--', label='Setpoint', linewidth=1.5)
    ax1.axhspan(19.5, 24.0, alpha=0.2, color='green', label='Comfort Band')
    ax1.set_ylabel('Indoor Temperature (°C)', fontsize=11)
    ax1.set_title('Micro Dynamics: 4-Hour Zoom', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Actions
    ax2 = axes[1]
    ax2.step(time_min, baseline_ep_data['actions'][:min_len], 'b-', where='post', label='Baseline', linewidth=1.5, alpha=0.7)
    ax2.step(time_min, pidrl_ep_data['actions'][:min_len], 'r-', where='post', label='PI-DRL', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Action (0=OFF, 1=ON)', fontsize=11)
    ax2.set_xlabel('Time (minutes)', fontsize=11)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_1_Micro_Dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved Figure_1_Micro_Dynamics.png")


def generate_figure_2_safety_verification(
    baseline_data: Dict[str, Any],
    pidrl_data: Dict[str, Any],
    output_dir: str
):
    """Figure 2: Safety verification - ON-runtime histograms."""
    print("Generating Figure_2_Safety_Verification.png...")
    
    def extract_runtimes(episode_data_list):
        """Extract ON-runtime durations from episode data."""
        all_runtimes = []
        for ep_data in episode_data_list:
            actions = np.array(ep_data['episode_data']['actions'])
            runtimes = []
            current_runtime = 0
            for a in actions:
                if a == 1:
                    current_runtime += 1
                else:
                    if current_runtime > 0:
                        runtimes.append(current_runtime)
                    current_runtime = 0
            if current_runtime > 0:
                runtimes.append(current_runtime)
            all_runtimes.extend(runtimes)
        return all_runtimes
    
    baseline_runtimes = extract_runtimes(baseline_data['all_episodes'])
    pidrl_runtimes = extract_runtimes(pidrl_data['all_episodes'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline histogram
    ax1 = axes[0]
    ax1.hist(baseline_runtimes, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=15, color='red', linestyle='--', linewidth=2, label='15-min minimum')
    ax1.set_xlabel('ON-Runtime (minutes)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Baseline Thermostat', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # PI-DRL histogram
    ax2 = axes[1]
    ax2.hist(pidrl_runtimes, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(x=15, color='red', linestyle='--', linewidth=2, label='15-min minimum')
    ax2.set_xlabel('ON-Runtime (minutes)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('PI-DRL Controller', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Check safety: PI-DRL should have zero counts < 15 min
    pidrl_violations = sum(1 for r in pidrl_runtimes if r < 15)
    if pidrl_violations > 0:
        print(f"  ⚠ Warning: PI-DRL has {pidrl_violations} runtime violations < 15 min")
    else:
        print(f"  ✓ PI-DRL safety verified: zero violations < 15 min")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_2_Safety_Verification.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved Figure_2_Safety_Verification.png")


def generate_figure_3_policy_heatmap(
    env: SafetyHVACEnv,
    model: PPO,
    output_dir: str
):
    """Figure 3: Policy heatmap of P(ON) vs (hour-of-day, T_out)."""
    print("Generating Figure_3_Policy_Heatmap.png...")
    
    # Create grid
    hours = np.arange(0, 24, 0.5)
    T_out_range = np.arange(-5, 35, 1.0)
    
    prob_on = np.zeros((len(T_out_range), len(hours)))
    
    # Sample T_in and T_mass values
    T_in_sample = 21.0
    T_mass_sample = 21.0
    
    for i, T_out in enumerate(T_out_range):
        for j, hour in enumerate(hours):
            # Create observation
            time_sin = np.sin(2 * np.pi * hour / 24.0)
            time_cos = np.cos(2 * np.pi * hour / 24.0)
            Price = 0.30 if 16 <= hour < 21 else 0.10
            
            obs = np.array([T_in_sample, T_out, T_mass_sample, Price, time_sin, time_cos], dtype=np.float32)
            
            # Get action probabilities using policy's action distribution
            # For discrete actions, we can sample many times or use the policy directly
            # Use a simple approach: sample actions and compute empirical probability
            n_samples = 100
            actions = []
            for _ in range(n_samples):
                action, _ = model.predict(obs, deterministic=False)
                actions.append(action)
            prob_on[i, j] = np.mean([a == 1 for a in actions])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(prob_on, aspect='auto', origin='lower', cmap='RdYlBu_r', vmin=0, vmax=1)
    
    # Set ticks
    hour_ticks = np.arange(0, len(hours), 4)
    hour_labels = [f"{int(hours[i])}" for i in hour_ticks]
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels)
    
    temp_ticks = np.arange(0, len(T_out_range), 5)
    temp_labels = [f"{int(T_out_range[i])}" for i in temp_ticks]
    ax.set_yticks(temp_ticks)
    ax.set_yticklabels(temp_labels)
    
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Outdoor Temperature (°C)', fontsize=11)
    ax.set_title('PI-DRL Policy Heatmap: P(ON) vs (Hour, T_out)', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(ON)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_3_Policy_Heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved Figure_3_Policy_Heatmap.png")


def generate_figure_4_multi_objective_radar(
    baseline_data: Dict[str, Any],
    pidrl_data: Dict[str, Any],
    output_dir: str
):
    """Figure 4: Multi-objective radar chart."""
    print("Generating Figure_4_Multi_Objective_Radar.png...")
    
    # Extract metrics
    baseline_metrics = {
        'Cost': baseline_data['total_cost'],
        'Discomfort': baseline_data['total_discomfort'],
        'Peak Load': 4.0,  # Maximum power (simplified)
        'Cycles': baseline_data['total_cycles'],
        'Energy': baseline_data['total_energy']
    }
    
    pidrl_metrics = {
        'Cost': pidrl_data['total_cost'],
        'Discomfort': pidrl_data['total_discomfort'],
        'Peak Load': 4.0,  # Maximum power (simplified)
        'Cycles': pidrl_data['total_cycles'],
        'Energy': pidrl_data['total_energy']
    }
    
    # Normalize: higher is better (invert cost/discomfort/cycles/energy)
    categories = list(baseline_metrics.keys())
    n_categories = len(categories)
    
    # Normalize to [0, 1] where 1 = best (lowest cost/discomfort/etc)
    max_vals = {k: max(baseline_metrics[k], pidrl_metrics[k]) for k in categories}
    min_vals = {k: min(baseline_metrics[k], pidrl_metrics[k]) for k in categories}
    
    def normalize(val, key):
        if max_vals[key] == min_vals[key]:
            return 0.5
        # Invert so lower is better becomes higher normalized value
        return 1.0 - (val - min_vals[key]) / (max_vals[key] - min_vals[key])
    
    baseline_norm = [normalize(baseline_metrics[k], k) for k in categories]
    pidrl_norm = [normalize(pidrl_metrics[k], k) for k in categories]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    baseline_norm += baseline_norm[:1]
    pidrl_norm += pidrl_norm[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, baseline_norm, 'o-', linewidth=2, label='Baseline', color='blue', alpha=0.7)
    ax.fill(angles, baseline_norm, alpha=0.25, color='blue')
    
    ax.plot(angles, pidrl_norm, 'o-', linewidth=2, label='PI-DRL', color='red', alpha=0.7)
    ax.fill(angles, pidrl_norm, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Multi-Objective Performance Comparison\n(Normalized: Higher = Better)', 
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_4_Multi_Objective_Radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved Figure_4_Multi_Objective_Radar.png")


def generate_figure_5_robustness(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_controller: BaselineThermostat,
    pidrl_model: PPO,
    output_dir: str
):
    """Figure 5: Robustness - Total cost vs R-multiplier."""
    print("Generating Figure_5_Robustness.png...")
    
    R_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
    baseline_costs = []
    pidrl_costs = []
    
    for R_mult in R_multipliers:
        # Create environment with modified R parameters
        nominal_params = {
            'R_i': 0.05 * R_mult,
            'R_w': 0.02 * R_mult,
            'R_o': 0.03 * R_mult,
            'C_in': 0.3,
            'C_m': 0.5
        }
        
        # Evaluate baseline
        env_baseline = SafetyHVACEnv(test_df, nominal_params=nominal_params, use_domain_randomization=False)
        baseline_controller.reset()
        baseline_result = evaluate_controller(env_baseline, baseline_controller, n_episodes=3)
        baseline_costs.append(baseline_result['total_cost'])
        
        # Evaluate PI-DRL
        env_pidrl = SafetyHVACEnv(test_df, nominal_params=nominal_params, use_domain_randomization=False)
        pidrl_result = evaluate_controller(env_pidrl, pidrl_model, n_episodes=3)
        pidrl_costs.append(pidrl_result['total_cost'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(R_multipliers, baseline_costs, 'o-', linewidth=2, label='Baseline', color='blue', markersize=8)
    ax.plot(R_multipliers, pidrl_costs, 'o-', linewidth=2, label='PI-DRL', color='red', markersize=8)
    ax.set_xlabel('R-Multiplier', fontsize=11)
    ax.set_ylabel('Total Cost ($)', fontsize=11)
    ax.set_title('Robustness Analysis: Performance vs Parameter Variation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_5_Robustness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved Figure_5_Robustness.png")


def generate_figure_6_comfort_distribution(
    baseline_data: Dict[str, Any],
    pidrl_data: Dict[str, Any],
    output_dir: str
):
    """Figure 6: Comfort distribution - Violin/box plots."""
    print("Generating Figure_6_Comfort_Distribution.png...")
    
    # Collect all temperatures
    baseline_temps = []
    pidrl_temps = []
    
    for ep in baseline_data['all_episodes']:
        baseline_temps.extend(ep['episode_data']['T_in'])
    
    for ep in pidrl_data['all_episodes']:
        pidrl_temps.extend(ep['episode_data']['T_in'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create violin plot
    parts = ax.violinplot([baseline_temps, pidrl_temps], positions=[1, 2], widths=0.6, showmeans=True, showmedians=True)
    
    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    # Add box plot overlay
    bp = ax.boxplot([baseline_temps, pidrl_temps], positions=[1, 2], widths=0.3, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.5)
    
    # Highlight comfort band
    ax.axhspan(19.5, 24.0, alpha=0.2, color='green', label='Comfort Band [19.5, 24.0]°C')
    ax.axhline(y=21.0, color='g', linestyle='--', linewidth=1.5, label='Setpoint 21°C')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', 'PI-DRL'], fontsize=11)
    ax.set_ylabel('Indoor Temperature (°C)', fontsize=11)
    ax.set_title('Comfort Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_6_Comfort_Distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved Figure_6_Comfort_Distribution.png")


def generate_figure_7_price_response(
    baseline_data: Dict[str, Any],
    pidrl_data: Dict[str, Any],
    output_dir: str
):
    """Figure 7: Price response - Average power vs hour-of-day."""
    print("Generating Figure_7_Price_Response.png...")
    
    # Collect power by hour
    baseline_power_by_hour = {h: [] for h in range(24)}
    pidrl_power_by_hour = {h: [] for h in range(24)}
    
    for ep in baseline_data['all_episodes']:
        ep_data = ep['episode_data']
        for i, power in enumerate(ep_data['power']):
            hour = i % (24 * 60) // 60
            baseline_power_by_hour[hour].append(power)
    
    for ep in pidrl_data['all_episodes']:
        ep_data = ep['episode_data']
        for i, power in enumerate(ep_data['power']):
            hour = i % (24 * 60) // 60
            pidrl_power_by_hour[hour].append(power)
    
    hours = np.arange(24)
    baseline_avg = [np.mean(baseline_power_by_hour[h]) if baseline_power_by_hour[h] else 0 for h in hours]
    pidrl_avg = [np.mean(pidrl_power_by_hour[h]) if pidrl_power_by_hour[h] else 0 for h in hours]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hours, baseline_avg, 'o-', linewidth=2, label='Baseline', color='blue', markersize=6)
    ax.plot(hours, pidrl_avg, 'o-', linewidth=2, label='PI-DRL', color='red', markersize=6)
    
    # Highlight peak hours
    ax.axvspan(16, 21, alpha=0.2, color='orange', label='Peak Hours (16:00-21:00)')
    
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Average Power (kW)', fontsize=11)
    ax.set_title('Price Response: Average Power vs Hour-of-Day', fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_7_Price_Response.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved Figure_7_Price_Response.png")


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_table_1_system_parameters(output_dir: str, env: SafetyHVACEnv, model: PPO):
    """Table 1: System parameters."""
    print("Generating Table_1_System_Parameters.csv...")
    
    params = env.nominal_params
    
    data = {
        'Parameter': [
            'R_i (K/kW)', 'R_w (K/kW)', 'R_o (K/kW)', 'C_in (kWh/K)', 'C_m (kWh/K)',
            'Lockout Time (min)', 'Setpoint (°C)', 'Deadband (°C)',
            'PPO Learning Rate', 'PPO n_steps', 'PPO batch_size', 'PPO n_epochs',
            'lambda_cost', 'lambda_discomfort', 'lambda_penalty'
        ],
        'Value': [
            params['R_i'], params['R_w'], params['R_o'], params['C_in'], params['C_m'],
            env.min_runtime, 21.0, 1.5,
            3e-4, 2048, 64, 10,
            env.lambda_cost, env.lambda_discomfort, env.lambda_penalty
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'Table_1_System_Parameters.csv'), index=False)
    print("  ✓ Saved Table_1_System_Parameters.csv")


def generate_table_2_performance_summary(
    baseline_data: Dict[str, Any],
    pidrl_data: Dict[str, Any],
    output_dir: str
):
    """Table 2: Performance summary."""
    print("Generating Table_2_Performance_Summary.csv...")
    
    data = {
        'Controller': ['Baseline', 'PI-DRL'],
        'Total Cost ($)': [baseline_data['total_cost'], pidrl_data['total_cost']],
        'Total Discomfort': [baseline_data['total_discomfort'], pidrl_data['total_discomfort']],
        'Total Cycles': [baseline_data['total_cycles'], pidrl_data['total_cycles']],
        'Total Energy (kWh)': [baseline_data['total_energy'], pidrl_data['total_energy']]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'Table_2_Performance_Summary.csv'), index=False)
    print("  ✓ Saved Table_2_Performance_Summary.csv")


def generate_table_3_grid_impact(
    baseline_data: Dict[str, Any],
    pidrl_data: Dict[str, Any],
    output_dir: str
):
    """Table 3: Grid impact."""
    print("Generating Table_3_Grid_Impact.csv...")
    
    # Calculate average power in peak (16-21) and off-peak hours
    def calc_peak_offpeak(episode_list):
        peak_powers = []
        offpeak_powers = []
        for ep in episode_list:
            ep_data = ep['episode_data']
            for i, power in enumerate(ep_data['power']):
                hour = i % (24 * 60) // 60
                if 16 <= hour < 21:
                    peak_powers.append(power)
                else:
                    offpeak_powers.append(power)
        return np.mean(peak_powers) if peak_powers else 0, np.mean(offpeak_powers) if offpeak_powers else 0
    
    baseline_peak, baseline_offpeak = calc_peak_offpeak(baseline_data['all_episodes'])
    pidrl_peak, pidrl_offpeak = calc_peak_offpeak(pidrl_data['all_episodes'])
    
    peak_reduction = ((baseline_peak - pidrl_peak) / baseline_peak * 100) if baseline_peak > 0 else 0
    
    data = {
        'Controller': ['Baseline', 'PI-DRL'],
        'Avg Power Peak (16-21) (kW)': [baseline_peak, pidrl_peak],
        'Avg Power Off-Peak (kW)': [baseline_offpeak, pidrl_offpeak],
        'Peak Reduction (%)': [0, peak_reduction]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'Table_3_Grid_Impact.csv'), index=False)
    print("  ✓ Saved Table_3_Grid_Impact.csv")


def generate_table_4_safety_shield_activity(
    baseline_data: Dict[str, Any],
    pidrl_data: Dict[str, Any],
    output_dir: str
):
    """Table 4: Safety shield activity."""
    print("Generating Table_4_Safety_Shield_Activity.csv...")
    
    # Calculate totals
    baseline_total_steps = sum(len(ep['episode_data']['actions']) for ep in baseline_data['all_episodes'])
    pidrl_total_steps = sum(len(ep['episode_data']['actions']) for ep in pidrl_data['all_episodes'])
    
    baseline_masked_off = sum(ep['masked_off_count'] for ep in baseline_data['all_episodes'])
    baseline_masked_on = sum(ep['masked_on_count'] for ep in baseline_data['all_episodes'])
    
    pidrl_masked_off = pidrl_data['masked_off_count']
    pidrl_masked_on = pidrl_data['masked_on_count']
    
    baseline_mask_pct = ((baseline_masked_off + baseline_masked_on) / baseline_total_steps * 100) if baseline_total_steps > 0 else 0
    pidrl_mask_pct = ((pidrl_masked_off + pidrl_masked_on) / pidrl_total_steps * 100) if pidrl_total_steps > 0 else 0
    
    data = {
        'Phase': ['Training', 'Testing'],
        'Total Timesteps': [0, pidrl_total_steps],  # Training not tracked separately
        'Masked OFF Actions': [0, pidrl_masked_off],
        'Masked ON Actions': [0, pidrl_masked_on],
        'Mask Active (%)': [0, pidrl_mask_pct]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'Table_4_Safety_Shield_Activity.csv'), index=False)
    print("  ✓ Saved Table_4_Safety_Shield_Activity.csv")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Robust, Safety-Critical 2R2C PI-DRL Controller for Residential HVAC")
    print("="*80)
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    
    # Step 1: Download and process data
    train_df, test_df = download_and_process_data()
    
    # Step 2: Create environments
    print("\n" + "="*80)
    print("STEP 2: Creating Environments")
    print("="*80)
    
    # Training environment with domain randomization
    train_env = SafetyHVACEnv(
        train_df,
        use_domain_randomization=True,
        episode_length_days=7,
        seed=42
    )
    
    # Test environment without domain randomization
    test_env = SafetyHVACEnv(
        test_df,
        use_domain_randomization=False,
        episode_length_days=7,
        seed=42
    )
    
    print("Environments created successfully!")
    print("="*80)
    
    # Step 3: Train PI-DRL agent
    pidrl_model = train_pidrl_agent(train_env, total_timesteps=200000)
    
    # Step 4: Evaluate controllers
    print("\n" + "="*80)
    print("STEP 3: Evaluating Controllers")
    print("="*80)
    
    # Evaluate baseline
    print("Evaluating Baseline Thermostat...")
    baseline_controller = BaselineThermostat(setpoint=21.0, deadband=1.5)
    baseline_results = evaluate_controller(test_env, baseline_controller, n_episodes=5)
    
    print(f"Baseline Results:")
    print(f"  Total Cost: ${baseline_results['total_cost']:.2f}")
    print(f"  Total Discomfort: {baseline_results['total_discomfort']:.2f}")
    print(f"  Total Cycles: {baseline_results['total_cycles']:.0f}")
    print(f"  Total Energy: {baseline_results['total_energy']:.2f} kWh")
    
    # Evaluate PI-DRL
    print("\nEvaluating PI-DRL Controller...")
    pidrl_results = evaluate_controller(test_env, pidrl_model, n_episodes=5)
    
    print(f"PI-DRL Results:")
    print(f"  Total Cost: ${pidrl_results['total_cost']:.2f}")
    print(f"  Total Discomfort: {pidrl_results['total_discomfort']:.2f}")
    print(f"  Total Cycles: {pidrl_results['total_cycles']:.0f}")
    print(f"  Total Energy: {pidrl_results['total_energy']:.2f} kWh")
    print(f"  Masked OFF: {pidrl_results['masked_off_count']}")
    print(f"  Masked ON: {pidrl_results['masked_on_count']}")
    print("="*80)
    
    # Step 5: Generate figures
    print("\n" + "="*80)
    print("STEP 4: Generating Figures")
    print("="*80)
    
    generate_figure_1_micro_dynamics(baseline_results, pidrl_results, output_dir)
    generate_figure_2_safety_verification(baseline_results, pidrl_results, output_dir)
    generate_figure_3_policy_heatmap(test_env, pidrl_model, output_dir)
    generate_figure_4_multi_objective_radar(baseline_results, pidrl_results, output_dir)
    generate_figure_5_robustness(train_df, test_df, baseline_controller, pidrl_model, output_dir)
    generate_figure_6_comfort_distribution(baseline_results, pidrl_results, output_dir)
    generate_figure_7_price_response(baseline_results, pidrl_results, output_dir)
    
    # Step 6: Generate tables
    print("\n" + "="*80)
    print("STEP 5: Generating Tables")
    print("="*80)
    
    generate_table_1_system_parameters(output_dir, test_env, pidrl_model)
    generate_table_2_performance_summary(baseline_results, pidrl_results, output_dir)
    generate_table_3_grid_impact(baseline_results, pidrl_results, output_dir)
    generate_table_4_safety_shield_activity(baseline_results, pidrl_results, output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE! All outputs saved to:", output_dir)
    print("="*80)
    print("\nGenerated outputs:")
    print("  Figures: 7 PNG files (300 dpi)")
    print("  Tables: 4 CSV files")
    print("="*80)
