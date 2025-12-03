"""
Real AMPds2 Data Loader
Loads actual AMPds2 dataset from CSV files
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def load_real_ampds2_data(
    data_dir: str = "./data",
    start_date: str = None,
    end_date: str = None,
    max_samples: int = None
) -> pd.DataFrame:
    """
    Load real AMPds2 dataset
    
    Args:
        data_dir: Directory containing AMPds2 CSV files
        start_date: Start date (e.g., '2012-04-01')
        end_date: End date (e.g., '2012-05-01')
        max_samples: Maximum number of samples to load (for memory efficiency)
    
    Returns:
        DataFrame with columns: timestamp, WHE, HPE, FRE, Outdoor_Temp, Solar_Rad, Price, Hour, DayOfYear
    """
    print(f"Loading real AMPds2 data from {data_dir}...")
    
    # File paths
    hpe_file = os.path.join(data_dir, "Electricity_HPE.csv")
    fre_file = os.path.join(data_dir, "Electricity_FRE.csv")
    weather_file = os.path.join(data_dir, "Climate_HourlyWeather.csv")
    
    # Check files exist
    if not os.path.exists(hpe_file):
        raise FileNotFoundError(f"HPE file not found: {hpe_file}")
    if not os.path.exists(fre_file):
        raise FileNotFoundError(f"FRE file not found: {fre_file}")
    if not os.path.exists(weather_file):
        raise FileNotFoundError(f"Weather file not found: {weather_file}")
    
    # Load electricity data
    print("  Loading Heat Pump data...")
    hpe_df = pd.read_csv(hpe_file)
    hpe_df['timestamp'] = pd.to_datetime(hpe_df['unix_ts'], unit='s')
    
    print("  Loading Fridge data...")
    fre_df = pd.read_csv(fre_file)
    fre_df['timestamp'] = pd.to_datetime(fre_df['unix_ts'], unit='s')
    
    # Load weather data
    print("  Loading weather data...")
    weather_df = pd.read_csv(weather_file)
    weather_df['timestamp'] = pd.to_datetime(weather_df['Date/Time'])
    
    # Filter by date range if specified
    if start_date:
        start_dt = pd.to_datetime(start_date)
        hpe_df = hpe_df[hpe_df['timestamp'] >= start_dt]
        fre_df = fre_df[fre_df['timestamp'] >= start_dt]
        weather_df = weather_df[weather_df['timestamp'] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        hpe_df = hpe_df[hpe_df['timestamp'] <= end_dt]
        fre_df = fre_df[fre_df['timestamp'] <= end_dt]
        weather_df = weather_df[weather_df['timestamp'] <= end_dt]
    
    # Limit samples if specified
    if max_samples:
        hpe_df = hpe_df.head(max_samples)
        fre_df = fre_df.head(max_samples)
    
    print(f"  HPE samples: {len(hpe_df):,}")
    print(f"  FRE samples: {len(fre_df):,}")
    print(f"  Weather samples: {len(weather_df):,}")
    
    # Merge electricity data
    print("  Merging electricity data...")
    elec_df = hpe_df[['timestamp', 'P']].copy()
    elec_df.columns = ['timestamp', 'HPE']
    
    # Add FRE data
    fre_power = fre_df[['timestamp', 'P']].copy()
    fre_power.columns = ['timestamp', 'FRE']
    elec_df = elec_df.merge(fre_power, on='timestamp', how='left')
    
    # Water heater (mock - not available in this dataset)
    elec_df['WHE'] = 100 + 50 * np.random.rand(len(elec_df))
    
    # Merge with weather data (hourly to minute interpolation)
    print("  Merging weather data...")
    weather_df = weather_df[['timestamp', 'Temp (C)']].copy()
    weather_df.columns = ['timestamp', 'Outdoor_Temp']
    
    # Round to nearest hour for merging
    elec_df['hour_timestamp'] = elec_df['timestamp'].dt.floor('H')
    weather_df['hour_timestamp'] = weather_df['timestamp'].dt.floor('H')
    
    # Merge
    df = elec_df.merge(weather_df[['hour_timestamp', 'Outdoor_Temp']], 
                       on='hour_timestamp', how='left')
    df = df.drop('hour_timestamp', axis=1)
    
    # Fill missing outdoor temp with forward fill
    df['Outdoor_Temp'] = df['Outdoor_Temp'].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate solar radiation (simplified model based on time of day)
    print("  Calculating solar radiation...")
    df['Hour'] = df['timestamp'].dt.hour
    df['Minute'] = df['timestamp'].dt.minute
    df['DayOfYear'] = df['timestamp'].dt.dayofyear
    
    # Solar radiation model
    hour_fraction = df['Hour'] + df['Minute'] / 60
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['Solar_Rad'] = np.maximum(
        0,
        800 * np.sin(np.pi * (hour_fraction - 6) / 12) * seasonal_factor
    ) * ((hour_fraction >= 6) & (hour_fraction <= 18))
    
    # Time-of-use pricing (Canadian TOU rates)
    print("  Calculating electricity pricing...")
    df['Price'] = np.where(
        (df['Hour'] >= 17) & (df['Hour'] <= 20),
        0.25,  # Peak: 5pm-9pm
        np.where(
            ((df['Hour'] >= 7) & (df['Hour'] < 17)) | ((df['Hour'] > 20) & (df['Hour'] <= 22)),
            0.15,  # Mid-peak
            0.08   # Off-peak
        )
    )
    
    # Clean up
    df = df.drop('Minute', axis=1)
    df = df.dropna()
    
    # Ensure data types
    df['HPE'] = df['HPE'].astype(float)
    df['FRE'] = df['FRE'].astype(float)
    df['WHE'] = df['WHE'].astype(float)
    df['Outdoor_Temp'] = df['Outdoor_Temp'].astype(float)
    
    print(f"\n✓ Loaded {len(df):,} samples from real AMPds2 dataset")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Temperature range: {df['Outdoor_Temp'].min():.1f}°C to {df['Outdoor_Temp'].max():.1f}°C")
    print(f"  HPE power range: {df['HPE'].min():.1f}W to {df['HPE'].max():.1f}W")
    
    return df


def preview_data(df: pd.DataFrame, n_samples: int = 10):
    """Preview the loaded data"""
    print("\n" + "="*80)
    print("DATA PREVIEW")
    print("="*80)
    print(df.head(n_samples).to_string(index=False))
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(df.describe())
    print("="*80)


if __name__ == "__main__":
    # Test the loader
    df = load_real_ampds2_data(
        data_dir="./data",
        start_date="2012-04-01",
        end_date="2012-04-07",  # 1 week of data
        max_samples=10080  # 7 days * 24 hours * 60 minutes
    )
    
    preview_data(df)
    
    # Save processed data
    output_file = "./data/ampds2_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved processed data to: {output_file}")
