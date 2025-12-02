#!/usr/bin/env python3
"""
Quick Statistics Summary Tool
Generates a concise terminal-based summary of the energy consumption dataset.

Usage:
    python quick_stats.py [--csv energydata_complete.csv]
"""

import pandas as pd
import argparse
from datetime import datetime


def format_value(val, unit=""):
    """Format numeric value for display."""
    if abs(val) >= 1000:
        return f"{val:,.1f}{unit}"
    elif abs(val) >= 1:
        return f"{val:.2f}{unit}"
    else:
        return f"{val:.4f}{unit}"


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def quick_summary(filepath='energydata_complete.csv'):
    """Generate and print a quick summary of the energy dataset."""
    
    # Load data
    print(f"\n📊 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Basic info
    print_section("📁 DATASET OVERVIEW")
    print(f"  Records:     {len(df):,}")
    print(f"  Features:    {len(df.columns)}")
    print(f"  Date range:  {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Duration:    {(df['date'].max() - df['date'].min()).days} days")
    
    # Energy consumption stats
    print_section("⚡ ENERGY CONSUMPTION (Appliances)")
    appliances = df['Appliances']
    print(f"  Mean:        {format_value(appliances.mean(), ' Wh')}")
    print(f"  Median:      {format_value(appliances.median(), ' Wh')}")
    print(f"  Std Dev:     {format_value(appliances.std(), ' Wh')}")
    print(f"  Min:         {format_value(appliances.min(), ' Wh')}")
    print(f"  Max:         {format_value(appliances.max(), ' Wh')}")
    print(f"  Total:       {format_value(appliances.sum() / 1000, ' kWh')}")
    
    # Temperature summary
    print_section("🌡️  TEMPERATURE SENSORS")
    temp_cols = [c for c in df.columns if c.startswith('T') and c not in ['T_out']]
    for col in temp_cols[:5]:  # Show first 5
        print(f"  {col:8s}:   {df[col].mean():.1f}°C  (range: {df[col].min():.1f} - {df[col].max():.1f})")
    if len(temp_cols) > 5:
        print(f"  ... and {len(temp_cols) - 5} more sensors")
    
    # Peak hours analysis
    print_section("🕐 PEAK CONSUMPTION HOURS")
    df['hour'] = df['date'].dt.hour
    hourly_avg = df.groupby('hour')['Appliances'].mean()
    top_hours = hourly_avg.nlargest(3)
    for hour, val in top_hours.items():
        print(f"  {hour:02d}:00       {format_value(val, ' Wh')} avg")
    
    # Data quality
    print_section("✅ DATA QUALITY")
    missing = df.isnull().sum().sum()
    print(f"  Missing values:  {missing}")
    print(f"  Completeness:    {(1 - missing / df.size) * 100:.1f}%")
    
    print(f"\n{'═' * 50}")
    print(f"  Summary generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick energy dataset statistics")
    parser.add_argument('--csv', default='energydata_complete.csv', 
                        help='Path to the CSV file')
    args = parser.parse_args()
    
    quick_summary(args.csv)
