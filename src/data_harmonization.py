"""
Multi-Source Data Harmonization & Physics-Invariant Feature Alignment

This module implements rigorous harmonization protocols to establish a unified
feature space across heterogeneous building typologies, enabling robust
generalization for the Digital Twin surrogate model.

Key Features:
- Schema Standardization: Maps inconsistent variable identifiers to standard nomenclature
- Temporal Synchronization: Resamples time-series to unified frequency
- Physics-Based Feature Synthesis: Derives Delta-T gradient (Indoor-Outdoor Temperature)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHarmonizer:
    """
    Harmonizes heterogeneous building energy datasets to create a unified feature space.
    
    Scientific Justification:
    According to Fourier's Law of Conduction (Q ∝ ΔT), the temperature difference
    is the primary driving force for heat flux through the building envelope.
    By explicitly calculating ΔT, we provide the model with a "common thermodynamic
    language," allowing it to learn fundamental heat transfer laws irrespective of
    the specific building's location or baseline temperature.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataHarmonizer with configuration.
        
        Args:
            config: Configuration dictionary containing harmonization parameters
        """
        self.config = config
        self.resample_freq = config.get('resample_freq', '1H')
        
        # Standardized nomenclature mapping
        self.standard_mapping = {
            # Temperature variables
            'Outdoor Temp': 'T_out',
            'Dry Bulb': 'T_out',
            'T_outdoor': 'T_out',
            'Outdoor_Temperature': 'T_out',
            'T_out': 'T_out',
            
            'Indoor Temp': 'T_in',
            'T_indoor': 'T_in',
            'Indoor_Temperature': 'T_in',
            'T_in': 'T_in',
            
            # Humidity variables
            'RH_out': 'RH_out',
            'RH_outdoor': 'RH_out',
            'Outdoor_RH': 'RH_out',
            'Relative Humidity Outdoor': 'RH_out',
            
            'RH_in': 'RH_in',
            'RH_indoor': 'RH_in',
            'Indoor_RH': 'RH_in',
            'Relative Humidity Indoor': 'RH_in',
            
            # Energy variables
            'Use [kW]': 'E_load',
            'Appliances': 'E_load',
            'Energy': 'E_load',
            'Power': 'E_load',
            'Load': 'E_load',
        }
        
    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize variable identifiers to create dataset-agnostic input vector.
        
        Args:
            df: Input dataframe with potentially inconsistent column names
            
        Returns:
            DataFrame with standardized column names
        """
        df_standardized = df.copy()
        
        # Map columns to standard nomenclature
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            for key, value in self.standard_mapping.items():
                if key.lower() in col_lower:
                    column_mapping[col] = value
                    break
        
        df_standardized = df_standardized.rename(columns=column_mapping)
        
        logger.info(f"Schema standardization: {len(column_mapping)} columns mapped")
        return df_standardized
    
    def synchronize_temporal(self, df: pd.DataFrame, 
                            datetime_col: Optional[str] = None) -> pd.DataFrame:
        """
        Resample time-series data to unified frequency.
        
        Objective: Align dynamic response characteristics of different buildings
        and eliminate sampling rate bias.
        
        Args:
            df: Input dataframe with time-series data
            datetime_col: Name of datetime column (if None, uses index)
            
        Returns:
            Resampled dataframe with unified temporal frequency
        """
        df_sync = df.copy()
        
        # Ensure datetime index
        if datetime_col and datetime_col in df_sync.columns:
            df_sync[datetime_col] = pd.to_datetime(df_sync[datetime_col], errors='coerce')
            df_sync = df_sync.set_index(datetime_col)
        elif not isinstance(df_sync.index, pd.DatetimeIndex):
            # Try to infer datetime from index
            try:
                df_sync.index = pd.to_datetime(df_sync.index, errors='coerce')
            except:
                logger.warning("Could not convert index to datetime. Creating default index.")
                df_sync.index = pd.date_range(start='2020-01-01', periods=len(df_sync), freq='1H')
        
        # Remove any rows with NaT (Not a Time) in index
        df_sync = df_sync[~df_sync.index.isna()]
        
        # CRITICAL FIX: Only resample numeric columns to prevent TypeError
        # Convert all columns to numeric where possible, coercing errors to NaN
        for col in df_sync.columns:
            # Try to convert to numeric (coerce errors to NaN)
            df_sync[col] = pd.to_numeric(df_sync[col], errors='coerce')
        
        # Filter to only numeric columns using select_dtypes (more reliable)
        df_sync = df_sync.select_dtypes(include=[np.number])
        
        if df_sync.empty or len(df_sync.columns) == 0:
            raise ValueError("No numeric columns found after type coercion. Check data format.")
        
        # Identify continuous vs cumulative variables
        continuous_vars = ['T_out', 'T_in', 'RH_out', 'RH_in', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9']
        cumulative_vars = ['E_load', 'Appliances', 'lights']
        
        # Resample: mean for continuous, sum for cumulative
        resampled_data = {}
        
        for col in df_sync.columns:
            if col in continuous_vars or any(var in col for var in continuous_vars):
                resampled_data[col] = df_sync[col].resample(self.resample_freq).mean()
            elif col in cumulative_vars or any(var in col for var in cumulative_vars):
                resampled_data[col] = df_sync[col].resample(self.resample_freq).sum()
            else:
                # Default to mean for other numeric columns
                resampled_data[col] = df_sync[col].resample(self.resample_freq).mean()
        
        df_resampled = pd.DataFrame(resampled_data)
        df_resampled = df_resampled.dropna()
        
        logger.info(f"Temporal synchronization: Resampled to {self.resample_freq} frequency")
        logger.info(f"Original length: {len(df_sync)}, Resampled length: {len(df_resampled)}")
        
        return df_resampled
    
    def compute_delta_t(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Indoor-Outdoor Temperature Gradient (ΔT).
        
        Scientific Justification:
        ΔT(t) = T_in(t) - T_out(t)
        
        This physics-informed feature represents the driving force for heat flux
        through the building envelope (Fourier's Law: Q ∝ ΔT). By explicitly
        calculating ΔT, the model learns fundamental heat transfer laws rather
        than site-specific correlations.
        
        Args:
            df: DataFrame with T_in and T_out columns
            
        Returns:
            DataFrame with added Delta_T column
        """
        df_with_delta = df.copy()
        
        # Find temperature columns
        t_in_cols = [col for col in df.columns if 'T_in' in col or 'T1' in col]
        t_out_cols = [col for col in df.columns if 'T_out' in col]
        
        if not t_in_cols or not t_out_cols:
            # Try alternative naming conventions
            if 'T1' in df.columns:
                # Assume T1 is indoor temperature (common in energy datasets)
                t_in_cols = ['T1']
            if 'T_out' not in df.columns and 'T_outdoor' in df.columns:
                t_out_cols = ['T_outdoor']
        
        if t_in_cols and t_out_cols:
            # Use first available column
            t_in = df[t_in_cols[0]]
            t_out = df[t_out_cols[0]]
            
            df_with_delta['Delta_T'] = t_in - t_out
            logger.info("Computed Delta_T (Indoor-Outdoor Temperature Gradient)")
        else:
            logger.warning("Could not find T_in and T_out columns for Delta_T computation")
            df_with_delta['Delta_T'] = 0.0
        
        return df_with_delta
    
    def harmonize(self, df: pd.DataFrame, 
                  datetime_col: Optional[str] = None,
                  compute_physics_features: bool = True) -> pd.DataFrame:
        """
        Complete harmonization pipeline: Schema → Temporal → Physics Features.
        
        Args:
            df: Input dataframe
            datetime_col: Name of datetime column
            compute_physics_features: Whether to compute physics-based features (Delta-T)
            
        Returns:
            Fully harmonized dataframe
        """
        logger.info("Starting data harmonization pipeline...")
        
        # Step 1: Schema Standardization
        df_harmonized = self.standardize_schema(df)
        
        # Auto-detect datetime column if not provided
        if datetime_col is None:
            datetime_candidates = ['date', 'Date', 'datetime', 'DateTime', 'time', 'Time']
            for candidate in datetime_candidates:
                if candidate in df_harmonized.columns:
                    datetime_col = candidate
                    break
        
        # Step 2: Temporal Synchronization
        df_harmonized = self.synchronize_temporal(df_harmonized, datetime_col)
        
        # Step 3: Physics-Based Feature Synthesis
        if compute_physics_features:
            df_harmonized = self.compute_delta_t(df_harmonized)
        
        logger.info("Data harmonization completed successfully")
        return df_harmonized


def load_and_harmonize_data(file_path: str, config: Dict) -> pd.DataFrame:
    """
    Convenience function to load and harmonize data from file.
    
    Args:
        file_path: Path to CSV file
        config: Configuration dictionary
        
    Returns:
        Harmonized dataframe
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    harmonizer = DataHarmonizer(config)
    df_harmonized = harmonizer.harmonize(df)
    
    return df_harmonized
