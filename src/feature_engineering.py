"""
Thermodynamic Feature Engineering Module

This module implements physics-aware feature engineering from a mechanical
engineering perspective, transforming raw sensor data into thermodynamic states
that reflect the building's energy dynamics.

Key Features:
- Enthalpy Calculation: Air enthalpy from Temperature and Relative Humidity
- Thermal Inertia Lag: EMA on indoor temperature (building envelope heat storage)
- Delta-T: Indoor-Outdoor temperature difference (driving force for heat transfer)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThermodynamicFeatureEngineer:
    """
    Engineers physics-based features that reflect thermodynamic states.
    
    Mechanical Perspective:
    - Enthalpy (h): Total energy content of air (sensible + latent)
    - Thermal Inertia: Building envelope's heat storage capacity
    - Delta-T: Driving force for conductive heat transfer
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer with thermodynamic parameters.
        
        Args:
            config: Configuration dictionary with feature engineering parameters
        """
        self.config = config
        fe_config = config.get('feature_engineering', {})
        
        # Thermodynamic constants
        enthalpy_config = fe_config.get('enthalpy', {})
        self.cp_air = enthalpy_config.get('cp_air', 1.006)  # kJ/(kg·K)
        self.hfg = enthalpy_config.get('hfg', 2501.0)  # kJ/kg
        
        # EMA parameters
        self.ema_alpha = fe_config.get('ema_alpha', 0.3)
        
        # Standardization
        self.standardize = fe_config.get('standardize', True)
        
    def calculate_enthalpy(self, T: pd.Series, RH: pd.Series, 
                          P_atm: float = 101.325) -> pd.Series:
        """
        Calculate air enthalpy from Temperature and Relative Humidity.
        
        Enthalpy represents the total energy content of air, crucial for
        latent vs. sensible load analysis in HVAC systems.
        
        Formula:
        h = cp_air * T + ω * (hfg + cp_vapor * T)
        where ω is the humidity ratio
        
        Args:
            T: Temperature in Celsius
            RH: Relative Humidity in percentage (0-100)
            P_atm: Atmospheric pressure in kPa (default: 101.325 kPa)
            
        Returns:
            Enthalpy in kJ/kg
        """
        # Convert temperature to Kelvin
        T_K = T + 273.15
        
        # Saturation pressure of water vapor (Antoine equation approximation)
        # P_sat = 0.61078 * exp(17.27 * T / (T + 237.3)) [kPa]
        P_sat = 0.61078 * np.exp(17.27 * T / (T + 237.3))
        
        # Partial pressure of water vapor
        P_vapor = (RH / 100.0) * P_sat
        
        # Humidity ratio (kg water vapor / kg dry air)
        # ω = 0.622 * P_vapor / (P_atm - P_vapor)
        omega = 0.622 * P_vapor / (P_atm - P_vapor)
        
        # Specific heat of water vapor [kJ/(kg·K)]
        cp_vapor = 1.86
        
        # Enthalpy calculation
        # h = cp_air * T + ω * (hfg + cp_vapor * T)
        h = self.cp_air * T + omega * (self.hfg + cp_vapor * T)
        
        return h
    
    def compute_thermal_inertia(self, T_in: pd.Series) -> pd.Series:
        """
        Apply Exponential Moving Average (EMA) on indoor temperature.
        
        This represents the building envelope's heat storage capacity (mass).
        Thermal inertia causes temperature to lag behind setpoint changes,
        which is critical for HVAC control optimization.
        
        Args:
            T_in: Indoor temperature time series
            
        Returns:
            EMA-smoothed temperature representing thermal inertia
        """
        # Exponential Moving Average
        T_inertia = T_in.ewm(alpha=self.ema_alpha, adjust=False).mean()
        
        return T_inertia
    
    def compute_delta_t(self, T_in: pd.Series, T_out: pd.Series) -> pd.Series:
        """
        Compute Delta-T: difference between Outdoor and Indoor temperature.
        
        ΔT = T_out - T_in
        
        This represents the driving force of conductive heat transfer through
        the building envelope (Fourier's Law: Q ∝ ΔT).
        
        Args:
            T_in: Indoor temperature
            T_out: Outdoor temperature
            
        Returns:
            Temperature difference (Delta-T)
        """
        delta_t = T_out - T_in
        return delta_t
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete thermodynamic feature engineering pipeline.
        
        Args:
            df: Input dataframe with raw sensor data
            
        Returns:
            DataFrame with engineered thermodynamic features
        """
        logger.info("Starting thermodynamic feature engineering...")
        df_engineered = df.copy()
        
        # Find temperature and humidity columns
        t_in_cols = [col for col in df.columns if 'T_in' in col or col == 'T1']
        t_out_cols = [col for col in df.columns if 'T_out' in col]
        rh_in_cols = [col for col in df.columns if 'RH_in' in col or 'RH_1' in col]
        rh_out_cols = [col for col in df.columns if 'RH_out' in col]
        
        # Use first available columns
        T_in = df[t_in_cols[0]] if t_in_cols else None
        T_out = df[t_out_cols[0]] if t_out_cols else None
        RH_in = df[rh_in_cols[0]] if rh_in_cols else None
        RH_out = df[rh_out_cols[0]] if rh_out_cols else None
        
        # 1. Calculate Enthalpy (if temperature and humidity available)
        if T_in is not None and RH_in is not None:
            df_engineered['h_in'] = self.calculate_enthalpy(T_in, RH_in)
            logger.info("Computed indoor air enthalpy (h_in)")
        
        if T_out is not None and RH_out is not None:
            df_engineered['h_out'] = self.calculate_enthalpy(T_out, RH_out)
            logger.info("Computed outdoor air enthalpy (h_out)")
        
        # 2. Thermal Inertia Lag (EMA on indoor temperature)
        if T_in is not None:
            df_engineered['T_inertia'] = self.compute_thermal_inertia(T_in)
            logger.info("Computed thermal inertia (EMA-smoothed T_in)")
        
        # 3. Delta-T (if not already computed in harmonization)
        if 'Delta_T' not in df_engineered.columns:
            if T_in is not None and T_out is not None:
                df_engineered['Delta_T'] = self.compute_delta_t(T_in, T_out)
                logger.info("Computed Delta-T (T_out - T_in)")
        
        # 4. Additional physics-based features
        if T_in is not None and T_out is not None:
            # Enthalpy difference (driving force for latent load)
            if 'h_in' in df_engineered.columns and 'h_out' in df_engineered.columns:
                df_engineered['Delta_h'] = df_engineered['h_out'] - df_engineered['h_in']
            
            # Temperature gradient magnitude
            df_engineered['|Delta_T|'] = np.abs(df_engineered['Delta_T'])
        
        # Handle missing values
        df_engineered = df_engineered.ffill().bfill()
        
        logger.info(f"Feature engineering completed. Added {len(df_engineered.columns) - len(df.columns)} new features")
        
        return df_engineered
    
    def standardize_features(self, df: pd.DataFrame, 
                           feature_cols: Optional[list] = None,
                           fit_scaler: Optional[object] = None) -> tuple:
        """
        Standardize features to create unified feature space.
        
        Args:
            df: Input dataframe
            feature_cols: List of columns to standardize (if None, uses all numeric)
            fit_scaler: Fitted scaler (if None, fits new scaler)
            
        Returns:
            Tuple of (standardized_df, fitted_scaler)
        """
        from sklearn.preprocessing import StandardScaler
        
        if feature_cols is None:
            # Exclude target variable and datetime columns
            exclude_cols = ['Appliances', 'E_load', 'date']
            feature_cols = [col for col in df.columns 
                          if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        if fit_scaler is None:
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        else:
            scaler = fit_scaler
            df_scaled = df.copy()
            df_scaled[feature_cols] = scaler.transform(df[feature_cols])
        
        return df_scaled, scaler
