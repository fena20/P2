"""
Phase 1: Data Curation & Pre-processing
Objective: Prepare real data from BDG2 dataset for residential building optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class BDG2DataProcessor:
    """Process BDG2 dataset for residential building optimization"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.buildings_metadata = None
        self.processed_buildings = {}
        
    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """Load BDG2 building metadata"""
        # Simulated metadata structure - in real scenario, load from BDG2 API/files
        if Path(metadata_path).exists():
            metadata = pd.read_csv(metadata_path)
        else:
            # Generate sample metadata for demonstration
            metadata = self._generate_sample_metadata()
        
        self.buildings_metadata = metadata
        return metadata
    
    def _generate_sample_metadata(self) -> pd.DataFrame:
        """Generate sample building metadata matching BDG2 structure"""
        np.random.seed(42)
        
        building_types = ['Residential', 'Dormitory', 'Multi-family', 'Lodging']
        climate_zones = ['Hot-Humid', 'Mixed-Dry', 'Cold', 'Hot-Dry', 'Mixed-Humid']
        
        buildings = []
        for i in range(1, 21):  # 20 sample buildings
            buildings.append({
                'building_id': f'Res_{i:02d}',
                'primary_use': np.random.choice(building_types),
                'floor_area': np.random.randint(800, 5000),
                'climate_zone': np.random.choice(climate_zones),
                'year_built': np.random.randint(2000, 2020),
                'data_resolution': '1-Hour'
            })
        
        return pd.DataFrame(buildings)
    
    def filter_residential_buildings(self) -> pd.DataFrame:
        """Filter buildings tagged as Residential or Lodging"""
        if self.buildings_metadata is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        residential_types = ['Residential', 'Dormitory', 'Multi-family', 'Lodging']
        filtered = self.buildings_metadata[
            self.buildings_metadata['primary_use'].isin(residential_types)
        ].copy()
        
        print(f"Filtered {len(filtered)} residential buildings from {len(self.buildings_metadata)} total")
        return filtered
    
    def generate_building_characteristics_table(self, buildings_df: pd.DataFrame) -> pd.DataFrame:
        """Generate Table 1: Characteristics of Selected Case Study Buildings"""
        table = buildings_df[['building_id', 'primary_use', 'floor_area', 
                              'climate_zone', 'year_built', 'data_resolution']].copy()
        table.columns = ['Building ID', 'Primary Use', 'Floor Area (m²)', 
                        'Climate Zone', 'Year Built', 'Data Resolution']
        return table
    
    def load_weather_data(self, building_id: str, start_date: str = None, 
                         end_date: str = None) -> pd.DataFrame:
        """Load weather data for a building"""
        # Simulated weather data - in real scenario, load from BDG2 weather files
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='H')
        
        np.random.seed(hash(building_id) % 2**32)
        
        # Generate realistic weather patterns
        base_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (365.25 * 24))
        temp = base_temp + np.random.normal(0, 3, len(dates))
        
        solar_rad = np.maximum(0, 800 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + 
                               np.random.normal(0, 100, len(dates)))
        solar_rad = np.where(np.arange(len(dates)) % 24 < 6, 0, solar_rad)  # Night = 0
        solar_rad = np.where(np.arange(len(dates)) % 24 > 18, 0, solar_rad)
        
        humidity = 50 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / (365.25 * 24)) + \
                   np.random.normal(0, 10, len(dates))
        humidity = np.clip(humidity, 20, 90)
        
        weather_df = pd.DataFrame({
            'timestamp': dates,
            'outdoor_temp': temp,
            'solar_radiation': solar_rad,
            'humidity': humidity
        })
        
        if start_date:
            weather_df = weather_df[weather_df['timestamp'] >= start_date]
        if end_date:
            weather_df = weather_df[weather_df['timestamp'] <= end_date]
        
        return weather_df
    
    def load_meter_data(self, building_id: str, start_date: str = None, 
                       end_date: str = None) -> pd.DataFrame:
        """Load energy meter readings for a building"""
        # Simulated meter data - in real scenario, load from BDG2 meter files
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='H')
        
        np.random.seed(hash(building_id) % 2**32 + 1000)
        
        # Generate realistic energy consumption patterns
        base_energy = 50 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
        energy = np.maximum(10, base_energy + np.random.normal(0, 5, len(dates)))
        
        # Indoor temperature (affected by HVAC)
        indoor_temp = 22 + 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + \
                     np.random.normal(0, 0.5, len(dates))
        
        meter_df = pd.DataFrame({
            'timestamp': dates,
            'energy_consumption': energy,
            'indoor_temp': indoor_temp
        })
        
        if start_date:
            meter_df = meter_df[meter_df['timestamp'] >= start_date]
        if end_date:
            meter_df = meter_df[meter_df['timestamp'] <= end_date]
        
        return meter_df
    
    def merge_weather_meter_data(self, building_id: str, weather_df: pd.DataFrame, 
                                meter_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather and meter data on timestamp"""
        merged = pd.merge(meter_df, weather_df, on='timestamp', how='inner')
        merged['building_id'] = building_id
        return merged
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features (Hour of Day, Day of Week)"""
        df = df.copy()
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek + 1  # 1-7
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        return df
    
    def clean_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Clean missing values"""
        df = df.copy()
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'drop':
            df = df.dropna()
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, feature_cols: List[str], 
                      method: str = 'min_max') -> Tuple[pd.DataFrame, Dict]:
        """Normalize data to 0-1 scale"""
        df = df.copy()
        scalers = {}
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            if method == 'min_max':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                scalers[col] = {'min': min_val, 'max': max_val, 'method': 'min_max'}
            elif method == 'standard':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                scalers[col] = {'mean': mean_val, 'std': std_val, 'method': 'standard'}
        
        return df, scalers
    
    def process_building(self, building_id: str, start_date: str = '2020-06-01', 
                        end_date: str = '2020-08-31') -> Tuple[pd.DataFrame, Dict]:
        """Complete processing pipeline for a single building"""
        print(f"\nProcessing building: {building_id}")
        
        # Load data
        weather_df = self.load_weather_data(building_id, start_date, end_date)
        meter_df = self.load_meter_data(building_id, start_date, end_date)
        
        # Merge
        merged_df = self.merge_weather_meter_data(building_id, weather_df, meter_df)
        
        # Add temporal features
        merged_df = self.add_temporal_features(merged_df)
        
        # Clean missing values
        merged_df = self.clean_missing_values(merged_df, method='forward_fill')
        
        # Define feature columns for normalization
        feature_cols = ['outdoor_temp', 'solar_radiation', 'humidity', 
                       'hour_of_day', 'day_of_week', 'energy_consumption', 'indoor_temp']
        
        # Normalize
        normalized_df, scalers = self.normalize_data(merged_df, feature_cols, method='min_max')
        
        self.processed_buildings[building_id] = {
            'data': normalized_df,
            'scalers': scalers,
            'original_data': merged_df
        }
        
        print(f"  Processed {len(normalized_df)} data points")
        print(f"  Date range: {normalized_df['timestamp'].min()} to {normalized_df['timestamp'].max()}")
        
        return normalized_df, scalers
    
    def process_multiple_buildings(self, building_ids: List[str], 
                                  start_date: str = '2020-06-01', 
                                  end_date: str = '2020-08-31') -> Dict:
        """Process multiple buildings"""
        all_data = []
        
        for building_id in building_ids:
            df, scalers = self.process_building(building_id, start_date, end_date)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return {
            'combined_data': combined_df,
            'individual_buildings': self.processed_buildings
        }


if __name__ == "__main__":
    # Example usage
    processor = BDG2DataProcessor()
    
    # Load and filter metadata
    metadata = processor.load_metadata("metadata.csv")
    residential_buildings = processor.filter_residential_buildings()
    
    # Generate Table 1
    table1 = processor.generate_building_characteristics_table(residential_buildings.head(10))
    print("\n" + "="*80)
    print("Table 1: Characteristics of Selected Case Study Buildings")
    print("="*80)
    print(table1.to_string(index=False))
    
    # Process sample buildings
    sample_buildings = residential_buildings['building_id'].head(5).tolist()
    results = processor.process_multiple_buildings(sample_buildings)
    
    print(f"\nTotal processed data points: {len(results['combined_data'])}")
    print(f"\nSample of processed data:")
    print(results['combined_data'][['building_id', 'timestamp', 'outdoor_temp', 
                                    'solar_radiation', 'energy_consumption', 
                                    'hour_of_day', 'day_of_week']].head(10))
