"""
Phase 1: Data Curation & Pre-processing
Filter BDG2 dataset for residential buildings and prepare data
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

class BDG2DataCurator:
    """Handles BDG2 dataset curation and preprocessing"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        self.buildings_info = []
        
    def generate_synthetic_bdg2_data(self):
        """
        Generate synthetic BDG2-like data for residential buildings
        This simulates the structure of the actual BDG2 dataset
        """
        print("Generating synthetic BDG2-like residential building data...")
        
        # Define building characteristics for diverse climate zones
        buildings = [
            {
                'building_id': 'Res_01',
                'primary_use': 'Residential',
                'floor_area_m2': 1200,
                'climate_zone': 'Hot-Humid',
                'year_built': 2010,
                'data_resolution': '1-Hour',
                'latitude': 29.76,  # Houston-like
                'longitude': -95.37
            },
            {
                'building_id': 'Res_02',
                'primary_use': 'Dormitory',
                'floor_area_m2': 3500,
                'climate_zone': 'Mixed-Dry',
                'year_built': 2005,
                'data_resolution': '1-Hour',
                'latitude': 39.74,  # Denver-like
                'longitude': -104.99
            },
            {
                'building_id': 'Res_03',
                'primary_use': 'Multi-family',
                'floor_area_m2': 2100,
                'climate_zone': 'Cold',
                'year_built': 2015,
                'data_resolution': '1-Hour',
                'latitude': 44.98,  # Minneapolis-like
                'longitude': -93.27
            },
            {
                'building_id': 'Res_04',
                'primary_use': 'Residential',
                'floor_area_m2': 1800,
                'climate_zone': 'Hot-Dry',
                'year_built': 2012,
                'data_resolution': '1-Hour',
                'latitude': 33.45,  # Phoenix-like
                'longitude': -112.07
            },
            {
                'building_id': 'Res_05',
                'primary_use': 'Lodging',
                'floor_area_m2': 4200,
                'climate_zone': 'Marine',
                'year_built': 2008,
                'data_resolution': '1-Hour',
                'latitude': 47.61,  # Seattle-like
                'longitude': -122.33
            }
        ]
        
        # Save building metadata
        buildings_df = pd.DataFrame(buildings)
        buildings_df.to_csv(f'{self.data_dir}/building_metadata.csv', index=False)
        self.buildings_info = buildings
        
        # Generate time series data for each building (1 year of hourly data)
        date_range = pd.date_range('2023-01-01', '2023-12-31 23:00:00', freq='h')
        
        for building in buildings:
            building_id = building['building_id']
            climate = building['climate_zone']
            floor_area = building['floor_area_m2']
            
            # Generate weather data based on climate zone
            weather_data = self._generate_weather_data(date_range, climate)
            
            # Generate energy consumption and indoor temperature
            energy_data = self._generate_energy_data(
                date_range, weather_data, floor_area, climate
            )
            
            # Combine all data
            full_data = pd.DataFrame({
                'timestamp': date_range,
                'building_id': building_id,
                'outdoor_temp': weather_data['outdoor_temp'],
                'solar_radiation': weather_data['solar_radiation'],
                'humidity': weather_data['humidity'],
                'wind_speed': weather_data['wind_speed'],
                'energy_consumption': energy_data['energy_consumption'],
                'indoor_temp': energy_data['indoor_temp'],
                'hvac_setpoint': energy_data['hvac_setpoint'],
                'hour_of_day': date_range.hour,
                'day_of_week': date_range.dayofweek,
                'month': date_range.month,
                'is_weekend': (date_range.dayofweek >= 5).astype(int)
            })
            
            # Save building data
            full_data.to_csv(f'{self.data_dir}/{building_id}_data.csv', index=False)
            print(f"Generated data for {building_id}")
        
        print(f"Generated data for {len(buildings)} residential buildings")
        return buildings_df
    
    def _generate_weather_data(self, date_range, climate_zone):
        """Generate realistic weather data based on climate zone"""
        n_points = len(date_range)
        hour_of_day = np.array(date_range.hour)
        day_of_year = np.array(date_range.dayofyear)
        
        # Base temperature patterns by climate zone
        climate_params = {
            'Hot-Humid': {'base_temp': 27, 'summer_boost': 8, 'winter_drop': 10, 'humidity_base': 70},
            'Mixed-Dry': {'base_temp': 18, 'summer_boost': 15, 'winter_drop': 18, 'humidity_base': 40},
            'Cold': {'base_temp': 10, 'summer_boost': 18, 'winter_drop': 25, 'humidity_base': 50},
            'Hot-Dry': {'base_temp': 30, 'summer_boost': 10, 'winter_drop': 12, 'humidity_base': 25},
            'Marine': {'base_temp': 15, 'summer_boost': 10, 'winter_drop': 8, 'humidity_base': 65}
        }
        
        params = climate_params.get(climate_zone, climate_params['Mixed-Dry'])
        
        # Outdoor temperature with seasonal and daily variation
        seasonal_variation = np.sin((day_of_year - 80) / 365 * 2 * np.pi)
        daily_variation = -4 * np.cos(hour_of_day / 24 * 2 * np.pi)
        outdoor_temp = (params['base_temp'] + 
                       seasonal_variation * params['summer_boost']/2 -
                       (1-seasonal_variation) * params['winter_drop']/2 +
                       daily_variation +
                       np.random.normal(0, 2, n_points))
        
        # Solar radiation (0 at night, peak at noon)
        solar_radiation = np.maximum(0, 
            800 * np.sin((hour_of_day - 6) / 12 * np.pi) * 
            (1 + 0.3 * seasonal_variation) +
            np.random.normal(0, 50, n_points)
        )
        solar_radiation[hour_of_day < 6] = 0
        solar_radiation[hour_of_day > 20] = 0
        
        # Humidity
        humidity = (params['humidity_base'] + 
                   np.random.normal(0, 10, n_points) +
                   15 * np.sin(hour_of_day / 24 * 2 * np.pi))
        humidity = np.clip(humidity, 20, 95)
        
        # Wind speed
        wind_speed = np.abs(np.random.normal(3, 2, n_points))
        
        return {
            'outdoor_temp': outdoor_temp,
            'solar_radiation': solar_radiation,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
    
    def _generate_energy_data(self, date_range, weather_data, floor_area, climate_zone):
        """Generate energy consumption and indoor temperature data"""
        n_points = len(date_range)
        hour_of_day = np.array(date_range.hour)
        is_weekend = np.array((date_range.dayofweek >= 5).astype(int))
        
        # HVAC setpoint schedule (realistic residential behavior)
        hvac_setpoint = np.ones(n_points) * 22.0  # Base setpoint
        
        # Lower setpoint at night (sleep hours)
        night_hours = (hour_of_day >= 22) | (hour_of_day <= 6)
        hvac_setpoint[night_hours] = 20.0
        
        # Higher setpoint during work hours on weekdays
        work_hours = (hour_of_day >= 8) & (hour_of_day <= 17) & (is_weekend == 0)
        hvac_setpoint[work_hours] = 24.0
        
        # Add some random variation
        hvac_setpoint += np.random.normal(0, 1, n_points)
        hvac_setpoint = np.clip(hvac_setpoint, 19, 26)
        
        # Indoor temperature (influenced by outdoor temp, setpoint, and thermal inertia)
        indoor_temp = np.zeros(n_points)
        indoor_temp[0] = hvac_setpoint[0]
        
        thermal_mass = floor_area / 100  # Larger buildings have more thermal mass
        
        for i in range(1, n_points):
            # Thermal dynamics
            outdoor_influence = 0.05 * (weather_data['outdoor_temp'][i] - indoor_temp[i-1])
            solar_influence = 0.0001 * weather_data['solar_radiation'][i]
            hvac_influence = 0.3 * (hvac_setpoint[i] - indoor_temp[i-1])
            
            indoor_temp[i] = (indoor_temp[i-1] + outdoor_influence + 
                             solar_influence + hvac_influence +
                             np.random.normal(0, 0.2))
        
        # Energy consumption (kWh)
        temp_difference = np.abs(indoor_temp - weather_data['outdoor_temp'])
        base_load = floor_area * 0.005  # Base load (lighting, appliances)
        
        hvac_load = (floor_area * 0.02 * temp_difference / 10 * 
                    (1 + weather_data['solar_radiation'] / 2000))
        
        # Higher consumption during occupied hours
        occupied_multiplier = np.ones(n_points)
        occupied_hours = (hour_of_day >= 6) & (hour_of_day <= 23)
        occupied_multiplier[occupied_hours] = 1.3
        
        energy_consumption = (base_load + hvac_load) * occupied_multiplier
        energy_consumption += np.random.normal(0, base_load * 0.1, n_points)
        energy_consumption = np.maximum(energy_consumption, base_load * 0.5)
        
        return {
            'energy_consumption': energy_consumption,
            'indoor_temp': indoor_temp,
            'hvac_setpoint': hvac_setpoint
        }
    
    def load_and_merge_data(self):
        """Load all building data and merge into a single dataset"""
        print("\nLoading and merging building data...")
        
        all_data = []
        metadata = pd.read_csv(f'{self.data_dir}/building_metadata.csv')
        
        for _, building in metadata.iterrows():
            building_id = building['building_id']
            data = pd.read_csv(f'{self.data_dir}/{building_id}_data.csv')
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            all_data.append(data)
        
        merged_data = pd.concat(all_data, ignore_index=True)
        print(f"Merged data shape: {merged_data.shape}")
        
        return merged_data, metadata
    
    def clean_and_normalize(self, data):
        """Clean missing values and normalize features"""
        print("\nCleaning and normalizing data...")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        print(f"Missing values:\n{missing_counts[missing_counts > 0]}")
        
        # Forward fill missing values (common for time series)
        data = data.ffill().bfill()
        
        # Select features for normalization
        features_to_normalize = [
            'outdoor_temp', 'solar_radiation', 'humidity', 'wind_speed',
            'energy_consumption', 'indoor_temp', 'hvac_setpoint'
        ]
        
        # Normalize to 0-1 range
        normalized_data = data.copy()
        normalized_features = self.scaler.fit_transform(data[features_to_normalize])
        
        for i, col in enumerate(features_to_normalize):
            normalized_data[f'{col}_normalized'] = normalized_features[:, i]
        
        # Save scaler for later use
        with open(f'{self.data_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Data normalized and scaler saved")
        
        return normalized_data
    
    def generate_table1(self, metadata):
        """Generate Table 1: Characteristics of Selected Case Study Buildings"""
        print("\nGenerating Table 1: Building Characteristics...")
        
        table1 = metadata.copy()
        table1 = table1[[
            'building_id', 'primary_use', 'floor_area_m2', 
            'climate_zone', 'year_built', 'data_resolution'
        ]]
        table1.columns = [
            'Building ID', 'Primary Use', 'Floor Area (m²)', 
            'Climate Zone', 'Year Built', 'Data Resolution'
        ]
        
        # Save table
        table1.to_csv('tables/table1_building_characteristics.csv', index=False)
        
        # Also save as formatted text
        with open('tables/table1_building_characteristics.txt', 'w') as f:
            f.write("Table 1: Characteristics of Selected Case Study Buildings\n")
            f.write("="*80 + "\n\n")
            f.write(table1.to_string(index=False))
        
        print("Table 1 saved to tables/")
        return table1


def main():
    """Execute Phase 1: Data Curation & Pre-processing"""
    print("="*80)
    print("PHASE 1: DATA CURATION & PRE-PROCESSING")
    print("="*80)
    
    # Initialize curator
    curator = BDG2DataCurator()
    
    # Step 1: Generate/Filter residential building data
    print("\nStep 1: Filtering residential buildings from BDG2 dataset...")
    metadata = curator.generate_synthetic_bdg2_data()
    
    # Step 2: Merge meter readings with weather data
    print("\nStep 2: Merging meter readings with weather data...")
    merged_data, metadata = curator.load_and_merge_data()
    
    # Step 3: Clean and normalize
    print("\nStep 3: Cleaning and normalizing data...")
    normalized_data = curator.clean_and_normalize(merged_data)
    
    # Save processed data
    normalized_data.to_csv('data/processed_data.csv', index=False)
    print("\nProcessed data saved to data/processed_data.csv")
    
    # Generate Table 1
    table1 = curator.generate_table1(metadata)
    print("\n" + "="*80)
    print(table1)
    print("="*80)
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Total records: {len(normalized_data):,}")
    print(f"Number of buildings: {len(metadata)}")
    print(f"Date range: {normalized_data['timestamp'].min()} to {normalized_data['timestamp'].max()}")
    print(f"Average energy consumption: {normalized_data['energy_consumption'].mean():.2f} kWh")
    print(f"Average indoor temperature: {normalized_data['indoor_temp'].mean():.2f}°C")
    
    print("\n✓ Phase 1 completed successfully!")


if __name__ == "__main__":
    main()
