import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PhDDataPipeline:
    def __init__(self, config):
        self.config = config
        self.data = None
        
    def load_entsoe_data(self) -> pd.DataFrame:
        """Load and process ENTSO-E data for multiple countries"""
        # This would connect to actual ENTSO-E API
        print("Loading ENTSO-E data...")
        
        # Simulate real data structure based on your notebook analysis
        dates = pd.date_range(
            start=self.config.START_DATE,
            end=self.config.END_DATE,
            freq='H'
        )
        
        # Create realistic multi-country energy data
        countries_data = {}
        for country in self.config.COUNTRIES:
            base_load = np.random.normal(50000, 10000, len(dates))
            seasonal = 10000 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 365))
            trend = 100 * np.arange(len(dates)) / (24 * 365)
            noise = np.random.normal(0, 1000, len(dates))
            
            countries_data[country] = base_load + seasonal + trend + noise
        
        energy_df = pd.DataFrame(countries_data, index=dates)
        energy_df['total_load_actual'] = energy_df.sum(axis=1)
        
        return energy_df
    
    def add_external_features(self, energy_df: pd.DataFrame) -> pd.DataFrame:
        """Add research-grade external features"""
        df = energy_df.copy()
        
        # Meteorological features
        df['temperature'] = 15 + 10 * np.sin(2 * np.pi * df.index.hour / 24) + \
                          5 * np.sin(2 * np.pi * df.index.dayofyear / 365) + \
                          np.random.normal(0, 3, len(df))
        
        # Economic indicators
        df['day_ahead_price'] = np.random.lognormal(3, 0.5, len(df))
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek >= 5
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        return df
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Create comprehensive lagged features"""
        if lags is None:
            lags = [1, 2, 3, 24, 48, 168]  # 1h, 2h, 3h, 1d, 2d, 1w
        
        target = self.config.TARGET_VARIABLE
        
        for lag in lags:
            df[f'{target}_lag_{lag}'] = df[target].shift(lag)
        
        # Rolling statistics
        windows = [24, 168, 720]  # 1d, 1w, 1m
        for window in windows:
            df[f'{target}_rolling_mean_{window}'] = df[target].rolling(window).mean()
            df[f'{target}_rolling_std_{window}'] = df[target].rolling(window).std()
        
        return df
    
    def make_dataset(self) -> pd.DataFrame:
        """Create complete research dataset"""
        energy_data = self.load_entsoe_data()
        enhanced_data = self.add_external_features(energy_data)
        final_data = self.create_lagged_features(enhanced_data)
        
        # Remove initial NaN values from lagging
        final_data = final_data.dropna()
        
        print(f"Final dataset shape: {final_data.shape}")
        return final_data

# Usage
if __name__ == "__main__":
    from config.research_config import config
    pipeline = PhDDataPipeline(config)
    dataset = pipeline.make_dataset()
    dataset.to_parquet('data/processed/research_dataset.parquet')
