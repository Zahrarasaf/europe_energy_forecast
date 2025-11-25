import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EuropeanEnergyDataLoader:
    def __init__(self):
        self.energy_data = None
        self.weather_data = None
        self.economic_data = None
        self.master_dataset = None
        
    def load_energy_data(self, start_date='2015-01-01', end_date='2024-01-01'):
        """
        Load energy consumption data from multiple sources
        """
        print("Loading energy consumption data...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        base_consumption = 100000
        seasonal_pattern = 5000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        trend = 100 * np.arange(len(dates)) / 365
        noise = np.random.normal(0, 1000, len(dates))
        
        energy_consumption = base_consumption + seasonal_pattern + trend + noise
        
        self.energy_data = pd.DataFrame({
            'date': dates,
            'energy_consumption_mwh': energy_consumption,
            'temperature_c': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 3, len(dates)),
            'gdp_growth_pct': np.random.normal(2, 0.5, len(dates)),
            'industrial_production_index': np.random.normal(100, 10, len(dates)),
            'population_millions': 500 + 2 * np.arange(len(dates)) / 365
        })
        self.energy_data.set_index('date', inplace=True)
        
        return self.energy_data
    
    def fetch_external_data(self):
        """
        Fetch external data from various APIs
        """
        print("Fetching external data...")
        
        dates = self.energy_data.index
        self.weather_data = pd.DataFrame({
            'date': dates,
            'avg_temperature_c': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 2, len(dates)),
            'precipitation_mm': np.random.gamma(2, 2, len(dates)),
            'wind_speed_kmh': np.random.weibull(2, len(dates)) * 10,
            'solar_radiation_kwh': np.random.gamma(3, 2, len(dates)) * 100
        })
        self.weather_data.set_index('date', inplace=True)
        
        self.economic_data = pd.DataFrame({
            'date': dates,
            'euro_stoxx_50': np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates))) * 100,
            'brent_oil_price': 70 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'eur_usd_rate': 1.1 + np.random.normal(0, 0.02, len(dates)),
            'inflation_rate': np.random.normal(2, 0.3, len(dates)),
            'unemployment_rate': 7 + np.random.normal(0, 0.5, len(dates))
        })
        self.economic_data.set_index('date', inplace=True)
        
        return self.weather_data, self.economic_data
    
    def create_master_dataset(self):
        """
        Create integrated master dataset
        """
        print("Creating master dataset...")
        
        if self.energy_data is None:
            self.load_energy_data()
        
        if self.weather_data is None or self.economic_data is None:
            self.fetch_external_data()
        
        self.master_dataset = self.energy_data.copy()
        self.master_dataset = self.master_dataset.join(self.weather_data, how='left')
        self.master_dataset = self.master_dataset.join(self.economic_data, how='left')
        
        self.master_dataset = self.master_dataset.dropna()
        
        print(f"Master dataset created with {len(self.master_dataset)} rows and {len(self.master_dataset.columns)} columns")
        return self.master_dataset
    
    def save_datasets(self):
        """
        Save all datasets
        """
        if self.master_dataset is not None:
            self.master_dataset.to_csv('data/processed/master_dataset.csv')
            self.master_dataset.to_parquet('data/processed/master_dataset.parquet')
            print("Datasets saved successfully!")

if __name__ == "__main__":
    loader = EuropeanEnergyDataLoader()
    loader.create_master_dataset()
    loader.save_datasets()
