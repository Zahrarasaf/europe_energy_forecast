import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Create sample energy data for testing"""
    print("Creating sample energy dataset...")
    
    # Create 2 years of hourly data
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='H')
    
    np.random.seed(42)
    
    # Realistic energy consumption patterns
    data = {
        'timestamp': dates,
        'germany_energy': 50000 + 8000 * np.sin(2 * np.pi * dates.dayofyear/365) + np.random.normal(0, 1500, len(dates)),
        'france_energy': 45000 + 6000 * np.sin(2 * np.pi * dates.dayofyear/365) + np.random.normal(0, 1200, len(dates)),
        'temperature': 15 + 10 * np.sin(2 * np.pi * dates.hour/24) + np.random.normal(0, 3, len(dates)),
        'day_of_week': dates.dayofweek,
        'hour': dates.hour
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/energy_data.csv', index=False)
    
    print(f"✅ Sample data created: {len(df)} records")
    return df

def load_data():
    """Load data - create if doesn't exist"""
    if os.path.exists('data/energy_data.csv'):
        df = pd.read_csv('data/energy_data.csv')
        print(f"✅ Data loaded: {len(df)} records")
    else:
        print("📊 Creating new sample data...")
        df = create_sample_data()
    
    return df
