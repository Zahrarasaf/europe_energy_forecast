import pandas as pd
import numpy as np
import os
import sys

print("🎯 European Energy Forecasting - PhD Project")
print("=" * 50)

try:
    # Try to import from config
    from config.research_config import *
    print("✅ Config loaded from research_config.py")
except:
    # Fallback configuration
    DATA_PATH = "data/europe_energy.csv"
    COUNTRIES = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']
    TARGET_COUNTRY = 'DE'
    print("✅ Using fallback config")

def main():
    # 1. Check data
    if not os.path.exists(DATA_PATH):
        print("❌ Data file not found!")
        return
    
    print("✅ Data file found")
    
    # 2. Load data
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Convert DateTime if exists
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            print(f"📅 Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 3. Basic analysis
    print(f"\n📊 Basic Analysis for {TARGET_COUNTRY}:")
    if TARGET_COUNTRY in df.columns:
        target_data = df[TARGET_COUNTRY]
        print(f"   Records: {len(target_data):,}")
        print(f"   Mean: {target_data.mean():.2f}")
        print(f"   Std:  {target_data.std():.2f}")
        print(f"   Min:  {target_data.min():.2f}")
        print(f"   Max:  {target_data.max():.2f}")
    else:
        print(f"❌ Target country {TARGET_COUNTRY} not found in data")
    
    print("\n🎉 Project setup is correct!")
    print("Next: Add machine learning models and advanced analysis")

if __name__ == "__main__":
    main()
