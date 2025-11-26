import pandas as pd
import numpy as np
import os
import sys
import requests

def download_dataset():
    """Download dataset if not exists"""
    data_path = "data/europe_energy.csv"
    
    if os.path.exists(data_path):
        print("✅ Dataset found")
        return pd.read_csv(data_path)
    
    print("📥 Downloading dataset...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    try:
        # Your Google Drive file ID
        file_id = "1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s"
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Download file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Save file
        with open(data_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print("✅ Dataset downloaded successfully!")
        return pd.read_csv(data_path)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    print("🎯 European Energy Forecasting - PhD Project")
    print("=" * 50)
    
    # Download or load dataset
    df = download_dataset()
    
    if df is None:
        print("\n🚨 Please download the dataset manually:")
        print("1. Go to: https://drive.google.com/file/d/1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s/view")
        print("2. Download the file")
        print("3. Save as 'data/europe_energy.csv'")
        return
    
    # Show data info
    print(f"✅ Dataset loaded: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Sample columns: {list(df.columns)[:10]}...")
    
    # Basic info
    if 'utc_timestamp' in df.columns:
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
        print(f"Date range: {df['utc_timestamp'].min()} to {df['utc_timestamp'].max()}")

if __name__ == "__main__":
    main()
