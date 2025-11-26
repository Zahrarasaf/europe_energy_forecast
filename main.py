import pandas as pd
import numpy as np
import os
import requests

def download_from_github_release():
    """Download dataset from GitHub Releases"""
    data_path = "data/europe_energy.csv"
    
    if os.path.exists(data_path):
        print("✅ Dataset found locally")
        return pd.read_csv(data_path)
    
    print("📥 Downloading from GitHub Releases...")
    
  
    release_url = "https://drive.google.com/file/d/1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s/view?usp=sharing"
    
    os.makedirs("data", exist_ok=True)
    
    try:
        response = requests.get(release_url, stream=True)
        response.raise_for_status()
        
        with open(data_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print("✅ Dataset downloaded from GitHub Releases!")
        return pd.read_csv(data_path)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    print("🎯 European Energy Forecasting - PhD Project")
    print("=" * 50)
    
    df = download_from_github_release()
    
    if df is not None:
        print(f"✅ Success! Dataset shape: {df.shape}")
    else:
        print("❌ Please upload dataset to GitHub Releases")

if __name__ == "__main__":
    main()
