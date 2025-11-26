import pandas as pd
import numpy as np
import requests
import os
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def download_dataset():
    """Download dataset from Google Drive"""
    data_path = "data/europe_energy.csv"
    
    if os.path.exists(data_path):
        print("✅ Dataset found locally")
        return pd.read_csv(data_path)
    
    print("📥 Downloading dataset from Google Drive...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    try:
        # Your Google Drive direct link
        url = "https://drive.google.com/uc?export=download&id=1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s"
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(data_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}%", end='\r')
        
        print("\n✅ Dataset downloaded successfully!")
        
        # Verify the file is not empty
        file_size = os.path.getsize(data_path)
        if file_size == 0:
            print("❌ Downloaded file is empty!")
            os.remove(data_path)
            return None
            
        print(f"📦 File size: {file_size / 1024 / 1024:.2f} MB")
        return pd.read_csv(data_path)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def calculate_real_improvement(df):
    """Calculate real improvement percentage from your data"""
    
    print("🔍 Analyzing your dataset...")
    
    # Show basic info about the dataset
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Sample columns: {list(df.columns)[:5]}...")
    
    # Find target column (Germany load)
    target_col = None
    target_candidates = [
        'DE_load_actual_entsoe_transparency',
        'DE_load_actual',
        'Germany_load',
        'load_DE'
    ]
    
    # Also search for any column with DE and load
    for col in df.columns:
        if 'DE' in col and 'load' in col.lower():
            target_col = col
            break
    
    if target_col is None and len(df.columns) > 0:
        # Use first numeric column as target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
            print(f"⚠️  Using first numeric column as target: {target_col}")
    
    if target_col is None:
        print("❌ No suitable target column found")
        return None
    
    print(f"🎯 Target column: {target_col}")
    
    # Prepare data
    df = df.copy()
    
    # Handle timestamp
    timestamp_col = None
    for col in ['utc_timestamp', 'timestamp', 'DateTime', 'date', 'Time']:
        if col in df.columns:
            timestamp_col = col
            try:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                print(f"📅 Time column: {timestamp_col}")
                print(f"   Date range: {df.index.min()} to {df.index.max()}")
                break
            except:
                continue
    
    # Remove rows with missing target
    initial_size = len(df)
    df = df[df[target_col].notna()]
    print(f"📈 Records after cleaning: {len(df)}/{initial_size}")
    
    if len(df) < 100:
        print("❌ Not enough data after cleaning")
        return None
    
    # 1. BASELINE MODEL (Simple approach)
    df_sorted = df.sort_index() if timestamp_col else df
    
    # Use previous value as baseline
    baseline_predictions = df_sorted[target_col].shift(1)
    
    # Remove NaN for baseline comparison
    valid_mask = baseline_predictions.notna() & df_sorted[target_col].notna()
    y_true_baseline = df_sorted[target_col][valid_mask]
    y_pred_baseline = baseline_predictions[valid_mask]
    
    if len(y_true_baseline) == 0:
        print("❌ Not enough data for baseline")
        return None
    
    baseline_mae = mean_absolute_error(y_true_baseline, y_pred_baseline)
    baseline_rmse = np.sqrt(np.mean((y_true_baseline - y_pred_baseline) ** 2))
    
    print(f"📊 Baseline Performance:")
    print(f"   MAE: {baseline_mae:.2f}")
    print(f"   RMSE: {baseline_rmse:.2f}")
    print(f"   Samples: {len(y_true_baseline):,}")
    
    # 2. ADVANCED MODEL
    print("\n🤖 Building advanced model...")
    
    features = []
    
    # Create basic features
    if timestamp_col:
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        features.extend(['hour', 'day_of_week', 'month'])
    
    # Lag features
    for lag in [1, 2, 3, 24]:
        lag_col = f'lag_{lag}'
        df[lag_col] = df[target_col].shift(lag)
        features.append(lag_col)
    
    # Add other numeric columns as features (limited to 10 to avoid overfitting)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    other_features = [col for col in numeric_cols if col != target_col and col not in features]
    features.extend(other_features[:10])  # Use first 10 additional features
    
    print(f"🔧 Using {len(features)} features")
    
    # Prepare data for modeling
    df_clean = df[features + [target_col]].dropna()
    
    if len(df_clean) < 50:
        print("❌ Not enough data for modeling")
        return None
    
    X = df_clean[features]
    y = df_clean[target_col]
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📚 Training samples: {len(X_train):,}")
    print(f"🧪 Test samples: {len(X_test):,}")
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance
    advanced_mae = mean_absolute_error(y_test, y_pred)
    advanced_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    print(f"🚀 Advanced Model Performance:")
    print(f"   MAE: {advanced_mae:.2f}")
    print(f"   RMSE: {advanced_rmse:.2f}")
    
    # 3. CALCULATE IMPROVEMENT
    improvement_mae = ((baseline_mae - advanced_mae) / baseline_mae) * 100
    improvement_rmse = ((baseline_rmse - advanced_rmse) / baseline_rmse) * 100
    
    print(f"\n🎯 PERFORMANCE IMPROVEMENT:")
    print(f"   MAE Improvement: {improvement_mae:+.1f}%")
    print(f"   RMSE Improvement: {improvement_rmse:+.1f}%")
    
    avg_improvement = (improvement_mae + improvement_rmse) / 2
    
    return avg_improvement

def main():
    print("🎯 European Energy Forecasting - Real Performance Calculation")
    print("=" * 65)
    print("📁 Using your dataset from Google Drive")
    print("=" * 65)
    
    # Download and load data
    df = download_dataset()
    
    if df is None:
        print("\n💡 Please download manually and place in data/ folder:")
        print("1. Go to: https://drive.google.com/file/d/1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s/view")
        print("2. Click 'Download'")
        print("3. Save as 'data/europe_energy.csv'")
        print("4. Run this script again")
        return
    
    # Calculate real improvement
    improvement = calculate_real_improvement(df)
    
    if improvement is not None:
        print(f"\n" + "=" * 50)
        print(f"✅ FINAL RESULT FOR YOUR CV:")
        print(f"   Achieved {improvement:+.1f}% improvement in forecasting accuracy")
        print(f"   Based on YOUR actual dataset")
        print(f"   Using machine learning vs simple baseline")
        print("=" * 50)
    else:
        print("\n❌ Could not calculate improvement from your data")

if __name__ == "__main__":
    main()
