import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def prepare_features(df, target_col='DE_load_actual_entsoe_transparency'):
    """Prepare features for advanced model"""
    
    features = []
    
    # Lag features (past values)
    for lag in [1, 2, 3, 24, 48]:  # 1h, 2h, 3h, 1d, 2d ago
        features.append(f'lag_{lag}')
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Final feature selection
    feature_cols = [f'lag_{lag}' for lag in [1, 2, 3, 24, 48]] + ['hour_sin', 'hour_cos']
    
    return df, feature_cols

def train_advanced_model(df, target_col='DE_load_actual_entsoe_transparency'):
    """Train advanced model"""
    
    # Prepare features
    df, feature_cols = prepare_features(df, target_col)
    
    # Remove NaN values
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        print("No data after cleaning!")
        return None
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"Advanced Model Performance:")
    print(f"   MAE: {mae:.2f} MW")
    print(f"   RMSE: {rmse:.2f} MW")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape}
