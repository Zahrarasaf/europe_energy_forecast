import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class RealImprovementCalculator:
    def __init__(self):
        self.baseline_mae = None
        self.advanced_mae = None
        self.improvement = None
    
    def calculate_real_improvement(self, df, target_col='DE_load_actual_entsoe_transparency'):
        """Calculate REAL improvement from YOUR data"""
        
        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found")
            print(f"Available columns: {[col for col in df.columns if 'load' in col.lower()][:10]}")
            return None
        
        # Clean data
        df_clean = df[[target_col]].dropna()
        
        if len(df_clean) < 100:
            print("❌ Not enough data for calculation")
            return None
        
        # 1. Calculate BASELINE (naive forecast)
        baseline_predictions = df_clean[target_col].shift(24)  # 24 hours ago
        valid_baseline = baseline_predictions.notna() & df_clean[target_col].notna()
        
        if valid_baseline.sum() < 50:
            print("❌ Not enough data for baseline")
            return None
            
        y_true_baseline = df_clean[target_col][valid_baseline]
        y_pred_baseline = baseline_predictions[valid_baseline]
        
        self.baseline_mae = mean_absolute_error(y_true_baseline, y_pred_baseline)
        print(f"📊 Baseline MAE: {self.baseline_mae:.2f}")
        
        # 2. Calculate ADVANCED model performance
        advanced_mae = self._calculate_advanced_model(df, target_col)
        
        if advanced_mae is None:
            return None
            
        self.advanced_mae = advanced_mae
        print(f"🚀 Advanced Model MAE: {self.advanced_mae:.2f}")
        
        # 3. Calculate REAL improvement
        self.improvement = ((self.baseline_mae - self.advanced_mae) / self.baseline_mae) * 100
        print(f"🎯 REAL Improvement: {self.improvement:+.1f}%")
        
        return self.improvement
    
    def _calculate_advanced_model(self, df, target_col):
        """Calculate advanced model performance"""
        try:
            # Prepare features
            features = []
            df_temp = df.copy()
            
            # Add lag features
            for lag in [1, 2, 24, 25, 48]:
                lag_col = f'lag_{lag}'
                df_temp[lag_col] = df_temp[target_col].shift(lag)
                features.append(lag_col)
            
            # Add time features if timestamp available
            timestamp_cols = ['utc_timestamp', 'timestamp', 'DateTime']
            for ts_col in timestamp_cols:
                if ts_col in df_temp.columns:
                    df_temp[ts_col] = pd.to_datetime(df_temp[ts_col])
                    df_temp['hour'] = df_temp[ts_col].dt.hour
                    df_temp['day_of_week'] = df_temp[ts_col].dt.dayofweek
                    features.extend(['hour', 'day_of_week'])
                    break
            
            # Clean data for modeling
            modeling_data = df_temp[features + [target_col]].dropna()
            
            if len(modeling_data) < 100:
                return None
            
            X = modeling_data[features]
            y = modeling_data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict and calculate MAE
            y_pred = model.predict(X_test)
            advanced_mae = mean_absolute_error(y_test, y_pred)
            
            return advanced_mae
            
        except Exception as e:
            print(f"❌ Error in advanced model: {e}")
            return None
    
    def get_detailed_results(self):
        """Get detailed results for reporting"""
        return {
            'baseline_mae': self.baseline_mae,
            'advanced_mae': self.advanced_mae,
            'improvement_percentage': self.improvement,
            'is_real_calculation': True
        }
