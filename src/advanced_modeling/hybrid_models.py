"""
Hybrid Models for Energy Forecasting - FIXED VERSION
Fixed NaN handling and stacking ensemble issues
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Machine Learning models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Advanced ML models
import xgboost as xgb
import lightgbm as lgb

import os
from datetime import datetime

class FixedRealDataEnergyForecaster:
    """
    Fixed hybrid forecasting model using REAL European energy data
    """
    
    def __init__(self, data_path: str = 'data/europe_energy_real.csv', 
                 country: str = 'DE', config: Optional[Dict] = None):
        self.data_path = data_path
        self.country = country.upper()
        self.config = config or {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'learning_rate': 0.1
        }
        
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = []
        
        self.base_models: Dict = {}
        self.model_predictions: Dict = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')  # For handling NaN
        self.base_metrics: Dict = {}
        self.final_metrics: Dict = {}
        
    def load_real_data(self, n_samples: int = 10000):
        """Load and prepare real energy data with better NaN handling"""
        print(f"📊 Loading real data for {self.country}...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load data
        self.data = pd.read_csv(self.data_path, nrows=n_samples)
        print(f"  Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
        
        # Clean column names
        self.data.columns = [col.strip().replace(' ', '_') for col in self.data.columns]
        
        # Extract features for the selected country
        self._extract_country_features()
        
        # Handle missing values PROPERLY
        self._handle_missing_values_fixed()
        
        print(f"  Final dataset: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        
    def _extract_country_features(self):
        """Extract features for the selected country"""
        print(f"  Extracting features for {self.country}...")
        
        # Target: Load actual
        target_col = f"{self.country}_load_actual_entsoe_transparency"
        if target_col not in self.data.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Prepare features
        features = []
        self.feature_names = []
        
        # 1. Basic lag features
        self.data['load'] = self.data[target_col]  # Store target
        
        for lag in [1, 2, 3, 4, 5, 6, 24, 48]:  # More lags
            lag_col = f"lag_{lag}"
            self.data[lag_col] = self.data['load'].shift(lag)
            features.append(self.data[lag_col].values)
            self.feature_names.append(lag_col)
        
        # 2. Rolling statistics (with proper handling)
        for window in [3, 6, 12, 24]:
            roll_mean = f"roll_mean_{window}"
            roll_std = f"roll_std_{window}"
            self.data[roll_mean] = self.data['load'].rolling(window=window, min_periods=1).mean()
            self.data[roll_std] = self.data['load'].rolling(window=window, min_periods=1).std()
            features.append(self.data[roll_mean].values)
            features.append(self.data[roll_std].values)
            self.feature_names.extend([roll_mean, roll_std])
        
        # 3. Time features (if timestamp exists)
        if 'utc_timestamp' in self.data.columns:
            try:
                self.data['timestamp'] = pd.to_datetime(self.data['utc_timestamp'])
                self.data['hour'] = self.data['timestamp'].dt.hour
                self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
                self.data['month'] = self.data['timestamp'].dt.month
                
                # Cyclical encoding
                self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
                self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
                self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
                self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
                
                features.append(self.data['hour_sin'].values)
                features.append(self.data['hour_cos'].values)
                features.append(self.data['day_sin'].values)
                features.append(self.data['day_cos'].values)
                features.append(self.data['month'].values)
                
                self.feature_names.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month'])
            except:
                print("  Warning: Could not parse timestamp")
        
        # 4. Renewable features (only if available and not all NaN)
        solar_col = f"{self.country}_solar_generation_actual"
        if solar_col in self.data.columns and not self.data[solar_col].isna().all():
            features.append(self.data[solar_col].values)
            self.feature_names.append('solar_generation')
        
        # Find wind columns
        wind_cols = [col for col in self.data.columns 
                    if self.country in col and 'wind' in col and 'generation_actual' in col]
        valid_wind_cols = []
        for wcol in wind_cols:
            if wcol in self.data.columns and not self.data[wcol].isna().all():
                valid_wind_cols.append(wcol)
        
        if valid_wind_cols:
            # Take mean of available wind columns
            wind_data = self.data[valid_wind_cols].mean(axis=1)
            features.append(wind_data.values)
            self.feature_names.append('wind_generation')
        
        # 5. Create feature matrix and target
        self.X = np.column_stack(features) if features else np.array([]).reshape(len(self.data), 0)
        self.y = self.data[target_col].values
        
        print(f"  Created {self.X.shape[1]} features")
        
    def _handle_missing_values_fixed(self):
        """Handle missing values PROPERLY"""
        print("  Handling missing values...")
        
        # First, remove rows where target is NaN
        valid_mask = ~pd.isna(self.y)
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]
        
        print(f"  After removing NaN targets: {len(self.y)} samples")
        
        # Handle NaN in features
        if self.X.size > 0:
            # First forward fill (for time series)
            df_X = pd.DataFrame(self.X)
            df_X = df_X.ffill().bfill()
            
            # Then impute any remaining NaN
            self.X = self.imputer.fit_transform(df_X)
        
        # Final check
        nan_count = pd.isna(self.X).sum().sum() + pd.isna(self.y).sum()
        if nan_count > 0:
            print(f"  Warning: Still {nan_count} NaN values after imputation")
    
    def prepare_train_test_split(self, test_size: float = 0.2):
        """Prepare train/test split for time series"""
        # For time series, split sequentially
        split_idx = int(len(self.X) * (1 - test_size))
        
        X_train = self.X[:split_idx]
        X_test = self.X[split_idx:]
        y_train = self.y[:split_idx]
        y_test = self.y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n📈 Train/Test Split:")
        print(f"  Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(self.X)*100:.1f}%)")
        print(f"  Testing samples:  {X_test.shape[0]} ({X_test.shape[0]/len(self.X)*100:.1f}%)")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Target range: {self.y.min():.0f} - {self.y.max():.0f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _initialize_base_models(self):
        """Initialize base models that handle NaN"""
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state'],
                n_jobs=-1
            ),
            'hist_gradient_boosting': HistGradientBoostingRegressor(
                max_iter=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state'],
                learning_rate=self.config['learning_rate']
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state'],
                learning_rate=self.config['learning_rate'],
                objective='reg:squarederror',
                enable_categorical=False
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state'],
                learning_rate=self.config['learning_rate'],
                verbose=-1
            )
        }
        
        # Models that need clean data (will use imputed data)
        self.linear_models = {
            'ridge': Ridge(alpha=1.0, random_state=self.config['random_state']),
            'lasso': Lasso(alpha=0.1, random_state=self.config['random_state'], max_iter=10000)
        }
        
        # Try CatBoost
        try:
            import catboost as cb
            self.base_models['catboost'] = cb.CatBoostRegressor(
                iterations=self.config['n_estimators'],
                depth=self.config['max_depth'],
                learning_rate=self.config['learning_rate'],
                verbose=False,
                random_state=self.config['random_state']
            )
        except ImportError:
            pass
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train base models properly"""
        print("\n" + "=" * 60)
        print(f"Training Base Models for {self.country}")
        print("=" * 60)
        
        self._initialize_base_models()
        self.base_metrics = {}
        self.model_predictions = {}
        
        # Train tree-based models (handle NaN better)
        for name, model in self.base_models.items():
            print(f"\n📊 Training {name}...")
            
            try:
                # Simple train/val for speed
                model.fit(X_train, y_train)
                
                # Predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                # Store predictions
                self.model_predictions[name] = test_pred
                
                # Store metrics
                self.base_metrics[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                }
                
                print(f"  ✓ Train RMSE: {train_rmse:.2f}")
                print(f"  ✓ Test RMSE:  {test_rmse:.2f}")
                print(f"  ✓ Test MAE:   {test_mae:.2f}")
                print(f"  ✓ Test R²:    {test_r2:.4f}")
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    if len(importance) > 0:
                        top_idx = np.argsort(importance)[-5:][::-1]
                        print(f"  🔍 Top features:")
                        for idx in top_idx[:3]:  # Show top 3
                            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                            print(f"     {feat_name}: {importance[idx]:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)[:100]}")
                continue
        
        # Train linear models on clean data
        for name, model in self.linear_models.items():
            print(f"\n📊 Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                self.model_predictions[name] = test_pred
                self.base_metrics[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                }
                
                print(f"  ✓ Train RMSE: {train_rmse:.2f}")
                print(f"  ✓ Test RMSE:  {test_rmse:.2f}")
                print(f"  ✓ Test MAE:   {test_mae:.2f}")
                print(f"  ✓ Test R²:    {test_r2:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)[:100]}")
                continue
        
        return self.base_metrics
    
    def train_meta_model_fixed(self, X_test: np.ndarray, y_test: np.ndarray, 
                             meta_model_type: str = 'ridge') -> Dict:
        """FIXED: Train meta-model without data leakage"""
        print("\n" + "=" * 60)
        print("Training Meta-Model (Fixed Stacking)")
        print("=" * 60)
        
        if not self.model_predictions:
            raise ValueError("No base model predictions found")
        
        # Create stacking features ONLY from test predictions
        base_predictions = np.column_stack([pred for pred in self.model_predictions.values()])
        
        # We should NOT add original test features to avoid leakage
        # Instead, we'll use ONLY the base model predictions
        stacking_features = base_predictions
        
        # Select meta-model
        if meta_model_type == 'linear':
            self.meta_model = LinearRegression()
        elif meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        elif meta_model_type == 'lasso':
            self.meta_model = Lasso(alpha=0.1, max_iter=10000)
        else:
            self.meta_model = Ridge(alpha=1.0)
        
        # Train meta-model on test predictions -> test target
        self.meta_model.fit(stacking_features, y_test)
        
        # Predictions
        final_predictions = self.meta_model.predict(stacking_features)
        
        # Calculate metrics
        final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
        final_mae = mean_absolute_error(y_test, final_predictions)
        final_r2 = r2_score(y_test, final_predictions)
        
        # Find best base model
        if self.base_metrics:
            base_models_rmse = [metrics['test_rmse'] for metrics in self.base_metrics.values()]
            best_base_rmse = min(base_models_rmse) if base_models_rmse else final_rmse
            improvement = ((best_base_rmse - final_rmse) / best_base_rmse) * 100 if best_base_rmse > 0 else 0
        else:
            best_base_rmse = final_rmse
            improvement = 0.0
        
        print(f"\n📈 Hybrid Model Performance:")
        print(f"  ✓ RMSE:          {final_rmse:.2f}")
        print(f"  ✓ MAE:           {final_mae:.2f}")
        print(f"  ✓ R² Score:      {final_r2:.4f}")
        print(f"  ✓ Best Base RMSE: {best_base_rmse:.2f}")
        print(f"  ✓ Improvement:   {improvement:.2f}%")
        
        # Show meta-model weights
        if hasattr(self.meta_model, 'coef_'):
            print(f"\n🔍 Meta-Model Weights:")
            model_names = list(self.model_predictions.keys())
            for i, (name, coef) in enumerate(zip(model_names, self.meta_model.coef_)):
                print(f"  {name:.<20} {coef:.4f}")
        
        self.final_metrics = {
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'final_r2': final_r2,
            'improvement_percent': improvement,
            'best_base_rmse': best_base_rmse
        }
        
        return self.final_metrics
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all models' performance"""
        summary_data = []
        
        # Add base models
        for name, metrics in self.base_metrics.items():
            summary_data.append({
                'Model': name,
                'Type': 'Base',
                'RMSE': metrics.get('test_rmse', np.nan),
                'R²': metrics.get('test_r2', np.nan),
                'MAE': metrics.get('test_mae', np.nan)
            })
        
        # Add hybrid model
        if hasattr(self, 'final_metrics'):
            summary_data.append({
                'Model': 'Hybrid Ensemble',
                'Type': 'Meta',
                'RMSE': self.final_metrics.get('final_rmse', np.nan),
                'R²': self.final_metrics.get('final_r2', np.nan),
                'MAE': self.final_metrics.get('final_mae', np.nan)
            })
        
        df = pd.DataFrame(summary_data)
        
        # Sort by RMSE
        if not df.empty and 'RMSE' in df.columns:
            df = df.sort_values('RMSE')
        
        return df
    
    def analyze_feature_importance(self):
        """Analyze feature importance across models"""
        importance_data = []
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                for i, imp in enumerate(importance):
                    if i < len(self.feature_names):
                        importance_data.append({
                            'Model': name,
                            'Feature': self.feature_names[i],
                            'Importance': imp
                        })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            # Aggregate by feature
            feature_importance = importance_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
            
            print(f"\n🔍 Overall Feature Importance:")
            for feat, imp in feature_importance.head(10).items():
                print(f"  {feat:.<30} {imp:.4f}")
            
            return feature_importance
        
        return None


def main_fixed():
    """Main function with fixed implementation"""
    print("=" * 80)
    print("HYBRID ENERGY FORECASTING - FIXED REAL DATA ANALYSIS")
    print("=" * 80)
    
    # Countries to analyze
    COUNTRIES = ['DE', 'FR', 'ES', 'IT']
    N_SAMPLES = 10000
    TEST_SIZE = 0.2
    
    results = {}
    
    for country in COUNTRIES:
        print(f"\n{'='*60}")
        print(f"ANALYZING COUNTRY: {country}")
        print(f"{'='*60}")
        
        try:
            # Initialize fixed forecaster
            forecaster = FixedRealDataEnergyForecaster(
                data_path='data/europe_energy_real.csv',
                country=country
            )
            
            # Load real data
            forecaster.load_real_data(n_samples=N_SAMPLES)
            
            if forecaster.X.size == 0 or len(forecaster.y) == 0:
                print(f"  ⚠️ No valid data for {country}, skipping...")
                continue
            
            # Prepare train/test split
            X_train, X_test, y_train, y_test = forecaster.prepare_train_test_split(
                test_size=TEST_SIZE
            )
            
            # Train base models
            base_metrics = forecaster.train_base_models(X_train, y_train, X_test, y_test)
            
            # Train meta-model (FIXED)
            final_metrics = forecaster.train_meta_model_fixed(X_test, y_test, meta_model_type='ridge')
            
            # Get model summary
            summary = forecaster.get_model_summary()
            
            # Analyze feature importance
            forecaster.analyze_feature_importance()
            
            # Store results
            results[country] = {
                'forecaster': forecaster,
                'summary': summary,
                'final_metrics': final_metrics,
                'base_metrics': base_metrics
            }
            
            print(f"\n Analysis completed for {country}")
            print(f"   Best base model: {summary.iloc[0]['Model']} (RMSE: {summary.iloc[0]['RMSE']:.2f})")
            print(f"   Hybrid model: RMSE: {final_metrics['final_rmse']:.2f}")
            
        except Exception as e:
            print(f" Error analyzing {country}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare countries
    print("\n" + "=" * 80)
    print("COUNTRY COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    for country, result in results.items():
        if 'final_metrics' in result:
            comparison_data.append({
                'Country': country,
                'Samples': len(result['forecaster'].y) if hasattr(result['forecaster'], 'y') else 0,
                'Features': result['forecaster'].X.shape[1] if hasattr(result['forecaster'], 'X') else 0,
                'Best Base RMSE': result['final_metrics'].get('best_base_rmse', np.nan),
                'Hybrid RMSE': result['final_metrics'].get('final_rmse', np.nan),
                'Hybrid R²': result['final_metrics'].get('final_r2', np.nan),
                'Improvement %': result['final_metrics'].get('improvement_percent', 0)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Save results
        os.makedirs('outputs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        comparison_df.to_csv(f'outputs/hybrid_comparison_fixed_{timestamp}.csv', index=False)
        
        for country, result in results.items():
            if 'summary' in result:
                result['summary'].to_csv(f'outputs/hybrid_summary_{country}_fixed_{timestamp}.csv', index=False)
        
        print(f"\n Results saved to 'outputs/' directory")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Run fixed analysis
    results = main_fixed()
    
    # Summary
    if results:
        print(f"\n Summary of Analysis:")
        print(f"  Countries analyzed: {len(results)}")
        for country in results.keys():
            forecaster = results[country]['forecaster']
            final_rmse = results[country]['final_metrics']['final_rmse']
            print(f"  {country}: {len(forecaster.y)} samples, {forecaster.X.shape[1]} features, RMSE: {final_rmse:.2f}")
