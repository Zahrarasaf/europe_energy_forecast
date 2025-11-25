import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class AdvancedEnergyModels:
    def __init__(self):
        self.models = {}
    
    def create_lstm_model(self, input_shape):
        """LSTM model for time series forecasting"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def create_ensemble(self):
        """Create ensemble of models"""
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lstm': None  # Will be created with specific input shape
        }
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        results = {}
        
        for name, model in self.models.items():
            if name != 'lstm':
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                results[name] = {
                    'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                    'mae': mean_absolute_error(y_test, predictions)
                }
        
        return results
