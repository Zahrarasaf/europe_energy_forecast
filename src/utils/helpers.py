import pandas as pd
import numpy as np

def validate_data(df):
    """Validate the energy dataset"""
    required_columns = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Data contains missing values")
    
    return True

def print_project_info():
    info = """
    === EUROPEAN ENERGY FORECASTING - PhD PROJECT ===
    
    Countries: DE, FR, IT, ES, UK, NL, BE, PL
    Period: 2006-2017
    Target: Energy consumption forecasting
    
    Models: Transformer, LSTM, Statistical models
    """
    print(info)
