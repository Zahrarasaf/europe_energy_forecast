import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

class StatisticalAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def comprehensive_analysis(self, target_country='DE'):
        print(f"=== STATISTICAL ANALYSIS FOR {target_country} ===")
        
        # Stationarity test
        result = adfuller(self.data[target_country].dropna())
        print(f"ADF p-value: {result[1]:.6f}")
        print(f"Stationary: {result[1] < 0.05}")
        
        # Basic stats
        print(f"Mean: {self.data[target_country].mean():.2f}")
        print(f"Std: {self.data[target_country].std():.2f}")
        print(f"Min: {self.data[target_country].min():.2f}")
        print(f"Max: {self.data[target_country].max():.2f}")
