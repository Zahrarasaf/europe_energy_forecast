# config.py - Updated for the new dataset

# Target variable - let's use Germany's actual load
TARGET_COUNTRY = 'DE'
TARGET_COLUMN = 'DE_load_actual_entsoe_transparency'

# Important countries for analysis
COUNTRIES = ['DE', 'FR', 'IT', 'ES', 'UK', 'NL', 'BE', 'PL']

# Key features for modeling
FEATURE_COLUMNS = [
    # Germany features
    'DE_load_forecast_entsoe_transparency',
    'DE_solar_generation_actual', 
    'DE_wind_generation_actual',
    'DE_price_day_ahead',
    
    # Neighbor countries actual load
    'FR_load_actual_entsoe_transparency',
    'IT_load_actual_entsoe_transparency', 
    'ES_load_actual_entsoe_transparency',
    'NL_load_actual_entsoe_transparency',
    
    # Time features will be created
]

# Model parameters
TEST_SIZE = 0.2
SEQUENCE_LENGTH = 24  # 24 hours for hourly data
FORECAST_HORIZON = 24  # Predict next 24 hours
