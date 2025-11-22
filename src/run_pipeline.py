import pandas as pd
from src.io_utils import load_raw_csv, save_processed
from src.preprocessing import clean_timeseries
from src.features import add_datetime_features
from src.models import train_arima, forecast_arima
from src.evaluation import evaluate_predictions

def main():
    # Load raw data
    df = load_raw_csv("data/raw/energy.csv")

    # Clean data
    df = clean_timeseries(df)

    # Feature engineering
    df = add_datetime_features(df)

    # Train ARIMA on target variable
    series = df["load"]
    model = train_arima(series)

    # Forecast next 24 hours
    forecast = forecast_arima(model, steps=24)

    print("Forecast:")
    print(forecast)

if __name__ == "__main__":
    main()
