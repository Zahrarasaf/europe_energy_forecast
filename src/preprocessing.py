import pandas as pd
import numpy as np


def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw OPSD time-series dataset.
    Steps:
        - convert timestamps
        - set datetime index
        - sort index
        - drop duplicates
        - remove columns with excessive missing values
        - interpolate remaining missing values
    """

    # Convert timestamps to datetime
    if "utc_timestamp" in df.columns:
        df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
        df = df.set_index("utc_timestamp")
        df = df.sort_index()

    # Drop duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # Remove columns where more than 40% values are missing
    df = df.dropna(axis=1, thresh=len(df) * 0.6)

    # Interpolate numeric columns
    df = df.interpolate(method="time")

    # Drop remaining NA
    df = df.dropna()

    return df


def select_features(df: pd.DataFrame, countries=None) -> pd.DataFrame:
    """
    Selects key features for forecasting.
    Typical columns include:
      - load
      - wind generation
      - solar generation
      - prices
    """

    if countries is None:
        countries = ["DE", "FR", "IT", "ES", "GB"]

    selected_cols = []

    for country in countries:
        for pattern in [
            f"{country}_load_actual",
            f"{country}_load_actual_entsoe_transparency",
            f"{country}_solar_generation_actual",
            f"{country}_wind_onshore_generation_actual",
            f"{country}_price_day_ahead"
        ]:
            cols = [c for c in df.columns if pattern in c]
            selected_cols.extend(cols)

    df = df[selected_cols]

    return df


def sample_subset(df: pd.DataFrame, n: int = 10000) -> pd.DataFrame:
    """Return a subset of the time series for experimentation."""
    if len(df) <= n:
        return df
    return df.iloc[:n]   # take the first N rows for modelling


def process_and_save(df: pd.DataFrame, output_path: str):
    """Full preprocessing pipeline."""
    df = clean_timeseries(df)
    df = select_features(df)
    df = sample_subset(df, n=10000)
    df.to_csv(output_path)
    return df
