# src/run_pipeline.py
import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime

# local paths (use your local repo path in production)
RAW_ZIP_PATH = "/mnt/data/time_series_60min_singleindex (1).zip"
EXTRACTED_CSV_PATH = "/mnt/data/opsd_extracted/time_series_60min_singleindex (1).csv"
# If you prefer to use data/raw in your repo, set RAW_CSV_LOCAL = "data/raw/time_series_60min_singleindex.csv"
RAW_CSV_LOCAL = "data/raw/time_series_60min_singleindex.csv"
PROCESSED_OUT = "data/processed/clean_europe_sample_10000.csv"
MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"

# set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except Exception:
    pass

# import project modules
from src.io_utils import load_csv_from_zip, load_csv, save_processed
from src.preprocessing import process_and_save, clean_timeseries, select_features, sample_subset
from src.features import build_feature_set, scale_features
from src.models import ForecastingModel, ARIMAModel, LSTMModel
from src.evaluation import evaluate_all

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_OUT), exist_ok=True)

def ensure_csv_available():
    """
    If uploaded file is in /mnt/data (environment), use it.
    Otherwise expect user to place the CSV at data/raw/...
    """
    if os.path.exists(EXTRACTED_CSV_PATH):
        print(f"Using extracted CSV at: {EXTRACTED_CSV_PATH}")
        return EXTRACTED_CSV_PATH
    if os.path.exists(RAW_ZIP_PATH):
        print(f"Reading CSV from zip: {RAW_ZIP_PATH}")
        # load_csv_from_zip returns DataFrame; we save local copy
        df = load_csv_from_zip(RAW_ZIP_PATH)
        df.to_csv(RAW_CSV_LOCAL, index=False)
        return RAW_CSV_LOCAL
    if os.path.exists(RAW_CSV_LOCAL):
        print(f"Using local CSV: {RAW_CSV_LOCAL}")
        return RAW_CSV_LOCAL
    raise FileNotFoundError("No data file found. Place your file at data/raw/ or upload to environment.")

def main():
    csv_path = ensure_csv_available()

    # 1) Load raw big CSV (careful with memory; for initial tests we sample)
    print("Loading data (this can take time)...")
    df_raw = load_csv(csv_path)

    # 2) Preprocess: timestamps, drop bad cols, interpolate, sample
    print("Preprocessing...")
    # use process_and_save from preprocessing: it will save 10k sample too
    processed = process_and_save(df_raw, PROCESSED_OUT)
    print("Saved processed sample to:", PROCESSED_OUT)

    # 3) Feature engineering: select target and create feature set
    print("Selecting features and building feature set...")
    # For demonstration pick aggregated totals if present, else pick a strong country (DE)
    # prefer solar_total/wind_total if autoproduced; otherwise pick DE_wind_onshore_generation_actual
    if "solar_total" not in processed.columns or "wind_total" not in processed.columns:
        # attempt to aggregate
        processed["solar_total"] = processed[[c for c in processed.columns if "solar" in c.lower()]].sum(axis=1, skipna=True)
        processed["wind_total"] = processed[[c for c in processed.columns if "wind" in c.lower()]].sum(axis=1, skipna=True)

    # choose target
    TARGET = "solar_total" if "solar_total" in processed.columns else processed.columns[0]

    # keep only target for prototype
    df_target = processed[[TARGET]].copy()

    # create features (lags, rolling) and scale
    features_df = build_feature_set(df_target, scaler_path=os.path.join(MODELS_DIR, "scaler.bin"))

    # split train/test by time
    n = len(features_df)
    split_index = int(n * 0.8)
    train_df = features_df.iloc[:split_index]
    test_df  = features_df.iloc[split_index:]

    # supervised for LSTM: last column is target(t)
    # prepare arrays
    X_train = train_df.drop(columns=[TARGET]).values
    y_train = train_df[TARGET].values
    X_test  = test_df.drop(columns=[TARGET]).values
    y_test  = test_df[TARGET].values

    # reshape for LSTM: (samples, timesteps, features_per_timestep)
    # We used create_lag_features and rolling that create flat columns; for simple LSTM, reshape by treating
    # each lag as one timestep and features_per_timestep=1 (this is a simplification).
    n_lags =  (len([c for c in train_df.columns if "_lag_" in c]) // 1)  # approx
    # To keep robust, compute timesteps as count of lag columns for target only:
    lag_cols = [c for c in train_df.columns if c.startswith(TARGET+"_lag_")]
    timesteps = len(lag_cols)
    X_train_seq = train_df[lag_cols].values.reshape((train_df.shape[0], timesteps, 1))
    X_test_seq  = test_df[lag_cols].values.reshape((test_df.shape[0], timesteps, 1))

    # 4) Train ARIMA and LSTM
    print("Training ARIMA model...")
    arima = ARIMAModel(order=(5,1,2))
    # ARIMA requires a univariate Series with datetime index; we use original df_target's index mapping
    # map train_df index back to original processed index by position
    original_series = df_target.loc[train_df.index[0]:train_df.index[-1]][TARGET]
    if len(original_series) < len(y_train):
        # fallback: use y_train as series
        arima_series = pd.Series(y_train)
    else:
        arima_series = original_series

    arima.fit(arima_series)
    joblib.dump(arima.model, os.path.join(MODELS_DIR, "arima_model.pkl"))

    print("Training LSTM model (this may take several minutes)...")
    lstm = LSTMModel(input_dim=1, neurons=64, n_in=timesteps)
    lstm.fit(X_train_seq, y_train, epochs=10, batch_size=64)
    lstm.model.save(os.path.join(MODELS_DIR, "lstm_model.h5"))

    # 5) Evaluate
    print("Evaluating models...")
    # ARIMA prediction for test length (naive approach)
    arima_preds = arima.predict(steps=len(y_test))
    # LSTM predict
    lstm_preds = lstm.predict(X_test_seq).reshape(-1)

    perf_arima = evaluate_all(y_test, arima_preds)
    perf_lstm  = evaluate_all(y_test, lstm_preds)

    print("ARIMA performance:", perf_arima)
    print("LSTM performance:", perf_lstm)

    # 6) Save artifacts (metrics, sample, models paths)
    artifacts = {
        "timestamp": datetime.utcnow().isoformat(),
        "target": TARGET,
        "n_rows_processed": len(processed),
        "train_rows": int(split_index),
        "test_rows": int(n - split_index),
        "performance": {
            "ARIMA": perf_arima,
            "LSTM": perf_lstm
        },
        "models": {
            "arima": os.path.abspath(os.path.join(MODELS_DIR, "arima_model.pkl")),
            "lstm": os.path.abspath(os.path.join(MODELS_DIR, "lstm_model.h5")),
            "scaler": os.path.abspath(os.path.join(MODELS_DIR, "scaler.bin"))
        }
    }
    with open(os.path.join(ARTIFACTS_DIR, "artifacts.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    print("Pipeline finished. Artifacts saved to:", ARTIFACTS_DIR)

if __name__ == "__main__":
    main()
