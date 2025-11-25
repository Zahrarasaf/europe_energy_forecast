import pandas as pd
import numpy as np
from config.research_config import config
from src.data.preprocessing import EnergyDataPreprocessor
from src.models.transformer_model import EnergyTransformer
from src.analysis.statistical_tests import StatisticalAnalyzer
from src.utils.helpers import validate_data, print_project_info
import torch
import torch.optim as optim
import os

def main():
    print_project_info()
    
    # Check if data file exists
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ ERROR: Data file not found at {config.DATA_PATH}")
        print("Please make sure 'data/europe_energy.csv' exists")
        return
    
    print("1. Loading and preprocessing data...")
    preprocessor = EnergyDataPreprocessor(config)
    
    try:
        df = preprocessor.load_data()
        print(f"✅ Data loaded: {df.shape}")
        
        # Validate data
        validate_data(df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Basic analysis
    analyzer = StatisticalAnalyzer(df)
    analyzer.basic_analysis()
    
    # Stationarity check for target country
    stationarity = analyzer.check_stationarity(config.TARGET_COUNTRY)
    print(f"\nStationarity test for {config.TARGET_COUNTRY}:")
    print(f"ADF p-value: {stationarity['p_value']:.4f}")
    print(f"Is stationary: {stationarity['is_stationary']}")
    
    print("\n2. Creating advanced features...")
    try:
        df_processed = preprocessor.create_advanced_features(df)
        print(f"✅ Features created: {df_processed.shape}")
        
    except Exception as e:
        print(f"❌ Error in feature creation: {e}")
        return
    
    print("\n3. Preparing sequences for deep learning...")
    try:
        X_sequences, y_sequences = preprocessor.prepare_sequences(df_processed, config.TARGET_COUNTRY)
        print(f"✅ Sequences: {X_sequences.shape}")
        
    except Exception as e:
        print(f"❌ Error in sequence preparation: {e}")
        return
    
    # Only proceed if we have enough data
    if len(X_sequences) < config.SEQUENCE_LENGTH * 2:
        print("❌ Not enough data for training")
        return
    
    split_idx = int(len(X_sequences) * (1 - config.TEST_SIZE))
    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
    
    print(f"✅ Train/Test split: {len(X_train)}/{len(X_test)}")
    
    print("\n4. Transformer model setup...")
    input_dim = X_sequences.shape[2]
    model = EnergyTransformer(input_dim, config.SEQUENCE_LENGTH)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("5. Training demonstration...")
    model.train()
    
    # Demo training with few epochs
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs.squeeze(), torch.FloatTensor(y_train))
        loss.backward()
        optimizer.step()
        print(f"   Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print("\n🎉 PROJECT SETUP COMPLETED SUCCESSFULLY!")
    print("Next steps:")
    print("1. Add more sophisticated models")
    print("2. Implement proper cross-validation")
    print("3. Add hyperparameter tuning")
    print("4. Create research paper visualizations")

if __name__ == "__main__":
    main()
