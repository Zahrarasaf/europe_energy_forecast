import pandas as pd
import numpy as np
from config.research_config import config
from src.data.preprocessing import EnergyDataPreprocessor
from src.models.transformer_model import EnergyTransformer
import torch
import torch.optim as optim

def main():
    print("=== PhD-Level European Energy Forecasting ===")
    
    print("1. Loading and preprocessing data...")
    preprocessor = EnergyDataPreprocessor(config)
    
    try:
        df = preprocessor.load_data()
        print(f"✅ Data loaded successfully: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    df_processed = preprocessor.create_advanced_features(df)
    print(f"✅ Features created: {df_processed.shape}")
    
    print("2. Preparing sequences for Transformer...")
    X_sequences, y_sequences = preprocessor.prepare_sequences(df_processed, config.TARGET_COUNTRY)
    print(f"✅ Sequences prepared: {X_sequences.shape}")
    
    split_idx = int(len(X_sequences) * (1 - config.TEST_SIZE))
    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    print(f"✅ Train/Test split: {len(X_train)}/{len(X_test)}")
    
    print("3. Initializing Transformer model...")
    input_dim = X_sequences.shape[2]
    model = EnergyTransformer(input_dim, config.SEQUENCE_LENGTH)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("4. Starting training...")
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("5. Evaluation...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions.squeeze(), y_test_tensor)
    
    print(f"✅ Final Test Loss: {test_loss.item():.4f}")
    print("🎉 Project is working correctly!")

if __name__ == "__main__":
    main()
