import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.research_config import config
from src.data.preprocessing import EnergyDataPreprocessor
from src.models.transformer_model import EnergyTransformer

def check_environment():
    """Check if all requirements are met"""
    print("=" * 60)
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Check data file
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ CRITICAL: Data file not found at {config.DATA_PATH}")
        print("Please make sure 'data/europe_energy.csv' exists")
        return False
    print(f"✅ Data file found: {config.DATA_PATH}")
    
    # Check required directories
    required_dirs = ['src/data', 'src/models', 'config']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Directory missing: {dir_path}")
            return False
    print("✅ All directories exist")
    
    # Check required files
    required_files = [
        'src/data/preprocessing.py',
        'src/models/transformer_model.py', 
        'config/research_config.py'
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ File missing: {file_path}")
            return False
    print("✅ All required files exist")
    
    return True

def print_dataset_info(df):
    """Print comprehensive dataset information"""
    print("\n" + "=" * 60)
    print("📊 DATASET INFORMATION")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total days: {(df.index.max() - df.index.min()).days}")
    print(f"Countries: {config.COUNTRIES}")
    print(f"Target country: {config.TARGET_COUNTRY}")
    
    print("\nBasic Statistics:")
    print(df[config.COUNTRIES].describe())
    
    print(f"\nMissing values:")
    print(df[config.COUNTRIES].isnull().sum())

def main():
    print("=" * 60)
    print("🎯 PhD-Level European Energy Forecasting")
    print("=" * 60)
    
    # 1. Environment check
    if not check_environment():
        print("\n🚨 Please fix the environment issues before continuing")
        return
    
    # 2. Data loading and preprocessing
    print("\n" + "=" * 60)
    print("1. LOADING AND PREPROCESSING DATA")
    print("=" * 60)
    
    preprocessor = EnergyDataPreprocessor(config)
    
    try:
        df = preprocessor.load_data()
        print_dataset_info(df)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 3. Feature engineering
    print("\n" + "=" * 60)
    print("2. FEATURE ENGINEERING")
    print("=" * 60)
    
    try:
        df_processed = preprocessor.create_advanced_features(df)
        print(f"✅ Processed data shape: {df_processed.shape}")
        
        # Show feature information
        feature_columns = [col for col in df_processed.columns if col not in config.COUNTRIES]
        print(f"✅ Created {len(feature_columns)} advanced features")
        print(f"Feature examples: {feature_columns[:8]}")
        
        if len(feature_columns) > 8:
            print(f"              ... and {len(feature_columns) - 8} more features")
            
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return
    
    # 4. Sequence preparation
    print("\n" + "=" * 60)
    print("3. SEQUENCE PREPARATION")
    print("=" * 60)
    
    try:
        X_sequences, y_sequences = preprocessor.prepare_sequences(df_processed, config.TARGET_COUNTRY)
        print(f"✅ X_sequences shape: {X_sequences.shape} (samples, sequence_length, features)")
        print(f"✅ y_sequences shape: {y_sequences.shape} (target values)")
        
        # Data validation
        if len(X_sequences) == 0:
            print("❌ Error: No sequences created")
            return
            
        print(f"✅ Total sequences: {len(X_sequences)}")
        print(f"✅ Sequence length: {config.SEQUENCE_LENGTH}")
        print(f"✅ Features per sequence: {X_sequences.shape[2]}")
        
    except Exception as e:
        print(f"❌ Error in sequence preparation: {e}")
        return
    
    # 5. Train-test split
    print("\n" + "=" * 60)
    print("4. DATA SPLITTING")
    print("=" * 60)
    
    split_idx = int(len(X_sequences) * (1 - config.TEST_SIZE))
    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
    
    print(f"✅ Training sequences: {len(X_train)}")
    print(f"✅ Testing sequences: {len(X_test)}")
    print(f"✅ Train/Test ratio: {len(X_train)/len(X_sequences)*100:.1f}% / {len(X_test)/len(X_sequences)*100:.1f}%")
    
    if len(X_train) < 100:
        print("⚠️  Warning: Very small training set")
    
    # 6. Model setup
    print("\n" + "=" * 60)
    print("5. TRANSFORMER MODEL SETUP")
    print("=" * 60)
    
    try:
        input_dim = X_sequences.shape[2]
        model = EnergyTransformer(input_dim, config.SEQUENCE_LENGTH)
        
        print(f"✅ Transformer model created successfully")
        print(f"✅ Input dimension: {input_dim}")
        print(f"✅ Sequence length: {config.SEQUENCE_LENGTH}")
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # 7. Training demonstration
    print("\n" + "=" * 60)
    print("6. DEMO TRAINING")
    print("=" * 60)
    
    try:
        # Use smaller subset for demo
        demo_size = min(500, len(X_train))
        X_demo = X_sequences[:demo_size]
        y_demo = y_sequences[:demo_size]
        
        X_tensor = torch.FloatTensor(X_demo)
        y_tensor = torch.FloatTensor(y_demo)
        
        print(f"Using {demo_size} sequences for demo training")
        
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        print("\nStarting training...")
        
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 1 == 0:
                print(f"   Epoch {epoch + 1}/5, Loss: {loss.item():.6f}")
        
        print("✅ Demo training completed successfully!")
        
        # Quick evaluation
        model.eval()
        with torch.no_grad():
            test_predictions = model(torch.FloatTensor(X_test[:100]))
            test_loss = criterion(test_predictions.squeeze(), torch.FloatTensor(y_test[:100]))
            print(f"✅ Demo test loss: {test_loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ Error in training: {e}")
        return
    
    # 8. Final summary
    print("\n" + "=" * 60)
    print("🎉 PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"\n📈 PROJECT SUMMARY:")
    print(f"   • Dataset: {df.shape[0]:,} records, {len(config.COUNTRIES)} countries")
    print(f"   • Features: {len(feature_columns)} engineered features") 
    print(f"   • Sequences: {len(X_sequences):,} training sequences")
    print(f"   • Model: Transformer with {total_params:,} parameters")
    print(f"   • Target: {config.TARGET_COUNTRY} energy consumption")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Implement full training with all data")
    print(f"   2. Add proper validation and cross-validation")
    print(f"   3. Implement additional models (LSTM, XGBoost, etc.)")
    print(f"   4. Add hyperparameter tuning")
    print(f"   5. Create research visualizations and analysis")
    
    print(f"\n💡 TIPS FOR PHD RESEARCH:")
    print(f"   • Focus on novel feature engineering methods")
    print(f"   • Compare Transformer against traditional time series models")
    print(f"   • Analyze cross-country energy dependencies")
    print(f"   • Implement uncertainty quantification")
    print(f"   • Prepare publication-ready results")

if __name__ == "__main__":
    main()
