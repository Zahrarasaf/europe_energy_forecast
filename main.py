import os
import sys

sys.path.append('src')

def main():
    print("🎯 European Energy Forecasting - REAL Calculation")
    print("=" * 60)
    
    try:
        from data_collection.data_loader import download_real_dataset
        from models.real_improvement_calculator import RealImprovementCalculator
        
        # 1. Load your REAL dataset
        print("1. Loading your dataset...")
        df = download_real_dataset()
        
        if df is None:
            print("❌ Could not load dataset")
            return
        
        print(f"✅ Dataset loaded: {df.shape}")
        
        # 2. Calculate REAL improvement
        print("2. Calculating REAL improvement from your data...")
        calculator = RealImprovementCalculator()
        
        # Try different target columns
        target_columns = [
            'DE_load_actual_entsoe_transparency',
            'DE_load_actual',
            'load_actual_DE'
        ]
        
        improvement = None
        for target_col in target_columns:
            if target_col in df.columns:
                improvement = calculator.calculate_real_improvement(df, target_col)
                if improvement is not None:
                    break
        
        if improvement is None:
            print("❌ Could not calculate improvement from your data")
            return
        
        # 3. Show REAL results
        results = calculator.get_detailed_results()
        print(f"\n🎯 REAL RESULTS FROM YOUR DATA:")
        print(f"   Baseline MAE: {results['baseline_mae']:.2f}")
        print(f"   Advanced MAE: {results['advanced_mae']:.2f}")
        print(f"   Improvement: {results['improvement_percentage']:+.1f}%")
        print(f"   Real Calculation: {results['is_real_calculation']}")
        
        print(f"\n✅ You can use {improvement:.1f}% in your CV - it's REAL!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
