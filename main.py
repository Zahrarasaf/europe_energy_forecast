import os
import sys

sys.path.append('src')

def main():
    print("🎯 European Energy Forecasting - REAL Dataset")
    print("=" * 60)
    print("📁 Using: time_series_60min_singleindex.csv (124MB)")
    print("=" * 60)
    
    try:
        from data_collection.data_loader import (
            download_real_dataset, 
            manual_download_instructions,
            check_existing_file
        )
        from models.real_improvement_calculator import RealImprovementCalculator
        
        # 1. First check if file already exists from manual download
        print("1. Checking for existing dataset...")
        df = check_existing_file()
        
        if df is None:
            # 2. Try automated download
            print("\n2. Attempting automated download...")
            df = download_real_dataset()
        
        if df is None:
            # 3. Show manual instructions
            print("\n❌ Automated download failed")
            manual_download_instructions()
            return
        
        # 4. Dataset loaded successfully - now calculate improvement
        print(f"\n3. Dataset ready: {df.shape}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # 5. Find the right target column
        print("\n4. Finding energy load columns...")
        load_columns = [col for col in df.columns if 'load' in col.lower()]
        print(f"   Found {len(load_columns)} load-related columns")
        print(f"   First 5: {load_columns[:5]}")
        
        # 6. Calculate improvement
        calculator = RealImprovementCalculator()
        
        # Try different target columns
        target_candidates = [
            'DE_load_actual_entsoe_transparency',
            'load_DE',
            'DE_load'
        ] + load_columns[:3]  # Try first 3 load columns
        
        improvement = None
        for target_col in target_candidates:
            if target_col in df.columns:
                print(f"🎯 Trying target: {target_col}")
                improvement = calculator.calculate_real_improvement(df, target_col)
                if improvement is not None:
                    break
        
        if improvement is None and len(load_columns) > 0:
            # Use first available load column
            target_col = load_columns[0]
            print(f"🎯 Using first load column: {target_col}")
            improvement = calculator.calculate_real_improvement(df, target_col)
        
        if improvement is None:
            print("❌ Could not calculate improvement from any column")
            return
        
        # 7. Show results
        results = calculator.get_detailed_results()
        print(f"\n" + "=" * 50)
        print(f"🎯 REAL RESULTS:")
        print(f"   Dataset: {df.shape[0]:,} records, {df.shape[1]} features")
        print(f"   Baseline MAE: {results['baseline_mae']:.2f}")
        print(f"   Advanced MAE: {results['advanced_mae']:.2f}")
        print(f"   Improvement: {results['improvement_percentage']:+.1f}%")
        print(f"   ✅ REAL calculation from YOUR data!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
