import pandas as pd
from baseline_model import calculate_baseline_accuracy
from advanced_model import train_advanced_model

def main():
    # Load data
    df = pd.read_csv('data/europe_energy.csv')
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('utc_timestamp', inplace=True)
    
    target_col = 'DE_load_actual_entsoe_transparency'
    
    print("Calculating Real Performance Improvement...")
    print("=" * 50)
    
    # 1. Calculate baseline
    baseline_results = calculate_baseline_accuracy(df, target_col)
    
    print("\n" + "=" * 50)
    
    # 2. Calculate advanced model
    advanced_results = train_advanced_model(df, target_col)
    
    if advanced_results:
        print("\n" + "=" * 50)
        print("PERFORMANCE IMPROVEMENT:")
        print("=" * 50)
        
        # Calculate percentage improvement
        mae_improvement = ((baseline_results['mae'] - advanced_results['mae']) / baseline_results['mae']) * 100
        rmse_improvement = ((baseline_results['rmse'] - advanced_results['rmse']) / baseline_results['rmse']) * 100
        mape_improvement = ((baseline_results['mape'] - advanced_results['mape']) / baseline_results['mape']) * 100
        
        print(f"MAE Improvement: {mae_improvement:+.1f}%")
        print(f"RMSE Improvement: {rmse_improvement:+.1f}%") 
        print(f"MAPE Improvement: {mape_improvement:+.1f}%")
        
        # Average improvement
        avg_improvement = (mae_improvement + rmse_improvement + mape_improvement) / 3
        print(f"Average Improvement: {avg_improvement:+.1f}%")
        
        return avg_improvement
    
    return None

if __name__ == "__main__":
    improvement = main()
