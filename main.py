"""
Europe Energy Forecast - Multi-Country Analysis
Analyzes ALL European countries in the dataset.
"""

import sys
import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

# Import analysis modules
try:
    from src.analysis.carbon_impact import CarbonImpactAnalyzer
    from src.analysis.renewable_integration import RenewableIntegrationAnalyzer
    from src.analysis.economic_analysis import EconomicAnalyzer
    imports_successful = True
except ImportError:
    print("Creating built-in analyzers...")
    imports_successful = False
    
    # (Keep the analyzer classes from previous code)
    # ... [same analyzer classes] ...

def download_real_data():
    """Download dataset from Google Drive"""
    file_id = '1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s'
    output_path = 'data/europe_energy_real.csv'
    
    os.makedirs('data', exist_ok=True)
    
    if os.path.exists(output_path):
        file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(output_path))).days
        if file_age < 30:
            return output_path
    
    print("Downloading dataset...")
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        gdown.download(url, output_path, quiet=False)
        return output_path
    except:
        return None

def load_data(sample_size=None):
    """Load and prepare data"""
    data_path = download_real_data()
    
    if data_path and os.path.exists(data_path):
        try:
            if sample_size:
                df = pd.read_csv(data_path, nrows=sample_size)
            else:
                df = pd.read_csv(data_path)
            
            df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
            
            time_cols = [col for col in df.columns if 'timestamp' in col]
            if time_cols:
                time_col = time_cols[0]
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df.set_index(time_col, inplace=True)
                df = df.sort_index()
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].ffill().bfill()
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    return None

def get_all_countries(df):
    """Extract all country codes from dataset"""
    countries = set()
    for col in df.columns:
        if '_' in col:
            prefix = col.split('_')[0]
            if len(prefix) == 2 and prefix.isalpha():
                countries.add(prefix.upper())
    
    # Filter countries that have load data
    valid_countries = []
    for country in sorted(countries):
        load_col = f"{country.lower()}_load_actual_entsoe_transparency"
        if load_col in df.columns:
            valid_countries.append(country)
    
    return valid_countries

def analyze_country(df, country_code, improvement=0.15):
    """Analyze a single country"""
    if imports_successful:
        carbon_analyzer = CarbonImpactAnalyzer()
        renewable_analyzer = RenewableIntegrationAnalyzer()
        economic_analyzer = EconomicAnalyzer()
    else:
        carbon_analyzer = CarbonImpactAnalyzer()
        renewable_analyzer = RenewableIntegrationAnalyzer()
        economic_analyzer = EconomicAnalyzer()
    
    try:
        # Carbon impact
        carbon_impact = carbon_analyzer.calculate_carbon_reduction(df, improvement, country_code)
        
        # Renewable integration
        renewable_analysis = renewable_analyzer.analyze_renewable_integration(df, country_code)
        
        # Economic analysis
        co2_reduction = carbon_impact.get('annual_co2_reduction_tons', 0)
        energy_savings = carbon_impact.get('annual_energy_savings_mwh', 0)
        
        economic_impact = economic_analyzer.calculate_economic_savings(
            df, improvement, co2_reduction, energy_savings, country_code
        )
        
        return {
            'country': country_code,
            'carbon': carbon_impact,
            'renewable': renewable_analysis,
            'economic': economic_impact,
            'success': True
        }
        
    except Exception as e:
        print(f"Error analyzing {country_code}: {e}")
        return {
            'country': country_code,
            'success': False,
            'error': str(e)
        }

def analyze_all_countries(df, improvement=0.15, max_countries=None):
    """Analyze all countries in the dataset"""
    countries = get_all_countries(df)
    
    if max_countries:
        countries = countries[:max_countries]
    
    print(f"\nAnalyzing {len(countries)} countries: {', '.join(countries)}")
    
    results = []
    for i, country in enumerate(countries, 1):
        print(f"\n[{i}/{len(countries)}] Analyzing {country}...")
        result = analyze_country(df, country, improvement)
        results.append(result)
    
    return results

def create_summary_table(results):
    """Create summary table for all countries"""
    summary_data = []
    
    for result in results:
        if not result['success']:
            continue
            
        country = result['country']
        carbon = result['carbon']
        renewable = result['renewable']
        economic = result['economic']
        
        fossil_pct = renewable.get('renewable_sources', {}).get('fossil', {}).get('penetration_percentage', 100)
        renewable_pct = 100 - fossil_pct
        
        summary_data.append({
            'Country': country,
            'Fossil_Dependency_%': fossil_pct,
            'Renewable_Share_%': renewable_pct,
            'CO2_Reduction_Potential_Mt': carbon.get('annual_co2_reduction_tons', 0) / 1_000_000,
            'Energy_Savings_TWh': carbon.get('annual_energy_savings_mwh', 0) / 1_000_000,
            'Investment_€B': economic.get('initial_investment_eur', 0) / 1_000_000_000,
            'Annual_Savings_€M': economic.get('total_annual_savings_eur', 0) / 1_000_000,
            'Payback_Years': economic.get('payback_period_years', 0),
            'ROI_%': economic.get('roi_percentage', 0)
        })
    
    return pd.DataFrame(summary_data)

def save_comprehensive_results(results_df):
    """Save comprehensive results to CSV"""
    os.makedirs('outputs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/all_countries_analysis_{timestamp}.csv"
    
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
    
    # Also save a ranked version
    ranked_df = results_df.sort_values('Fossil_Dependency_%', ascending=False)
    ranked_filename = f"outputs/countries_ranked_by_fossil_{timestamp}.csv"
    ranked_df.to_csv(ranked_filename, index=False)
    print(f"Ranked results saved to: {ranked_filename}")
    
    return filename

def print_top_insights(results_df):
    """Print key insights from the analysis"""
    print("\n" + "="*80)
    print("KEY INSIGHTS - EUROPEAN ENERGY ANALYSIS")
    print("="*80)
    
    # Most fossil dependent
    most_fossil = results_df.nlargest(5, 'Fossil_Dependency_%')
    print("\n🔴 MOST FOSSIL-DEPENDENT COUNTRIES:")
    for _, row in most_fossil.iterrows():
        print(f"  {row['Country']}: {row['Fossil_Dependency_%']:.1f}% fossil")
    
    # Most renewable
    most_renewable = results_df.nlargest(5, 'Renewable_Share_%')
    print("\n✅ MOST RENEWABLE COUNTRIES:")
    for _, row in most_renewable.iterrows():
        print(f"  {row['Country']}: {row['Renewable_Share_%']:.1f}% renewable")
    
    # Best investments (low payback, high ROI)
    investment_df = results_df[results_df['Payback_Years'] < 20].copy()
    if len(investment_df) > 0:
        investment_df['Investment_Score'] = investment_df['ROI_%'] / investment_df['Payback_Years']
        best_investments = investment_df.nlargest(5, 'Investment_Score')
        
        print("\n💰 BEST INVESTMENT OPPORTUNITIES:")
        for _, row in best_investments.iterrows():
            print(f"  {row['Country']}: {row['Payback_Years']:.1f}y payback, {row['ROI_%']:.1f}% ROI")
    
    # Total potential
    total_co2_reduction = results_df['CO2_Reduction_Potential_Mt'].sum()
    total_investment = results_df['Investment_€B'].sum()
    total_savings = results_df['Annual_Savings_€M'].sum()
    
    print(f"\n📊 EUROPE-WIDE POTENTIAL (15% efficiency improvement):")
    print(f"  • Total CO2 reduction: {total_co2_reduction:,.1f} million tons/year")
    print(f"  • Total investment needed: €{total_investment:,.0f} billion")
    print(f"  • Total annual savings: €{total_savings:,.0f} million")
    
    # Calculate payback for Europe
    if total_savings > 0:
        europe_payback = (total_investment * 1000) / total_savings  # Convert billion to million
        print(f"  • Europe-wide payback period: {europe_payback:.1f} years")

def main():
    print("="*80)
    print("EUROPE ENERGY FORECAST - MULTI-COUNTRY ANALYSIS")
    print("="*80)
    
    try:
        improvement = 0.15
        
        print("\n1. Loading data...")
        df = load_data(sample_size=10000)  # Use sample for faster analysis
        
        if df is None:
            print("Failed to load data")
            return 1
        
        print(f"Data shape: {df.shape}")
        
        print("\n2. Analyzing all countries...")
        results = analyze_all_countries(df, improvement, max_countries=10)  # Limit to 10 for speed
        
        print("\n3. Creating summary...")
        results_df = create_summary_table(results)
        
        if len(results_df) == 0:
            print("No successful analyses")
            return 1
        
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        print(results_df.to_string())
        
        print_top_insights(results_df)
        
        print("\n4. Saving results...")
        save_comprehensive_results(results_df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Create visualization file
        create_visualization_script(results_df)
        
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1
    
    return 0

def create_visualization_script(results_df):
    """Create a Python script for visualizing results"""
    viz_script = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_european_energy_analysis(results_file):
    # Load results
    df = pd.read_csv(results_file)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('European Energy Analysis - All Countries', fontsize=16, fontweight='bold')
    
    # 1. Fossil dependency bar chart
    df_sorted = df.sort_values('Fossil_Dependency_%', ascending=True)
    axes[0, 0].barh(df_sorted['Country'], df_sorted['Fossil_Dependency_%'])
    axes[0, 0].set_xlabel('Fossil Dependency (%)')
    axes[0, 0].set_title('Fossil Fuel Dependency by Country')
    axes[0, 0].axvline(x=50, color='red', linestyle='--', alpha=0.5)
    
    # 2. Renewable share
    axes[0, 1].bar(df_sorted['Country'], df_sorted['Renewable_Share_%'])
    axes[0, 1].set_xlabel('Country')
    axes[0, 1].set_ylabel('Renewable Share (%)')
    axes[0, 1].set_title('Renewable Energy Share')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=50, color='green', linestyle='--', alpha=0.5)
    
    # 3. CO2 reduction potential
    axes[0, 2].bar(df_sorted['Country'], df_sorted['CO2_Reduction_Potential_Mt'])
    axes[0, 2].set_xlabel('Country')
    axes[0, 2].set_ylabel('CO2 Reduction Potential (Million tons)')
    axes[0, 2].set_title('CO2 Reduction Potential (15% efficiency)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Investment vs Savings scatter
    axes[1, 0].scatter(df['Investment_€B'], df['Annual_Savings_€M'], 
                      s=df['CO2_Reduction_Potential_Mt']*50, alpha=0.6)
    for i, row in df.iterrows():
        axes[1, 0].text(row['Investment_€B'], row['Annual_Savings_€M'], 
                       row['Country'], fontsize=8, alpha=0.7)
    axes[1, 0].set_xlabel('Investment Required (€ Billion)')
    axes[1, 0].set_ylabel('Annual Savings (€ Million)')
    axes[1, 0].set_title('Investment vs Annual Savings')
    
    # 5. Payback period
    colors = ['green' if x < 10 else 'orange' if x < 20 else 'red' for x in df['Payback_Years']]
    axes[1, 1].bar(df['Country'], df['Payback_Years'], color=colors)
    axes[1, 1].set_xlabel('Country')
    axes[1, 1].set_ylabel('Payback Period (Years)')
    axes[1, 1].set_title('Investment Payback Period')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    
    # 6. ROI
    axes[1, 2].bar(df['Country'], df['ROI_%'])
    axes[1, 2].set_xlabel('Country')
    axes[1, 2].set_ylabel('Return on Investment (%)')
    axes[1, 2].set_title('Return on Investment (ROI)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/european_energy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\\n=== SUMMARY STATISTICS ===")
    print(f"Number of countries analyzed: {len(df)}")
    print(f"Average fossil dependency: {df['Fossil_Dependency_%'].mean():.1f}%")
    print(f"Average renewable share: {df['Renewable_Share_%'].mean():.1f}%")
    print(f"Total CO2 reduction potential: {df['CO2_Reduction_Potential_Mt'].sum():.1f} million tons/year")
    print(f"Total investment required: €{df['Investment_€B'].sum():,.0f} billion")
    print(f"Total annual savings: €{df['Annual_Savings_€M'].sum():,.0f} million")

# Run visualization
if __name__ == "__main__":
    import glob
    result_files = glob.glob('outputs/all_countries_analysis_*.csv')
    if result_files:
        latest_file = max(result_files, key=lambda x: x)
        plot_european_energy_analysis(latest_file)
    else:
        print("No analysis files found. Run main_multi_country.py first.")
"""
    
    os.makedirs('scripts', exist_ok=True)
    with open('scripts/visualize_results.py', 'w') as f:
        f.write(viz_script)
    
    print(f"\nVisualization script created: scripts/visualize_results.py")
    print("Run: python scripts/visualize_results.py")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    
    exit_code = main()
    sys.exit(exit_code)
