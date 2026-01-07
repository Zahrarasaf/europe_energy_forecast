import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

class EnhancedStatisticalAnalyzer:
    def __init__(self, data_path='data/europe_energy_real.csv'):
        try:
            self.df = pd.read_csv(data_path)
            print(f"Data loaded successfully: {self.df.shape}")
            
            self.load_columns = [col for col in self.df.columns if 'load_actual' in col]
            print(f"Found {len(self.load_columns)} load columns")
            
            if len(self.load_columns) > 0:
                self.country_load_map = {}
                for col in self.load_columns:
                    parts = col.split('_')
                    if len(parts) > 0:
                        country_code = parts[0].upper()
                        if country_code not in self.country_load_map:
                            self.country_load_map[country_code] = []
                        self.country_load_map[country_code].append(col)
                
                self.country_codes = list(self.country_load_map.keys())
                print(f"Countries detected: {self.country_codes}")
                print(f"Sample column names: {list(self.load_columns[:5])}")
                
            else:
                print("No load columns found!")
                self.country_codes = []
                
        except FileNotFoundError:
            print(f"File not found: {data_path}")
            self.df = None
            self.load_columns = []
            self.country_codes = []
        
        self.results = {}
    
    def analyze_country(self, country_code):
        if self.df is None:
            print("No data available")
            return None
        
        if country_code not in self.country_load_map:
            print(f"No load data found for {country_code}")
            print(f"Available columns for {country_code}: {self.country_load_map.get(country_code, [])}")
            return None
        
        load_cols = self.country_load_map[country_code]
        
        if not load_cols:
            print(f"No load columns for {country_code}")
            return None
        
        load_col = load_cols[0]
        series = self.df[load_col].dropna()
        
        print(f"\n{'='*60}")
        print(f"ANALYZING {country_code} - {load_col}")
        print(f"{'='*60}")
        
        analysis = self._perform_analysis(series, country_code, load_col)
        
        self.results[country_code] = analysis
        
        return analysis
    
    def _perform_analysis(self, series, country_code, column_name):
        analysis = {
            'country': country_code,
            'column': column_name,
            'n_samples': len(series),
            'basic_stats': {},
            'tests': {},
            'patterns': {}
        }
        
        analysis['basic_stats'] = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'median': float(series.median()),
            'q1': float(series.quantile(0.25)),
            'q3': float(series.quantile(0.75)),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis())
        }
        
        print(f"\nBASIC STATISTICS:")
        stats = analysis['basic_stats']
        print(f"   Samples: {analysis['n_samples']:,}")
        print(f"   Mean: {stats['mean']:,.1f} MW")
        print(f"   Std: {stats['std']:,.1f} MW")
        print(f"   Range: {stats['min']:,.0f} to {stats['max']:,.0f} MW")
        print(f"   Median: {stats['median']:,.0f} MW")
        
        print(f"\nSTATIONARITY TESTS:")
        
        try:
            adf_result = adfuller(series.dropna())
            adf_p = adf_result[1]
            is_adf_stationary = adf_p < 0.05
            
            print(f"   ADF Test p-value: {adf_p:.6f}")
            print(f"   ADF Stationary: {'YES' if is_adf_stationary else 'NO'}")
            
            analysis['tests']['adf'] = {
                'p_value': float(adf_p),
                'stationary': is_adf_stationary
            }
        except Exception as e:
            print(f"   ADF Test failed: {e}")
        
        print(f"\nAUTOCORRELATION:")
        
        try:
            max_lag = min(48, len(series) // 10)
            acf_vals = acf(series, nlags=max_lag, fft=True)
            pacf_vals = pacf(series, nlags=max_lag)
            
            significant_lags = []
            for lag in range(1, min(25, max_lag)):
                if abs(acf_vals[lag]) > 1.96 / np.sqrt(len(series)):
                    significant_lags.append(lag)
            
            print(f"   Significant ACF lags (first 24): {significant_lags[:24]}")
            
            analysis['patterns']['autocorrelation'] = {
                'significant_lags': significant_lags[:24],
                'acf_1': float(acf_vals[1]),
                'acf_24': float(acf_vals[24]) if len(acf_vals) > 24 else None,
                'acf_168': float(acf_vals[168]) if len(acf_vals) > 168 else None
            }
        except Exception as e:
            print(f"   Autocorrelation analysis failed: {e}")
        
        print(f"\nSEASONALITY ANALYSIS:")
        
        try:
            if len(series) >= 168 * 2:
                daily_seasonality = self._calculate_seasonality_strength(series, 24)
                weekly_seasonality = self._calculate_seasonality_strength(series, 168)
                
                print(f"   Daily (24h) seasonality strength: {daily_seasonality:.3f}")
                print(f"   Weekly (168h) seasonality strength: {weekly_seasonality:.3f}")
                
                analysis['patterns']['seasonality'] = {
                    'daily_strength': float(daily_seasonality),
                    'weekly_strength': float(weekly_seasonality),
                    'has_daily_seasonality': daily_seasonality > 0.3,
                    'has_weekly_seasonality': weekly_seasonality > 0.3
                }
        except Exception as e:
            print(f"   Seasonality analysis failed: {e}")
        
        print(f"\nOUTLIER DETECTION:")
        
        try:
            outliers = self._detect_outliers_iqr(series)
            outlier_percentage = (len(outliers) / len(series)) * 100
            
            print(f"   Outliers detected: {len(outliers):,}")
            print(f"   Percentage: {outlier_percentage:.2f}%")
            
            analysis['patterns']['outliers'] = {
                'count': len(outliers),
                'percentage': float(outlier_percentage),
                'values': outliers.tolist() if len(outliers) > 0 else []
            }
        except Exception as e:
            print(f"   Outlier detection failed: {e}")
        
        print(f"\nMODELING RECOMMENDATIONS:")
        
        recs = self._generate_modeling_recommendations(analysis)
        for rec in recs:
            print(f"   • {rec}")
        
        analysis['recommendations'] = recs
        
        return analysis
    
    def _calculate_seasonality_strength(self, series, period):
        if len(series) < period * 2:
            return 0
        
        try:
            series_values = series.values
            seasonal_pattern = []
            
            for i in range(period):
                indices = np.arange(i, len(series_values), period)
                if len(indices) > 1:
                    seasonal_pattern.append(np.mean(series_values[indices]))
            
            if len(seasonal_pattern) == 0:
                return 0
            
            return np.std(seasonal_pattern) / np.std(series_values)
        except:
            return 0
    
    def _detect_outliers_iqr(self, series, multiplier=1.5):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return series[(series < lower_bound) | (series > upper_bound)]
    
    def _generate_modeling_recommendations(self, analysis):
        recs = []
        
        if analysis['tests'].get('adf', {}).get('stationary', False):
            recs.append("Data is stationary - no differencing needed")
        else:
            recs.append("Consider differencing (d=1 or d=2)")
        
        sig_lags = analysis['patterns'].get('autocorrelation', {}).get('significant_lags', [])
        if len(sig_lags) > 0:
            recs.append(f"AR terms recommended (p={min(3, len(sig_lags))})")
        
        seasonality = analysis['patterns'].get('seasonality', {})
        if seasonality.get('has_weekly_seasonality', False):
            recs.append("Use weekly seasonality (s=168)")
        if seasonality.get('has_daily_seasonality', False):
            recs.append("Consider daily seasonality patterns")
        
        outlier_pct = analysis['patterns'].get('outliers', {}).get('percentage', 0)
        if outlier_pct > 5:
            recs.append(f"Handle outliers ({outlier_pct:.1f}% of data)")
        
        return recs
    
    def analyze_all_countries(self, max_countries=10):
        if not self.country_codes:
            print("No countries to analyze")
            return None
        
        print(f"\n{'='*70}")
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print(f"{'='*70}")
        
        all_analyses = {}
        
        countries_to_analyze = self.country_codes[:max_countries]
        
        for country_code in countries_to_analyze:
            try:
                print(f"\nAnalyzing {country_code}...")
                analysis = self.analyze_country(country_code)
                if analysis:
                    all_analyses[country_code] = analysis
                else:
                    print(f"Skipping {country_code} - no analysis results")
            except Exception as e:
                print(f"Error analyzing {country_code}: {e}")
                import traceback
                traceback.print_exc()
        
        if all_analyses:
            self._create_summary_report(all_analyses)
        else:
            print("\nNo analyses completed successfully")
        
        return all_analyses
    
    def _create_summary_report(self, analyses):
        if not analyses:
            return
        
        print(f"\n{'='*70}")
        print("SUMMARY REPORT")
        print(f"{'='*70}")
        
        summary_data = []
        for country_code, analysis in analyses.items():
            stats = analysis['basic_stats']
            tests = analysis['tests']
            patterns = analysis['patterns']
            
            summary_data.append({
                'Country': country_code,
                'Samples': analysis['n_samples'],
                'Mean_MW': stats['mean'],
                'Std_MW': stats['std'],
                'Min_MW': stats['min'],
                'Max_MW': stats['max'],
                'Stationary': tests.get('adf', {}).get('stationary', False),
                'Weekly_Seasonality': patterns.get('seasonality', {}).get('has_weekly_seasonality', False),
                'Outliers_Percentage': patterns.get('outliers', {}).get('percentage', 0),
                'ACF_Lag1': patterns.get('autocorrelation', {}).get('acf_1', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(float_format=lambda x: f"{x:,.2f}" if isinstance(x, float) else str(x)))
        
        output_file = 'enhanced_statistical_analysis.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\nReport saved to: {output_file}")
        
        self._create_visual_summary(summary_df)
    
    def _create_visual_summary(self, summary_df):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
            
            axes[0, 0].bar(summary_df['Country'], summary_df['Mean_MW'], color='skyblue')
            axes[0, 0].set_title('Average Load by Country')
            axes[0, 0].set_ylabel('MW')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            axes[0, 1].bar(summary_df['Country'], summary_df['Outliers_Percentage'], color='salmon')
            axes[0, 1].set_title('Outlier Percentage by Country')
            axes[0, 1].set_ylabel('%')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            colors = ['green' if s else 'red' for s in summary_df['Stationary']]
            axes[1, 0].bar(summary_df['Country'], [1] * len(summary_df), color=colors)
            axes[1, 0].set_title('Stationarity Status')
            axes[1, 0].set_yticks([])
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            axes[1, 1].bar(summary_df['Country'], summary_df['ACF_Lag1'], color='purple')
            axes[1, 1].set_title('Autocorrelation at Lag 1')
            axes[1, 1].set_ylabel('ACF Value')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('statistical_summary_plot.png', dpi=100, bbox_inches='tight')
            plt.show()
            
            print(f"\nVisualization saved to: statistical_summary_plot.png")
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def plot_detailed_analysis(self, country_code, save_path=None):
        if country_code not in self.results:
            print(f"No analysis found for {country_code}")
            return None
        
        analysis = self.results[country_code]
        
        load_cols = self.country_load_map.get(country_code, [])
        if not load_cols:
            return None
        
        load_col = load_cols[0]
        series = self.df[load_col].dropna()
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Detailed Analysis - {country_code} ({load_col})', fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(series.index, series.values, linewidth=0.5)
        axes[0, 0].set_title('Time Series')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('MW')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(series, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        sns.kdeplot(series, ax=axes[0, 1], color='red', linewidth=2)
        axes[0, 1].set_title('Distribution')
        axes[0, 1].set_xlabel('MW')
        axes[0, 1].set_ylabel('Density')
        
        axes[0, 2].boxplot(series, vert=True)
        axes[0, 2].set_title('Box Plot')
        axes[0, 2].set_ylabel('MW')
        
        plot_acf(series, lags=min(100, len(series)//4), ax=axes[1, 0])
        axes[1, 0].set_title('Autocorrelation Function (ACF)')
        
        plot_pacf(series, lags=min(50, len(series)//4), ax=axes[1, 1])
        axes[1, 1].set_title('Partial Autocorrelation Function (PACF)')
        
        stats.probplot(series, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot')
        
        window_size = min(100, len(series) // 10)
        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()
        
        axes[2, 0].plot(series.index, series.values, alpha=0.5, linewidth=0.5)
        axes[2, 0].plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2)
        axes[2, 0].set_title(f'Rolling Mean (window={window_size})')
        axes[2, 0].set_xlabel('Index')
        axes[2, 0].set_ylabel('MW')
        axes[2, 0].grid(True, alpha=0.3)
        
        if len(series) >= 168 * 7:
            daily_avg = series.groupby(series.index % 24).mean()
            axes[2, 1].plot(range(24), daily_avg[:24], 'g-', linewidth=2)
            axes[2, 1].set_title('Daily Pattern (24h)')
            axes[2, 1].set_xlabel('Hour of Day')
            axes[2, 1].set_ylabel('Average MW')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'Insufficient data for daily pattern', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Daily Pattern')
        
        stats_text = f"""
        Basic Statistics:
        Mean: {analysis['basic_stats']['mean']:,.1f} MW
        Std: {analysis['basic_stats']['std']:,.1f} MW
        Min: {analysis['basic_stats']['min']:,.0f} MW
        Max: {analysis['basic_stats']['max']:,.0f} MW
        
        Stationarity:
        ADF p-value: {analysis['tests'].get('adf', {}).get('p_value', 'N/A'):.6f}
        Stationary: {'YES' if analysis['tests'].get('adf', {}).get('stationary', False) else 'NO'}
        
        Seasonality:
        Weekly strength: {analysis['patterns'].get('seasonality', {}).get('weekly_strength', 0):.3f}
        Daily strength: {analysis['patterns'].get('seasonality', {}).get('daily_strength', 0):.3f}
        """
        
        axes[2, 2].axis('off')
        axes[2, 2].text(0, 0.95, stats_text, transform=axes[2, 2].transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detailed plot saved to: {save_path}")
        else:
            save_path = f"{country_code}_detailed_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detailed plot saved to: {save_path}")
        
        plt.show()
        return fig

def main():
    print("ENHANCED STATISTICAL ANALYZER")
    print("=" * 50)
    
    analyzer = EnhancedStatisticalAnalyzer('data/europe_energy_real.csv')
    
    if analyzer.df is not None and analyzer.country_codes:
        print(f"\nStarting analysis for first 10 countries...")
        print(f"Sample column names found:")
        for i, col in enumerate(analyzer.load_columns[:5]):
            print(f"  {i+1}. {col}")
        
        results = analyzer.analyze_all_countries(max_countries=10)
        
        if results:
            print(f"\nAnalysis completed successfully!")
            print(f"Countries analyzed: {list(results.keys())}")
            print(f"Reports generated: enhanced_statistical_analysis.csv")
            print(f"Visualization: statistical_summary_plot.png")
            
            if results:
                print(f"\nGenerating detailed analysis for first country...")
                first_country = list(results.keys())[0]
                analyzer.plot_detailed_analysis(first_country)
    else:
        print("\nUnable to load data or find load columns")
        print("Checking available columns...")
        if analyzer.df is not None:
            print(f"Total columns: {len(analyzer.df.columns)}")
            print(f"First 10 columns: {list(analyzer.df.columns[:10])}")

if __name__ == "__main__":
    main()
