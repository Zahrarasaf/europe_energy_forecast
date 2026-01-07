import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class StatisticalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.results = {}
    
    def comprehensive_analysis(self, target_country='DE'):
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE STATISTICAL ANALYSIS FOR {target_country}")
        print(f"{'='*60}")
        
        if target_country not in self.data.columns:
            print(f"Error: {target_country} not found in data columns")
            return None
        
        series = self.data[target_country].dropna()
        if len(series) < 50:
            print(f"Warning: Series has only {len(series)} observations")
        
        analysis_results = {
            'country': target_country,
            'basic_stats': {},
            'stationarity_tests': {},
            'distribution_tests': {},
            'autocorrelation': {},
            'seasonality': {}
        }
        
        # Basic Statistics
        print("\n1. BASIC STATISTICS:")
        print("-" * 40)
        
        basic_stats = {
            'count': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'median': series.median(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'iqr': series.quantile(0.75) - series.quantile(0.25),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'cv': (series.std() / series.mean()) * 100 if series.mean() != 0 else np.nan
        }
        
        for stat_name, stat_value in basic_stats.items():
            if isinstance(stat_value, float):
                print(f"{stat_name:15s}: {stat_value:.4f}")
            else:
                print(f"{stat_name:15s}: {stat_value}")
        
        analysis_results['basic_stats'] = basic_stats
        
        # Stationarity Tests
        print("\n2. STATIONARITY TESTS:")
        print("-" * 40)
        
        # ADF Test
        adf_result = adfuller(series, autolag='AIC')
        adf_test = {
            'test_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        print(f"Augmented Dickey-Fuller Test:")
        print(f"  ADF Statistic: {adf_test['test_statistic']:.6f}")
        print(f"  p-value: {adf_test['p_value']:.6f}")
        print(f"  Stationary: {'YES' if adf_test['is_stationary'] else 'NO'}")
        
        # KPSS Test
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            kpss_test = {
                'test_statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
            print(f"\nKPSS Test:")
            print(f"  KPSS Statistic: {kpss_test['test_statistic']:.6f}")
            print(f"  p-value: {kpss_test['p_value']:.6f}")
            print(f"  Stationary: {'YES' if kpss_test['is_stationary'] else 'NO'}")
        except Exception as e:
            print(f"\nKPSS Test Error: {e}")
            kpss_test = None
        
        analysis_results['stationarity_tests'] = {
            'adf': adf_test,
            'kpss': kpss_test
        }
        
        # Normality Tests
        print("\n3. NORMALITY TESTS:")
        print("-" * 40)
        
        # Shapiro-Wilk Test
        if len(series) <= 5000:
            shapiro_result = stats.shapiro(series)
            shapiro_test = {
                'test_statistic': shapiro_result[0],
                'p_value': shapiro_result[1],
                'is_normal': shapiro_result[1] > 0.05
            }
            print(f"Shapiro-Wilk Test:")
            print(f"  Statistic: {shapiro_test['test_statistic']:.6f}")
            print(f"  p-value: {shapiro_test['p_value']:.6f}")
            print(f"  Normal: {'YES' if shapiro_test['is_normal'] else 'NO'}")
        else:
            print("Shapiro-Wilk: Sample size too large (>5000)")
            shapiro_test = None
        
        # Jarque-Bera Test
        jb_result = stats.jarque_bera(series)
        jb_test = {
            'test_statistic': jb_result[0],
            'p_value': jb_result[1],
            'is_normal': jb_result[1] > 0.05
        }
        print(f"\nJarque-Bera Test:")
        print(f"  Statistic: {jb_test['test_statistic']:.6f}")
        print(f"  p-value: {jb_test['p_value']:.6f}")
        print(f"  Normal: {'YES' if jb_test['is_normal'] else 'NO'}")
        
        analysis_results['distribution_tests'] = {
            'shapiro': shapiro_test,
            'jarque_bera': jb_test
        }
        
        # Autocorrelation Analysis
        print("\n4. AUTOCORRELATION ANALYSIS:")
        print("-" * 40)
        
        max_lag = min(50, len(series) // 4)
        acf_values = acf(series, nlags=max_lag, fft=True)
        pacf_values = pacf(series, nlags=max_lag)
        
        significant_lags_acf = []
        significant_lags_pacf = []
        
        for lag in range(1, min(21, max_lag + 1)):
            if abs(acf_values[lag]) > 1.96 / np.sqrt(len(series)):
                significant_lags_acf.append(lag)
            if abs(pacf_values[lag]) > 1.96 / np.sqrt(len(series)):
                significant_lags_pacf.append(lag)
        
        print(f"First 20 ACF values:")
        for lag in range(1, min(21, len(acf_values))):
            print(f"  Lag {lag:2d}: {acf_values[lag]:.4f}")
        
        print(f"\nFirst 20 PACF values:")
        for lag in range(1, min(21, len(pacf_values))):
            print(f"  Lag {lag:2d}: {pacf_values[lag]:.4f}")
        
        print(f"\nSignificant ACF lags: {significant_lags_acf}")
        print(f"Significant PACF lags: {significant_lags_pacf}")
        
        analysis_results['autocorrelation'] = {
            'acf_values': acf_values.tolist(),
            'pacf_values': pacf_values.tolist(),
            'significant_acf_lags': significant_lags_acf,
            'significant_pacf_lags': significant_lags_pacf,
            'max_lag_analyzed': max_lag
        }
        
        # Seasonality Detection
        print("\n5. SEASONALITY ANALYSIS:")
        print("-" * 40)
        
        if len(series) >= 24 * 7 * 4:
            daily_pattern = self._detect_seasonality(series, period=24)
            weekly_pattern = self._detect_seasonality(series, period=24*7)
            
            print(f"Daily seasonality strength: {daily_pattern['strength']:.4f}")
            print(f"Weekly seasonality strength: {weekly_pattern['strength']:.4f}")
            
            analysis_results['seasonality'] = {
                'daily': daily_pattern,
                'weekly': weekly_pattern
            }
        else:
            print("Insufficient data for seasonality analysis")
            analysis_results['seasonality'] = None
        
        # Outlier Detection
        print("\n6. OUTLIER DETECTION:")
        print("-" * 40)
        
        outliers = self._detect_outliers(series)
        print(f"Number of outliers detected: {len(outliers)}")
        if len(outliers) > 0:
            print(f"Outlier values: {outliers[:10]}")  # Show first 10
            if len(outliers) > 10:
                print(f"... and {len(outliers)-10} more")
        
        analysis_results['outliers'] = {
            'count': len(outliers),
            'values': outliers.tolist() if len(outliers) > 0 else [],
            'percentage': (len(outliers) / len(series)) * 100
        }
        
        # Summary
        print("\n7. ANALYSIS SUMMARY:")
        print("-" * 40)
        
        summary = {
            'is_stationary': adf_test['is_stationary'],
            'is_normal': jb_test['is_normal'],
            'has_seasonality': analysis_results['seasonality'] is not None and 
                             (analysis_results['seasonality']['daily']['strength'] > 0.3 or
                              analysis_results['seasonality']['weekly']['strength'] > 0.3),
            'has_significant_autocorrelation': len(significant_lags_acf) > 0,
            'outlier_percentage': analysis_results['outliers']['percentage']
        }
        
        print(f"Stationary: {'YES' if summary['is_stationary'] else 'NO'}")
        print(f"Normally distributed: {'YES' if summary['is_normal'] else 'NO'}")
        print(f"Has seasonality: {'YES' if summary['has_seasonality'] else 'NO'}")
        print(f"Has autocorrelation: {'YES' if summary['has_significant_autocorrelation'] else 'NO'}")
        print(f"Outliers: {summary['outlier_percentage']:.2f}% of data")
        
        analysis_results['summary'] = summary
        self.results[target_country] = analysis_results
        
        return analysis_results
    
    def _detect_seasonality(self, series, period=24):
        if len(series) < period * 2:
            return {'strength': 0, 'period': period}
        
        series_values = series.values
        seasonal_components = []
        
        for i in range(period):
            idx = np.arange(i, len(series_values), period)
            if len(idx) > 1:
                seasonal_components.append(np.mean(series_values[idx]))
        
        seasonal_strength = np.std(seasonal_components) / np.std(series_values) if np.std(series_values) > 0 else 0
        
        return {
            'strength': seasonal_strength,
            'period': period,
            'seasonal_pattern': seasonal_components
        }
    
    def _detect_outliers(self, series, method='iqr', threshold=1.5):
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outliers = series[z_scores > 3]
        else:
            outliers = pd.Series([])
        
        return outliers
    
    def plot_comprehensive_analysis(self, target_country='DE', save_path=None):
        if target_country not in self.results:
            print(f"No analysis results found for {target_country}")
            return None
        
        series = self.data[target_country].dropna()
        analysis_results = self.results[target_country]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Comprehensive Statistical Analysis - {target_country}', fontsize=16, fontweight='bold')
        
        # 1. Time Series Plot
        axes[0, 0].plot(series.index, series.values, linewidth=1)
        axes[0, 0].set_title('Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram with KDE
        axes[0, 1].hist(series, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        sns.kdeplot(series, ax=axes[0, 1], color='red', linewidth=2)
        axes[0, 1].set_title('Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        
        # 3. Box Plot
        axes[0, 2].boxplot(series, vert=True)
        axes[0, 2].set_title('Box Plot')
        axes[0, 2].set_ylabel('Value')
        
        # 4. ACF Plot
        plot_acf(series, lags=min(50, len(series)//4), ax=axes[1, 0])
        axes[1, 0].set_title('Autocorrelation Function (ACF)')
        
        # 5. PACF Plot
        plot_pacf(series, lags=min(50, len(series)//4), ax=axes[1, 1])
        axes[1, 1].set_title('Partial Autocorrelation Function (PACF)')
        
        # 6. QQ Plot
        stats.probplot(series, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot')
        
        # 7. Rolling Statistics
        window_size = min(100, len(series) // 10)
        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()
        
        axes[2, 0].plot(series.index, series.values, alpha=0.5, label='Original', linewidth=0.5)
        axes[2, 0].plot(rolling_mean.index, rolling_mean.values, 'r-', label=f'Rolling Mean (window={window_size})', linewidth=2)
        axes[2, 0].set_title('Rolling Statistics')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Value')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Seasonal Decomposition (if enough data)
        if len(series) >= 24 * 7 * 2:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(series, model='additive', period=24)
                axes[2, 1].plot(decomposition.seasonal[:24*7])
                axes[2, 1].set_title('Weekly Seasonality Pattern')
                axes[2, 1].set_xlabel('Hour')
                axes[2, 1].set_ylabel('Seasonal Component')
            except:
                axes[2, 1].text(0.5, 0.5, 'Seasonal decomposition\nnot available', 
                               ha='center', va='center', transform=axes[2, 1].transAxes)
                axes[2, 1].set_title('Seasonal Pattern')
        else:
            axes[2, 1].text(0.5, 0.5, 'Insufficient data\nfor seasonal analysis', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Seasonal Pattern')
        
        # 9. Summary Statistics Text
        stats_text = f"""
        Basic Statistics:
        Mean: {analysis_results['basic_stats']['mean']:.2f}
        Std: {analysis_results['basic_stats']['std']:.2f}
        Min: {analysis_results['basic_stats']['min']:.2f}
        Max: {analysis_results['basic_stats']['max']:.2f}
        
        Stationarity:
        ADF p-value: {analysis_results['stationarity_tests']['adf']['p_value']:.6f}
        Stationary: {'YES' if analysis_results['summary']['is_stationary'] else 'NO'}
        
        Normality:
        JB p-value: {analysis_results['distribution_tests']['jarque_bera']['p_value']:.6f}
        Normal: {'YES' if analysis_results['summary']['is_normal'] else 'NO'}
        """
        
        axes[2, 2].axis('off')
        axes[2, 2].text(0, 0.95, stats_text, transform=axes[2, 2].transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def analyze_multiple_countries(self, country_codes=None):
        if country_codes is None:
            available_columns = [col for col in self.data.columns if col in ['DE', 'FR', 'IT', 'ES', 'GB', 'NL', 'PL', 'BE', 'AT']]
            country_codes = available_columns
        
        summary_results = []
        
        for country_code in country_codes:
            print(f"\n{'='*60}")
            print(f"Analyzing {country_code}...")
            result = self.comprehensive_analysis(country_code)
            
            if result:
                summary_results.append({
                    'Country': country_code,
                    'Mean': result['basic_stats']['mean'],
                    'Std': result['basic_stats']['std'],
                    'Stationary': result['summary']['is_stationary'],
                    'Normal': result['summary']['is_normal'],
                    'Seasonality': result['summary']['has_seasonality'],
                    'Outliers %': result['outliers']['percentage']
                })
        
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            print(f"\n{'='*60}")
            print("SUMMARY ACROSS ALL COUNTRIES:")
            print('='*60)
            print(summary_df.to_string(index=False))
            
            summary_csv_path = 'statistical_analysis_summary.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\nSummary saved to: {summary_csv_path}")
            
            return summary_df
        
        return None

def main():
    print("COMPREHENSIVE STATISTICAL ANALYSIS TOOL")
    print("=" * 50)
    
    try:
        df = pd.read_csv('data/europe_energy_real.csv')
        
        target_col = 'AT_load_actual_entsoe_transparency'
        if target_col in df.columns:
            print(f"Analyzing target column: {target_col}")
            data_series = pd.Series(df[target_col].values, name='AT')
        else:
            print("Target column not found, using first numeric column")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data_series = pd.Series(df[numeric_cols[0]].values, name='Data')
        
        analyzer = StatisticalAnalyzer(pd.DataFrame({target_col: data_series}))
        
        print("\n1. Running comprehensive analysis...")
        results = analyzer.comprehensive_analysis(target_col)
        
        print("\n2. Generating visualization...")
        analyzer.plot_comprehensive_analysis(target_col, 'statistical_analysis_plot.png')
        
        print("\n3. Analyzing multiple countries (if available)...")
        country_columns = [col for col in df.columns if any(country in col for country in ['DE', 'FR', 'IT', 'ES', 'GB'])]
        if len(country_columns) > 1:
            country_data = {}
            for col in country_columns[:5]:
                country_name = col.split('_')[0].upper()
                country_data[country_name] = df[col]
            
            multi_analyzer = StatisticalAnalyzer(pd.DataFrame(country_data))
            multi_analyzer.analyze_multiple_countries()
        
    except FileNotFoundError:
        print("Data file not found. Creating sample data for demonstration...")
        
        np.random.seed(42)
        n_samples = 1000
        time_index = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        
        base_trend = np.linspace(100, 150, n_samples)
        daily_seasonality = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
        weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24*7))
        noise = np.random.normal(0, 5, n_samples)
        
        sample_data = base_trend + daily_seasonality + weekly_seasonality + noise
        
        sample_df = pd.DataFrame({
            'DE': sample_data,
            'FR': sample_data * 0.8 + np.random.normal(0, 3, n_samples),
            'IT': sample_data * 1.2 + np.random.normal(0, 4, n_samples)
        }, index=time_index)
        
        analyzer = StatisticalAnalyzer(sample_df)
        
        print("\nAnalyzing Germany (DE) with sample data...")
        results = analyzer.comprehensive_analysis('DE')
        
        print("\nGenerating visualization...")
        analyzer.plot_comprehensive_analysis('DE')
        
        print("\nAnalyzing multiple countries...")
        analyzer.analyze_multiple_countries(['DE', 'FR', 'IT'])
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
