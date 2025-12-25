import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def generate_comprehensive_report():
    """Generate comprehensive analysis report"""
    
    # Find latest results file
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        print("No outputs directory found")
        return
    
    result_files = [f for f in os.listdir(output_dir) if f.startswith('all_countries_analysis')]
    if not result_files:
        print("No analysis files found")
        return
    
    latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    df = pd.read_csv(os.path.join(output_dir, latest_file))
    
    # Create report
    report = f"""
    ================================================================================
    EUROPEAN ENERGY TRANSITION ANALYSIS REPORT
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================================
    
    EXECUTIVE SUMMARY
    =================
    - Countries analyzed: {len(df)}
    - Average fossil dependency: {df['Fossil_Dependency_%'].mean():.1f}%
    - Average renewable share: {df['Renewable_Share_%'].mean():.1f}%
    - Total CO2 reduction potential: {df['CO2_Reduction_Potential_Mt'].sum():.1f} million tons/year
    - Total investment required: €{df['Investment_€B'].sum():,.1f} billion
    - Total annual savings: €{df['Annual_Savings_€M'].sum():,.1f} million
    
    KEY FINDINGS
    ============
    
    1. FOSSIL DEPENDENCY RANKING:
    """
    
    # Fossil dependency ranking
    df_sorted = df.sort_values('Fossil_Dependency_%', ascending=False)
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        report += f"\n   {i}. {row['Country']}: {row['Fossil_Dependency_%']:.1f}% fossil"
    
    report += "\n\n2. BEST INVESTMENT OPPORTUNITIES (ROI > 20%):"
    
    # Best investments
    high_roi = df[df['ROI_%'] > 20].sort_values('ROI_%', ascending=False)
    for i, (_, row) in enumerate(high_roi.iterrows(), 1):
        report += f"\n   {i}. {row['Country']}: {row['ROI_%']:.1f}% ROI, {row['Payback_Years']:.1f} year payback"
    
    report += "\n\n3. FASTEST PAYBACK PERIODS (<5 years):"
    
    # Fast payback
    fast_payback = df[df['Payback_Years'] < 5].sort_values('Payback_Years')
    for i, (_, row) in enumerate(fast_payback.iterrows(), 1):
        report += f"\n   {i}. {row['Country']}: {row['Payback_Years']:.1f} years, €{row['Annual_Savings_€M']:,.0f}M annual savings"
    
    report += "\n\n4. HIGHEST CO2 REDUCTION POTENTIAL:"
    
    # Highest CO2 reduction
    high_co2 = df.nlargest(3, 'CO2_Reduction_Potential_Mt')
    for i, (_, row) in enumerate(high_co2.iterrows(), 1):
        report += f"\n   {i}. {row['Country']}: {row['CO2_Reduction_Potential_Mt']:.1f} million tons/year"
    
    report += """
    
    RECOMMENDATIONS
    ===============
    
    IMMEDIATE ACTION REQUIRED:
    1. Invest in countries with ROI > 20% and payback < 5 years
    2. Prioritize Germany (DE) for maximum CO2 reduction impact
    3. Focus on grid modernization in high-fossil countries
    
    MEDIUM-TERM STRATEGY:
    1. Develop renewable energy capacity in fossil-dependent countries
    2. Implement energy efficiency programs across all sectors
    3. Create cross-border energy sharing infrastructure
    
    LONG-TERM VISION:
    1. Achieve 50% renewable energy share across Europe by 2030
    2. Reduce average fossil dependency below 40% by 2040
    3. Create carbon-neutral energy system by 2050
    
    METHODOLOGY
    ===========
    - Analysis based on 15% energy efficiency improvement
    - CO2 intensity values from European Environment Agency
    - Economic calculations based on EU carbon pricing (€80/ton)
    - Investment costs estimated at €500 per MWh annual savings
    
    DATA SOURCES
    ============
    - ENTSO-E Transparency Platform (2014-2020 data)
    - Open Power System Data (OPSD)
    - European Environment Agency
    
    LIMITATIONS
    ===========
    - Analysis based on historical data (2014-2020)
    - Assumes linear scaling of efficiency improvements
    - Does not account for technological breakthroughs
    - Regional variations within countries not considered
    
    ================================================================================
    END OF REPORT
    ================================================================================
    """
    
    # Save report
    report_file = os.path.join(output_dir, f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report generated: {report_file}")
    
    # Create visualization
    create_visualizations(df, output_dir)

def create_visualizations(df, output_dir):
    """Create visualization charts"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Fossil dependency bar chart
    df_sorted = df.sort_values('Fossil_Dependency_%', ascending=True)
    axes[0, 0].barh(df_sorted['Country'], df_sorted['Fossil_Dependency_%'], color='#FF6B6B')
    axes[0, 0].set_xlabel('Fossil Dependency (%)')
    axes[0, 0].set_title('Fossil Fuel Dependency by Country')
    axes[0, 0].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[0, 0].legend()
    
    # 2. ROI vs Payback scatter
    colors = ['green' if x > 20 else 'orange' if x > 10 else 'red' for x in df['ROI_%']]
    scatter = axes[0, 1].scatter(df['Payback_Years'], df['ROI_%'], 
                                 s=df['CO2_Reduction_Potential_Mt']*100, 
                                 c=colors, alpha=0.6)
    
    # Add country labels
    for i, row in df.iterrows():
        axes[0, 1].text(row['Payback_Years'], row['ROI_%'], 
                       row['Country'], fontsize=8, alpha=0.7)
    
    axes[0, 1].set_xlabel('Payback Period (Years)')
    axes[0, 1].set_ylabel('Return on Investment (%)')
    axes[0, 1].set_title('Investment Analysis: ROI vs Payback Period')
    axes[0, 1].axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    
    # 3. CO2 reduction potential
    axes[1, 0].bar(df['Country'], df['CO2_Reduction_Potential_Mt'], color='#4ECDC4')
    axes[1, 0].set_xlabel('Country')
    axes[1, 0].set_ylabel('CO2 Reduction Potential (Million tons/year)')
    axes[1, 0].set_title('CO2 Reduction Potential (15% Efficiency Improvement)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Investment vs Savings
    axes[1, 1].scatter(df['Investment_€B'], df['Annual_Savings_€M'], 
                      s=df['CO2_Reduction_Potential_Mt']*50, alpha=0.6)
    
    for i, row in df.iterrows():
        axes[1, 1].text(row['Investment_€B'], row['Annual_Savings_€M'], 
                       row['Country'], fontsize=8, alpha=0.7)
    
    axes[1, 1].set_xlabel('Investment Required (€ Billion)')
    axes[1, 1].set_ylabel('Annual Savings (€ Million)')
    axes[1, 1].set_title('Investment vs Annual Savings')
    
    plt.tight_layout()
    
    # Save figure
    viz_file = os.path.join(output_dir, f"analysis_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {viz_file}")

if __name__ == "__main__":
    generate_comprehensive_report()
