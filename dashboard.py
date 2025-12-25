import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="Europe Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Europe Energy Forecast Dashboard")
st.markdown("Real-time analysis of European energy data and decarbonization potential")

# Sidebar for configuration
st.sidebar.header("Configuration")
country = st.sidebar.selectbox(
    "Select Country",
    ["DE", "FR", "ES", "IT", "GB", "NL", "SE", "DK", "AT"],
    index=0
)

improvement = st.sidebar.slider(
    "Energy Efficiency Improvement (%)",
    min_value=5,
    max_value=30,
    value=15,
    step=5
) / 100

# Load results if available
results_file = None
if os.path.exists('outputs'):
    result_files = [f for f in os.listdir('outputs') if f.endswith('.csv')]
    if result_files:
        latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join('outputs', x)))
        results_file = os.path.join('outputs', latest_file)

if results_file:
    df_results = pd.read_csv(results_file)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Fossil Dependency",
            f"{df_results['Fossil_Percentage'].iloc[0]:.1f}%",
            delta="-2.5%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "CO2 Reduction Potential",
            f"{df_results['Annual_CO2_Reduction_tons'].iloc[0]/1_000_000:.1f}M tons",
            help="Annual CO2 reduction from efficiency improvements"
        )
    
    with col3:
        st.metric(
            "Investment Required",
            f"€{df_results['Initial_Investment_EUR'].iloc[0]/1_000_000_000:.1f}B",
            help="Initial investment for efficiency measures"
        )
    
    with col4:
        st.metric(
            "Payback Period",
            f"{df_results['Payback_Period_Years'].iloc[0]:.1f} years",
            delta="-0.5",
            delta_color="normal"
        )
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["Energy Mix", "Economic Impact", "Carbon Reduction"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Energy mix pie chart
            energy_data = {
                'Source': ['Solar', 'Wind', 'Hydro', 'Fossil'],
                'Percentage': [
                    df_results['Solar_Percentage'].iloc[0],
                    df_results['Wind_Percentage'].iloc[0],
                    df_results['Hydro_Percentage'].iloc[0],
                    df_results['Fossil_Percentage'].iloc[0]
                ]
            }
            
            fig = px.pie(
                energy_data,
                values='Percentage',
                names='Source',
                title=f'Energy Mix - {country}',
                color='Source',
                color_discrete_map={
                    'Solar': '#FFD700',
                    'Wind': '#87CEEB',
                    'Hydro': '#1E90FF',
                    'Fossil': '#8B0000'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Renewable vs Fossil bar chart
            fig = px.bar(
                x=['Renewable', 'Fossil'],
                y=[
                    100 - df_results['Fossil_Percentage'].iloc[0],
                    df_results['Fossil_Percentage'].iloc[0]
                ],
                title='Renewable vs Fossil Energy',
                labels={'x': 'Energy Type', 'y': 'Percentage (%)'},
                color=['Renewable', 'Fossil'],
                color_discrete_map={'Renewable': '#2E8B57', 'Fossil': '#8B0000'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Economic metrics
            economic_data = {
                'Metric': ['Annual Savings', 'Investment', 'NPV (20y)'],
                'Value (€B)': [
                    df_results['Total_Annual_Savings_EUR'].iloc[0] / 1_000_000_000,
                    df_results['Initial_Investment_EUR'].iloc[0] / 1_000_000_000,
                    df_results['NPV_EUR'].iloc[0] / 1_000_000_000
                ]
            }
            
            fig = px.bar(
                economic_data,
                x='Metric',
                y='Value (€B)',
                title='Economic Impact (Billion €)',
                color='Metric'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROI gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=df_results['ROI_Percentage'].iloc[0],
                title={'text': "Return on Investment (ROI)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgray"},
                        {'range': [10, 20], 'color': "gray"},
                        {'range': [20, 50], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 15
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # CO2 reduction impact
            impact_data = {
                'Impact': ['CO2 Reduction', 'Equivalent Cars', 'Trees Planted'],
                'Value': [
                    df_results['Annual_CO2_Reduction_tons'].iloc[0] / 1_000_000,
                    df_results['Equivalent_Cars_Removed'].iloc[0] / 1_000_000,
                    df_results['Equivalent_Trees_Planted'].iloc[0] / 1_000_000
                ],
                'Unit': ['M tons', 'M cars', 'M trees']
            }
            
            fig = px.bar(
                impact_data,
                x='Impact',
                y='Value',
                title='Environmental Impact',
                text='Unit',
                color='Impact'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payback period visualization
            years = list(range(21))
            cumulative_savings = [
                df_results['Total_Annual_Savings_EUR'].iloc[0] * year 
                for year in years
            ]
            investment = [df_results['Initial_Investment_EUR'].iloc[0]] * len(years)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=cumulative_savings,
                name='Cumulative Savings',
                line=dict(color='green', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=years,
                y=investment,
                name='Initial Investment',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig.update_layout(
                title='Payback Period Analysis',
                xaxis_title='Years',
                yaxis_title='Amount (€)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.header("Recommendations")
    
    fossil_percentage = df_results['Fossil_Percentage'].iloc[0]
    payback = df_results['Payback_Period_Years'].iloc[0]
    roi = df_results['ROI_Percentage'].iloc[0]
    
    if fossil_percentage > 70:
        st.error(f"🚨 Critical Situation: {fossil_percentage:.1f}% fossil dependency")
        st.markdown("""
        ### Immediate Actions Required:
        1. **Emergency renewable deployment** - Fast-track solar and wind projects
        2. **Grid modernization** - Invest in smart grid infrastructure
        3. **Energy storage** - Deploy battery storage systems
        4. **Policy intervention** - Implement carbon pricing and subsidies
        """)
    elif fossil_percentage > 50:
        st.warning(f"⚠️ High Fossil Dependency: {fossil_percentage:.1f}%")
        st.markdown("""
        ### Priority Actions:
        1. **Accelerate renewables** - Target 30% renewable by 2030
        2. **Energy efficiency programs** - Building retrofits and industrial efficiency
        3. **Electric vehicle infrastructure** - Expand charging network
        4. **Hydrogen economy** - Develop green hydrogen production
        """)
    else:
        st.success(f"✅ Moderate Fossil Dependency: {fossil_percentage:.1f}%")
        st.markdown("""
        ### Continued Progress:
        1. **Grid flexibility** - Enhance demand response capabilities
        2. **Sector coupling** - Integrate power, heat, and transport
        3. **Digitalization** - Implement AI for grid optimization
        4. **International cooperation** - Cross-border energy exchange
        """)
    
    if payback < 5 and roi > 15:
        st.success(f"💰 Excellent Investment Opportunity: {roi:.1f}% ROI, {payback:.1f} year payback")
    elif payback < 10 and roi > 8:
        st.info(f"💰 Good Investment: {roi:.1f}% ROI, {payback:.1f} year payback")
    else:
        st.warning(f"💰 Challenging Investment: {roi:.1f}% ROI, {payback:.1f} year payback")

else:
    st.info("No analysis results found. Run the main analysis script first.")
    if st.button("Run Analysis"):
        import subprocess
        with st.spinner("Running analysis..."):
            result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
            st.code(result.stdout)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
Europe Energy Forecast Dashboard  
Version 1.0  
Data Source: ENTSO-E Transparency Platform  
Analysis Period: 2014-2020
""")
