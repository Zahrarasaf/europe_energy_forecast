import subprocess
import sys

def install_requirements():
    """Install required packages for dashboard"""
    print("Installing dashboard requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "streamlit", "plotly"])

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("Starting Europe Energy Forecast Dashboard...")
    print("Dashboard will open in your browser at http://localhost:8501")
    subprocess.run(["streamlit", "run", "dashboard.py"])

if __name__ == "__main__":
    try:
        import streamlit
        import plotly
        run_dashboard()
    except ImportError:
        install_requirements()
        run_dashboard()
