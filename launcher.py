#!/usr/bin/env python3
"""
Simple launcher for the Landslide Risk Prediction Application
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading

def check_setup():
    """Check if setup is complete"""
    required_files = [
        'data/sample_landslide_data.csv',
        'ml_model/landslide_model.pkl',
        'app/streamlit_app.py'
    ]
    
    return all(os.path.exists(f) for f in required_files)

def open_browser_delayed():
    """Open browser after delay"""
    time.sleep(4)
    webbrowser.open('http://localhost:8501')

def main():
    print("🏔️ LANDSLIDE RISK PREDICTION SYSTEM")
    print("=" * 50)
    
    # Check if setup is needed
    if not check_setup():
        print("⚠️  Setup required. Running complete setup...")
        try:
            subprocess.run([sys.executable, "complete_setup.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Setup failed")
            return
    
    print("🚀 Starting application...")
    print("🌐 Opening browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    
    # Start browser opener in background
    browser_thread = threading.Thread(target=open_browser_delayed)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/streamlit_app.py",
            "--server.port=8501",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please run: pip install streamlit")

if __name__ == "__main__":
    main()
