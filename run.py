#!/usr/bin/env python3
"""
Run script for the Landslide Risk Prediction Application - Local Version
"""

import subprocess
import sys
import os
import webbrowser
import time
import threading

def check_files():
    """Check if required files exist"""
    required_files = [
        'data/sample_landslide_data.csv',
        'ml_model/landslide_model.pkl',
        'ml_model/scaler.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def open_browser():
    """Open browser after a delay"""
    time.sleep(3)  # Wait for Streamlit to start
    webbrowser.open('http://localhost:8501')

def main():
    """Main run function"""
    print("🏔️ Starting Landslide Risk Prediction Application")
    
    # Check if setup is needed
    missing_files = check_files()
    
    if missing_files:
        print("⚠️  Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        
        print("\nRunning setup first...")
        try:
            subprocess.run([sys.executable, "setup.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Setup failed. Please run 'python setup.py' manually.")
            sys.exit(1)
    
    # Start browser opening in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the Streamlit app
    print("🚀 Launching Streamlit application...")
    print("📱 Opening browser automatically...")
    print("🌐 Application will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start Streamlit application.")
        print("💡 Try installing streamlit: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")

if __name__ == "__main__":
    main()
