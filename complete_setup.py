#!/usr/bin/env python3
"""
Complete setup script for Landslide Risk Prediction Application
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python from https://python.org")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        print("   Please install pip first")
        return False

def install_dependencies():
    """Install all required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Try to run the dependency installer
    try:
        result = subprocess.run([sys.executable, "install_dependencies.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("⚠️  Some dependencies failed to install")
            print("Trying alternative installation method...")
            
            # Fallback: install essential packages only
            essential_packages = [
                "streamlit", "pandas", "numpy", "scikit-learn", 
                "plotly", "folium", "streamlit-folium", "joblib"
            ]
            
            for package in essential_packages:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, capture_output=True)
                    print(f"✅ {package} installed")
                except subprocess.CalledProcessError:
                    print(f"❌ Failed to install {package}")
                    return False
            
            return True
            
    except FileNotFoundError:
        print("⚠️  install_dependencies.py not found, installing manually...")
        
        # Manual installation
        packages = [
            "streamlit>=1.28.0", "pandas>=2.0.0", "numpy>=1.24.0", 
            "scikit-learn>=1.3.0", "plotly>=5.15.0", "folium>=0.14.0",
            "streamlit-folium>=0.13.0", "joblib>=1.3.0"
        ]
        
        for package in packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                print(f"✅ {package.split('>=')[0]} installed")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
        
        return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['data', 'ml_model', 'charts', 'app']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")

def generate_data():
    """Generate synthetic dataset"""
    print("\n📊 Generating synthetic dataset...")
    
    try:
        # Import and run data generation
        sys.path.append('data')
        from generate_dataset import generate_landslide_dataset
        generate_landslide_dataset()
        print("✅ Dataset generated successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to generate dataset: {e}")
        return False

def train_model():
    """Train the machine learning model"""
    print("\n🤖 Training machine learning model...")
    
    try:
        # Import and run model training
        sys.path.append('ml_model')
        from train_model import train_landslide_model
        train_landslide_model()
        print("✅ Model trained successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        return False

def verify_installation():
    """Verify that all components are working"""
    print("\n🔍 Verifying installation...")
    
    # Check required files
    required_files = [
        'data/sample_landslide_data.csv',
        'ml_model/landslide_model.pkl',
        'ml_model/scaler.pkl',
        'app/streamlit_app.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {len(missing_files)}")
        return False
    
    # Test import of key modules
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import plotly
        import folium
        print("✅ All modules can be imported")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🏔️ LANDSLIDE RISK PREDICTION SYSTEM - COMPLETE SETUP")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Check pip
    if not check_pip():
        return False
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        return False
    
    # Step 5: Generate data
    if not generate_data():
        print("\n❌ Failed to generate data")
        return False
    
    # Step 6: Train model
    if not train_model():
        print("\n❌ Failed to train model")
        return False
    
    # Step 7: Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n🚀 To run the application:")
    print("   python run.py")
    print("\n🌐 The app will open at:")
    print("   http://localhost:8501")
    print("\n📱 Browser will open automatically!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
