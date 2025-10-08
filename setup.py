#!/usr/bin/env python3
"""
Setup script for Landslide Risk Prediction Application - Local Version
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False
    return True

def install_requirements():
    """Install required packages"""
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'plotly',
        'folium',
        'streamlit-folium',
        'joblib'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}")
            return False
    return True

def main():
    """Main setup function"""
    print("🏔️ Setting up Landslide Risk Prediction Application (Local Version)")
    
    # Create directories
    directories = ['data', 'ml_model', 'charts', 'app']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    if not install_requirements():
        print("❌ Failed to install some packages. Please install manually.")
        return
    
    # Generate dataset
    print("\n📊 Generating synthetic dataset...")
    try:
        exec(open('data/generate_dataset.py').read())
        print("✅ Dataset generated successfully")
    except Exception as e:
        print(f"❌ Failed to generate dataset: {e}")
        return
    
    # Train model
    print("\n🤖 Training machine learning model...")
    try:
        exec(open('ml_model/train_model.py').read())
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To run the application, execute:")
    print("python run.py")

if __name__ == "__main__":
    main()
