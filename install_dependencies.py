#!/usr/bin/env python3
"""
Dependency installer for Landslide Risk Prediction Application
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install all required packages"""
    print("🏔️ Installing Dependencies for Landslide Risk Prediction System")
    print("=" * 60)
    
    # List of required packages
    packages = [
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "folium>=0.14.0",
        "streamlit-folium>=0.13.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0"
    ]
    
    print(f"📦 Installing {len(packages)} packages...")
    print()
    
    success_count = 0
    failed_packages = []
    
    for i, package in enumerate(packages, 1):
        package_name = package.split(">=")[0]
        print(f"[{i}/{len(packages)}] Installing {package_name}...", end=" ")
        
        if install_package(package):
            print("✅ Success")
            success_count += 1
        else:
            print("❌ Failed")
            failed_packages.append(package_name)
    
    print()
    print("=" * 60)
    print(f"📊 Installation Summary:")
    print(f"✅ Successful: {success_count}/{len(packages)}")
    print(f"❌ Failed: {len(failed_packages)}/{len(packages)}")
    
    if failed_packages:
        print(f"\n⚠️  Failed packages: {', '.join(failed_packages)}")
        print("💡 Try installing them manually:")
        for pkg in failed_packages:
            print(f"   pip install {pkg}")
        return False
    else:
        print("\n🎉 All dependencies installed successfully!")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
