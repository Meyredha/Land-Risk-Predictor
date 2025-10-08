#!/usr/bin/env python3
"""
Fix file locations for the Landslide Risk Prediction Application
"""

import os
import shutil

def fix_file_locations():
    """Fix file locations and organize project structure"""
    print("🔧 Fixing file locations...")
    
    # Check for nested data folder issue
    nested_csv = 'data/data/sample_landslide_data.csv'
    correct_csv = 'data/sample_landslide_data.csv'
    
    if os.path.exists(nested_csv) and not os.path.exists(correct_csv):
        print(f"📁 Moving {nested_csv} to {correct_csv}")
        shutil.move(nested_csv, correct_csv)
        print("✅ Dataset moved to correct location")
        
        # Remove empty nested data folder
        nested_dir = 'data/data'
        if os.path.exists(nested_dir) and not os.listdir(nested_dir):
            os.rmdir(nested_dir)
            print("✅ Removed empty nested directory")
    
    # Check if dataset exists in correct location
    if os.path.exists(correct_csv):
        file_size = os.path.getsize(correct_csv)
        print(f"✅ Dataset found at correct location: {correct_csv} ({file_size:,} bytes)")
    else:
        print(f"❌ Dataset not found at: {correct_csv}")
        
        # Look for it in other locations
        other_locations = [
            'sample_landslide_data.csv',
            'data/sample_landslide_data.csv',
            'data/data/sample_landslide_data.csv'
        ]
        
        for location in other_locations:
            if os.path.exists(location):
                print(f"📁 Found dataset at: {location}")
                if location != correct_csv:
                    shutil.copy2(location, correct_csv)
                    print(f"✅ Copied to correct location: {correct_csv}")
                break
        else:
            print("❌ Dataset not found anywhere. Please run data generation.")
            return False
    
    # Ensure all directories exist
    directories = ['data', 'ml_model', 'charts', 'app']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    return True

if __name__ == "__main__":
    print("🏔️ FILE LOCATION FIXER")
    print("=" * 30)
    
    if fix_file_locations():
        print("\n🎉 File locations fixed successfully!")
        print("You can now run: python ml_model/train_model.py")
    else:
        print("\n❌ Could not fix file locations")
