#!/usr/bin/env python3
"""
Fix encoding issues for CSV files
"""

import pandas as pd
import os
import sys

def fix_csv_encoding(input_path, output_path=None):
    """Fix CSV encoding issues"""
    
    if output_path is None:
        output_path = input_path
    
    print(f"Fixing encoding for: {input_path}")
    
    # Try different encodings
    encodings_to_try = [
        'utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 
        'iso-8859-1', 'windows-1252', 'ascii'
    ]
    
    df = None
    successful_encoding = None
    
    for encoding in encodings_to_try:
        try:
            print(f"Trying {encoding} encoding...")
            df = pd.read_csv(input_path, encoding=encoding)
            successful_encoding = encoding
            print(f"Success with {encoding} encoding!")
            break
        except (UnicodeDecodeError, UnicodeError) as e:
            print(f"Failed with {encoding}: {str(e)}")
            continue
        except Exception as e:
            print(f"Other error with {encoding}: {str(e)}")
            continue
    
    if df is None:
        print("Could not read the file with any encoding")
        return False
    
    # Save with UTF-8 encoding
    try:
        print(f"Saving fixed file to: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Verify the fixed file
        test_df = pd.read_csv(output_path, encoding='utf-8')
        print(f"File fixed successfully! Shape: {test_df.shape}")
        return True
        
    except Exception as e:
        print(f"Error saving fixed file: {str(e)}")
        return False

def main():
    """Main function to fix encoding issues"""
    print("CSV ENCODING FIXER")
    print("=" * 30)
    
    # Check if dataset exists and fix it
    dataset_path = 'data/sample_landslide_data.csv'
    
    if os.path.exists(dataset_path):
        print(f"Found dataset: {dataset_path}")
        if fix_csv_encoding(dataset_path):
            print("Encoding fixed successfully!")
        else:
            print("Could not fix encoding")
            return False
    else:
        print(f"Dataset not found: {dataset_path}")
        print("Please generate the dataset first")
        return False
    
    return True

if __name__ == "__main__":
    if main():
        print("All encoding issues fixed!")
    else:
        print("Failed to fix encoding issues")
        sys.exit(1)
