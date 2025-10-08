#!/usr/bin/env python3
"""
Clean dataset generator that avoids encoding issues
"""

import pandas as pd
import numpy as np
import os
import sys

def create_clean_dataset():
    """Create a clean dataset without encoding issues"""
    print("Creating clean landslide dataset...")
    
    # Set random seed
    np.random.seed(42)
    n_samples = 5000
    
    try:
        # Generate all numeric data first
        print("Generating numeric features...")
        
        data = {}
        
        # Environmental features
        data['rainfall'] = np.round(np.clip(np.random.normal(150, 50, n_samples), 50, 400), 2)
        data['soil_moisture'] = np.round(np.random.uniform(10, 80, n_samples), 2)
        data['elevation'] = np.round(np.random.uniform(0, 3000, n_samples), 1)
        data['slope'] = np.round(np.random.beta(2, 5, n_samples) * 45, 2)
        data['ndvi'] = np.round(np.random.beta(2, 2, n_samples) * 2 - 1, 3)
        
        # Location data
        data['latitude'] = np.round(np.random.uniform(25.0, 35.0, n_samples), 6)
        data['longitude'] = np.round(np.random.uniform(-125.0, -115.0, n_samples), 6)
        
        # Land use (encoded directly as numbers to avoid string issues)
        data['land_use_encoded'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
        # Create land use text based on encoded values
        land_use_map = {0: 'forest', 1: 'agriculture', 2: 'urban', 3: 'water', 4: 'barren'}
        data['land_use'] = [land_use_map[code] for code in data['land_use_encoded']]
        
        print("Calculating risk levels...")
        
        # Calculate risk scores
        risk_scores = np.zeros(n_samples)
        
        # Add risk factors
        risk_scores += np.where(data['rainfall'] > 200, 0.3, 0.0)
        risk_scores += np.where(data['rainfall'] > 250, 0.1, 0.0)
        risk_scores += np.where(data['soil_moisture'] > 60, 0.2, 0.0)
        risk_scores += np.where(data['soil_moisture'] > 70, 0.1, 0.0)
        risk_scores += np.where(data['slope'] > 25, 0.2, 0.0)
        risk_scores += np.where(data['slope'] > 35, 0.2, 0.0)
        risk_scores += np.where(data['elevation'] > 1500, 0.1, 0.0)
        risk_scores += np.where(data['elevation'] > 2500, 0.1, 0.0)
        risk_scores += np.where(data['ndvi'] < 0.2, 0.15, 0.0)
        risk_scores += np.where(data['ndvi'] < 0.0, 0.1, 0.0)
        
        # Land use risk
        land_use_risk_values = [0.2, 0.1, 0.05, 0.0, -0.1]  # barren, urban, agriculture, water, forest
        for i, risk_val in enumerate(land_use_risk_values):
            risk_scores += np.where(data['land_use_encoded'] == i, risk_val, 0.0)
        
        # Add random variation
        risk_scores += np.random.normal(0, 0.08, n_samples)
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Convert to risk levels
        data['risk_level'] = ['Low' if score < 0.35 else 'Medium' if score < 0.65 else 'High' for score in risk_scores]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        print(f"Dataset created with {len(df)} samples")
        
        # Ensure directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save with explicit parameters to avoid encoding issues
        output_path = 'data/sample_landslide_data.csv'
        
        print(f"Saving to: {output_path}")
        
        # Use simple ASCII-safe saving
        df.to_csv(output_path, index=False, encoding='utf-8', lineterminator='\n')
        
        # Verify file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"File saved successfully: {file_size:,} bytes")
            
            # Test reading
            test_df = pd.read_csv(output_path)
            print(f"Verification successful: {test_df.shape}")
            
            # Print distribution
            print("Risk distribution:")
            risk_counts = test_df['risk_level'].value_counts()
            for risk, count in risk_counts.items():
                pct = (count / len(test_df)) * 100
                print(f"  {risk}: {count:,} ({pct:.1f}%)")
            
            return True
        else:
            print("File was not created")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("CLEAN DATASET GENERATOR")
    print("=" * 30)
    
    if create_clean_dataset():
        print("Dataset created successfully!")
    else:
        print("Failed to create dataset")
        sys.exit(1)
