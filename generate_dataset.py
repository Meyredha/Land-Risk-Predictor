import pandas as pd
import numpy as np
import os
import sys

def generate_landslide_dataset(n_samples=5000):
    """Generate synthetic landslide risk dataset"""
    print("🔄 Generating synthetic landslide dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Generate features with realistic ranges
        print("📊 Creating environmental features...")
        
        # Rainfall data (mm/year) - normal distribution around 150mm
        rainfall = np.random.normal(150, 50, n_samples)
        rainfall = np.clip(rainfall, 50, 400)  # Clip to realistic range
        
        # Soil moisture (%) - uniform distribution
        soil_moisture = np.random.uniform(10, 80, n_samples)
        
        # Elevation (meters) - uniform distribution
        elevation = np.random.uniform(0, 3000, n_samples)
        
        # Slope (degrees) - beta distribution for realistic slope values
        slope = np.random.beta(2, 5, n_samples) * 45  # Scale to 0-45 degrees
        
        # NDVI (Normalized Difference Vegetation Index) - beta distribution
        ndvi = np.random.beta(2, 2, n_samples) * 2 - 1  # Scale to -1 to 1
        
        # Land use types
        land_use_types = ['forest', 'agriculture', 'urban', 'water', 'barren']
        land_use = np.random.choice(land_use_types, n_samples, 
                                   p=[0.4, 0.3, 0.15, 0.1, 0.05])  # Weighted probabilities
        
        # GPS coordinates (example region)
        latitude = np.random.uniform(25.0, 35.0, n_samples)
        longitude = np.random.uniform(-125.0, -115.0, n_samples)
        
        print("✅ Environmental features created successfully")
        
        # Create DataFrame
        print("📋 Creating DataFrame...")
        data = {
            'rainfall': rainfall,
            'soil_moisture': soil_moisture,
            'elevation': elevation,
            'slope': slope,
            'ndvi': ndvi,
            'land_use': land_use,
            'latitude': latitude,
            'longitude': longitude
        }
        
        df = pd.DataFrame(data)
        
        # Encode land_use to numeric values
        print("🔢 Encoding categorical variables...")
        land_use_mapping = {
            'forest': 0, 
            'agriculture': 1, 
            'urban': 2, 
            'water': 3, 
            'barren': 4
        }
        df['land_use_encoded'] = df['land_use'].map(land_use_mapping)
        
        # Create risk labels based on environmental factors
        print("🎯 Calculating risk levels...")
        
        # Risk scoring based on multiple factors
        risk_scores = np.zeros(n_samples)
        
        # Rainfall contribution (higher rainfall = higher risk)
        risk_scores += np.where(df['rainfall'] > 200, 0.3, 0.0)
        risk_scores += np.where(df['rainfall'] > 250, 0.1, 0.0)
        
        # Soil moisture contribution (higher moisture = higher risk)
        risk_scores += np.where(df['soil_moisture'] > 60, 0.2, 0.0)
        risk_scores += np.where(df['soil_moisture'] > 70, 0.1, 0.0)
        
        # Slope contribution (steeper slope = higher risk)
        risk_scores += np.where(df['slope'] > 25, 0.2, 0.0)
        risk_scores += np.where(df['slope'] > 35, 0.2, 0.0)
        
        # Elevation contribution (higher elevation = moderate risk increase)
        risk_scores += np.where(df['elevation'] > 1500, 0.1, 0.0)
        risk_scores += np.where(df['elevation'] > 2500, 0.1, 0.0)
        
        # NDVI contribution (lower vegetation = higher risk)
        risk_scores += np.where(df['ndvi'] < 0.2, 0.15, 0.0)
        risk_scores += np.where(df['ndvi'] < 0.0, 0.1, 0.0)
        
        # Land use contribution
        land_use_risk = {
            'barren': 0.2,
            'urban': 0.1,
            'agriculture': 0.05,
            'water': 0.0,
            'forest': -0.1  # Forest reduces risk
        }
        
        for land_type, risk_value in land_use_risk.items():
            risk_scores += np.where(df['land_use'] == land_type, risk_value, 0.0)
        
        # Add some random variation
        risk_scores += np.random.normal(0, 0.08, n_samples)
        
        # Ensure scores are within bounds
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Convert to categorical risk levels
        risk_level = np.where(risk_scores < 0.35, 'Low',
                             np.where(risk_scores < 0.65, 'Medium', 'High'))
        
        df['risk_level'] = risk_level
        
        print("✅ Risk levels calculated successfully")
        
        # Determine the correct path to save the file
        # Try to save in the main data directory, not nested
        possible_paths = [
            'data/sample_landslide_data.csv',  # If running from root
            '../data/sample_landslide_data.csv',  # If running from data folder
            'sample_landslide_data.csv'  # If running from data folder
        ]
        
        saved = False
        for save_path in possible_paths:
            try:
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # Save the file
                df.to_csv(save_path, index=False)
                
                # Verify the file was created
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    print(f"💾 Dataset saved to: {save_path} ({file_size:,} bytes)")
                    saved = True
                    break
                    
            except Exception as e:
                print(f"⚠️  Could not save to {save_path}: {str(e)}")
                continue
        
        if not saved:
            # Fallback: save in current directory
            fallback_path = 'sample_landslide_data.csv'
            df.to_csv(fallback_path, index=False)
            print(f"💾 Dataset saved to fallback location: {fallback_path}")
        
        # Print statistics
        print(f"✅ Generated dataset with {n_samples:,} samples")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📈 Risk level distribution:")
        
        risk_counts = df['risk_level'].value_counts()
        for risk_level, count in risk_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {risk_level}: {count:,} ({percentage:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error during dataset generation: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    try:
        print("🏔️ LANDSLIDE DATASET GENERATOR")
        print("=" * 40)
        
        # Generate dataset
        df = generate_landslide_dataset(5000)
        
        print("\n🎉 Dataset generation completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
