import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

def find_dataset():
    """Find the dataset file in possible locations"""
    possible_paths = [
        'data/sample_landslide_data.csv',
        'data/data/sample_landslide_data.csv',
        '../data/sample_landslide_data.csv',
        '../data/data/sample_landslide_data.csv',
        './sample_landslide_data.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found dataset at: {path}")
            return path
    
    return None

def train_landslide_model():
    """Train and save the landslide prediction model"""
    print("🤖 Training landslide prediction model...")
    
    # Find the dataset
    dataset_path = find_dataset()
    
    if dataset_path is None:
        print("❌ Dataset not found in any of these locations:")
        print("   - data/sample_landslide_data.csv")
        print("   - data/data/sample_landslide_data.csv")
        print("   - ../data/sample_landslide_data.csv")
        print("   - ../data/data/sample_landslide_data.csv")
        print("   - ./sample_landslide_data.csv")
        print("\n💡 Please run data generation first:")
        print("   python data/generate_dataset.py")
        return None
    
    try:
        # Load data
        print(f"📊 Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
        
        # Display basic info about the dataset
        print(f"📈 Dataset info:")
        print(f"   - Total samples: {len(df):,}")
        print(f"   - Features: {df.shape[1]}")
        print(f"   - Risk distribution:")
        
        risk_counts = df['risk_level'].value_counts()
        for risk, count in risk_counts.items():
            percentage = (count / len(df)) * 100
            print(f"     {risk}: {count:,} ({percentage:.1f}%)")
        
        # Check required columns
        required_columns = ['rainfall', 'soil_moisture', 'elevation', 'slope', 'ndvi', 'land_use_encoded', 'risk_level']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        print("✅ All required columns found")
        
        # Prepare features
        feature_columns = ['rainfall', 'soil_moisture', 'elevation', 'slope', 'ndvi', 'land_use_encoded']
        X = df[feature_columns]
        y = df['risk_level']
        
        print(f"🔢 Feature matrix shape: {X.shape}")
        print(f"🎯 Target vector shape: {y.shape}")
        
        # Check for missing values
        missing_values = X.isnull().sum()
        if missing_values.sum() > 0:
            print("⚠️  Found missing values:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"   {col}: {count}")
            
            # Fill missing values
            X = X.fillna(X.mean())
            print("✅ Missing values filled with column means")
        
        # Split data
        print("🔄 Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]:,} samples")
        print(f"📊 Test set: {X_test.shape[0]:,} samples")
        
        # Scale features
        print("⚖️  Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("✅ Feature scaling completed")
        
        # Train model
        print("🤖 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train_scaled, y_train)
        print("✅ Model training completed")
        
        # Make predictions
        print("🔮 Making predictions on test set...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 MODEL PERFORMANCE:")
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n🔢 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance
        print(f"\n🔍 Feature Importance Analysis:")
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.to_string(index=False))
        
        # Model statistics
        print(f"\n📈 Model Statistics:")
        print(f"   - Number of trees: {model.n_estimators}")
        print(f"   - Max depth: {model.max_depth}")
        print(f"   - Features used: {len(feature_columns)}")
        print(f"   - Classes: {list(model.classes_)}")
        
        # Ensure ml_model directory exists
        os.makedirs('ml_model', exist_ok=True)
        
        # Save model and scaler
        print(f"\n💾 Saving model and scaler...")
        
        model_path = 'ml_model/landslide_model.pkl'
        scaler_path = 'ml_model/scaler.pkl'
        feature_importance_path = 'ml_model/feature_importance.csv'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        feature_importance.to_csv(feature_importance_path, index=False)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Scaler saved to: {scaler_path}")
        print(f"✅ Feature importance saved to: {feature_importance_path}")
        
        # Verify saved files
        print(f"\n🔍 Verifying saved files...")
        saved_files = [model_path, scaler_path, feature_importance_path]
        
        for file_path in saved_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ {file_path} ({file_size:,} bytes)")
            else:
                print(f"❌ {file_path} not found")
        
        print(f"\n🎉 Model training completed successfully!")
        
        return model, scaler, accuracy, feature_importance
        
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def test_model():
    """Test the saved model"""
    try:
        print(f"\n🧪 Testing saved model...")
        
        # Load model and scaler
        model = joblib.load('ml_model/landslide_model.pkl')
        scaler = joblib.load('ml_model/scaler.pkl')
        
        print("✅ Model and scaler loaded successfully")
        
        # Create test data
        test_data = pd.DataFrame({
            'rainfall': [200.0],
            'soil_moisture': [65.0],
            'elevation': [1500.0],
            'slope': [30.0],
            'ndvi': [0.3],
            'land_use_encoded': [1]
        })
        
        # Make prediction
        test_scaled = scaler.transform(test_data)
        prediction = model.predict(test_scaled)
        probability = model.predict_proba(test_scaled)
        
        print(f"🔮 Test prediction: {prediction[0]}")
        print(f"📊 Prediction probabilities:")
        for i, class_name in enumerate(model.classes_):
            print(f"   {class_name}: {probability[0][i]:.3f}")
        
        print("✅ Model test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        print("🏔️ LANDSLIDE RISK PREDICTION - MODEL TRAINING")
        print("=" * 50)
        
        # Train model
        result = train_landslide_model()
        
        if result is not None:
            # Test the saved model
            if test_model():
                print(f"\n🎉 ALL OPERATIONS COMPLETED SUCCESSFULLY!")
                print("=" * 50)
                print("🚀 You can now run the web application:")
                print("   python run.py")
            else:
                print(f"\n⚠️  Model training completed but testing failed")
        else:
            print(f"\n❌ Model training failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)