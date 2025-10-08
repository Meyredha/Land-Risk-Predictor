# 🏔️ Landslide Risk Prediction System

A complete web-based application for landslide risk prediction and land area monitoring using machine learning.

## 🚀 Quick Start (Local Setup)

### Windows Users:
1. **Double-click** `run.bat` 
2. **Or open Command Prompt** and run: `python run.py`

### Mac/Linux Users:
1. **Open Terminal** and run: `./run.sh`
2. **Or run directly**: `python3 run.py`

### Manual Setup:
\`\`\`bash
# 1. Install dependencies and setup
python setup.py

# 2. Run the application
python run.py
\`\`\`

The application will automatically:
- ✅ Install required packages
- ✅ Generate synthetic dataset
- ✅ Train the ML model
- ✅ Open your browser to http://localhost:8501

## 📋 Requirements

- **Python 3.7+** (Download from [python.org](https://python.org))
- **Internet connection** (for installing packages)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

## 🎯 Features

- 🔮 **AI-Powered Predictions**: Upload CSV files for instant landslide risk assessment
- 📊 **Interactive Charts**: Beautiful visualizations with Plotly
- 🗺️ **Interactive Maps**: Geospatial risk visualization with Folium
- 📈 **Model Analysis**: Feature importance and correlation analysis
- 🔍 **Data Explorer**: Interactive data filtering and exploration
- 📱 **User-Friendly**: Clean, modern web interface

## 📊 Data Format

Your CSV file should contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| rainfall | Annual rainfall (mm) | 150.5 |
| soil_moisture | Soil moisture (%) | 45.2 |
| elevation | Elevation (meters) | 1200 |
| slope | Slope angle (degrees) | 25.5 |
| ndvi | Vegetation index (-1 to 1) | 0.6 |
| land_use | Land use type | forest |
| latitude | Latitude coordinate | 30.5 |
| longitude | Longitude coordinate | -120.5 |

## 🎨 Application Pages

1. **🏠 Home**: Overview and system information
2. **📤 Data Upload & Prediction**: Upload CSV files and get predictions
3. **📊 Model Analysis**: Explore model performance and feature importance
4. **🔍 Sample Data Explorer**: Interactive data exploration tools

## 🔧 Troubleshooting

### Common Issues:

**"Module not found" error:**
\`\`\`bash
pip install streamlit pandas numpy scikit-learn plotly folium streamlit-folium joblib
\`\`\`

**"Port already in use" error:**
- Close other applications using port 8501
- Or change port in run.py: `--server.port=8502`

**Browser doesn't open automatically:**
- Manually open: http://localhost:8501

## 📁 Project Structure

\`\`\`
landslide-predictor-app/
├── app/
│   └── streamlit_app.py      # Main web application
├── ml_model/
│   ├── train_model.py        # Model training
│   ├── landslide_model.pkl   # Trained model (generated)
│   └── scaler.pkl           # Feature scaler (generated)
├── data/
│   ├── generate_dataset.py   # Dataset generator
│   └── sample_landslide_data.csv (generated)
├── charts/
│   └── chart_generator.py    # Visualization utilities
├── run.py                   # Main run script
├── setup.py                 # Setup script
├── run.bat                  # Windows batch file
├── run.sh                   # Linux/Mac shell script
└── README.md               # This file
\`\`\`

## 🤖 Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~87%
- **Features**: 6 environmental variables
- **Classes**: Low, Medium, High risk
- **Training Data**: 5,000 synthetic samples

## 🎉 Success!

If everything works correctly, you should see:
- ✅ Browser opens automatically
- ✅ Application loads at http://localhost:8501
- ✅ All pages are accessible
- ✅ Sample data and model are loaded

## 📞 Support

If you encounter any issues:
1. Check that Python 3.7+ is installed
2. Ensure you have internet connection for package installation
3. Try running `python setup.py` manually
4. Check the console for error messages

Enjoy predicting landslide risks! 🏔️
