import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
import sys

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from charts.chart_generator import (
        create_risk_distribution_charts,
        create_feature_importance_chart,
        create_correlation_heatmap,
        create_risk_map,
        create_feature_distribution_charts
    )
except ImportError:
    st.error("Chart generator module not found. Please ensure all files are in the correct location.")

# Page configuration
st.set_page_config(
    page_title="Landslide Risk Predictor",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('ml_model/landslide_model.pkl')
        scaler = joblib.load('ml_model/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run setup first by executing: python setup.py")
        return None, None

@st.cache_data
def load_sample_data():
    """Load sample dataset"""
    try:
        return pd.read_csv('data/sample_landslide_data.csv')
    except FileNotFoundError:
        st.error("❌ Sample data not found. Please run setup first by executing: python setup.py")
        return None

def preprocess_data(df):
    """Preprocess uploaded data"""
    # Create land_use_encoded if it doesn't exist
    if 'land_use' in df.columns and 'land_use_encoded' not in df.columns:
        land_use_mapping = {'forest': 0, 'agriculture': 1, 'urban': 2, 'water': 3, 'barren': 4}
        df['land_use_encoded'] = df['land_use'].map(land_use_mapping)
        
        # Handle unknown land use types
        df['land_use_encoded'] = df['land_use_encoded'].fillna(0)
    
    return df

def make_predictions(model, scaler, df):
    """Make predictions on the dataset"""
    feature_columns = ['rainfall', 'soil_moisture', 'elevation', 'slope', 'ndvi', 'land_use_encoded']
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        return None
    
    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Add predictions to dataframe
    df_pred = df.copy()
    df_pred['predicted_risk'] = predictions
    df_pred['risk_probability'] = probabilities.max(axis=1)
    
    return df_pred

def main():
    # Header
    st.markdown('<h1 class="main-header">🏔️ Landslide Risk Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["🏠 Home", "📤 Data Upload & Prediction", "📊 Model Analysis", "🔍 Sample Data Explorer"],
        format_func=lambda x: x
    )
    
    # Load model and data
    model, scaler = load_model_and_scaler()
    sample_data = load_sample_data()
    
    if page == "🏠 Home":
        show_home_page(sample_data)
    elif page == "📤 Data Upload & Prediction":
        show_prediction_page(model, scaler)
    elif page == "📊 Model Analysis":
        show_analysis_page(sample_data)
    elif page == "🔍 Sample Data Explorer":
        show_data_explorer_page(sample_data)

def show_home_page(sample_data):
    """Display home page"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🌟 Welcome to the Landslide Risk Prediction System
        
        This application uses **machine learning** to predict landslide risk based on environmental factors.
        Our Random Forest model analyzes multiple environmental parameters to assess landslide probability.
        
        ### 🚀 Key Features:
        - 🔮 **AI-Powered Predictions**: Upload your data and get instant landslide risk assessments
        - 📊 **Interactive Visualizations**: Beautiful charts, graphs, and correlation analysis
        - 🗺️ **Geospatial Mapping**: View risk zones on interactive maps
        - 📈 **Model Insights**: Explore feature importance and model performance
        - 📱 **User-Friendly Interface**: Easy-to-use web interface
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Input Features:
        - 🌧️ **Rainfall** (mm/year)
        - 💧 **Soil Moisture** (%)
        - ⛰️ **Elevation** (meters)
        - 📐 **Slope** (degrees)
        - 🌿 **NDVI** (Vegetation Index)
        - 🏞️ **Land Use Type**
        - 📍 **GPS Coordinates**
        """)
    
    st.markdown("---")
    
    # Risk categories explanation
    st.subheader("🎯 Risk Categories")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #d4edda; padding: 1rem; border-radius: 10px; border-left: 5px solid green;">
        <h4 style="color: green;">🟢 Low Risk</h4>
        <p>Minimal landslide probability. Safe for most activities and development.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 5px solid orange;">
        <h4 style="color: orange;">🟡 Medium Risk</h4>
        <p>Moderate landslide probability. Requires monitoring and precautionary measures.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #f8d7da; padding: 1rem; border-radius: 10px; border-left: 5px solid red;">
        <h4 style="color: red;">🔴 High Risk</h4>
        <p>High landslide probability. Immediate attention and safety measures required.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display sample statistics
    if sample_data is not None:
        st.markdown("---")
        st.subheader("📊 Sample Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Samples", f"{len(sample_data):,}")
        with col2:
            high_risk = len(sample_data[sample_data['risk_level'] == 'High'])
            st.metric("🔴 High Risk Zones", f"{high_risk:,}")
        with col3:
            medium_risk = len(sample_data[sample_data['risk_level'] == 'Medium'])
            st.metric("🟡 Medium Risk Zones", f"{medium_risk:,}")
        with col4:
            low_risk = len(sample_data[sample_data['risk_level'] == 'Low'])
            st.metric("🟢 Low Risk Zones", f"{low_risk:,}")
        
        # Quick visualization
        st.subheader("📊 Quick Overview")
        risk_counts = sample_data['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Distribution in Sample Data",
            color=risk_counts.index,
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model, scaler):
    """Display prediction page"""
    st.header("📤 Data Upload & Prediction")
    
    if model is None:
        st.error("❌ Model not loaded. Please run setup first: `python setup.py`")
        return
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions:
    1. **Upload your CSV file** with environmental data
    2. **Preview your data** to ensure it's formatted correctly
    3. **Click 'Make Predictions'** to get landslide risk assessments
    4. **Explore results** with interactive charts and maps
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload your CSV file with environmental data",
        type=['csv'],
        help="File should contain columns: rainfall, soil_moisture, elevation, slope, ndvi, land_use, latitude, longitude"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show data preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data info
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📊 Data Info:**")
                st.write(f"- Rows: {df.shape[0]:,}")
                st.write(f"- Columns: {df.shape[1]}")
                st.write(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            with col2:
                st.write("**📋 Column Names:**")
                for col in df.columns:
                    st.write(f"- {col}")
            
            # Preprocess data
            df_processed = preprocess_data(df)
            
            # Make predictions button
            if st.button("🔮 Make Predictions", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your data..."):
                    predictions_df = make_predictions(model, scaler, df_processed)
                
                if predictions_df is not None:
                    st.balloons()
                    st.success("🎉 Predictions completed successfully!")
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    risk_counts = predictions_df['predicted_risk'].value_counts()
                    total_samples = len(predictions_df)
                    
                    with col1:
                        st.metric("📈 Total Zones", f"{total_samples:,}")
                    with col2:
                        high_risk = risk_counts.get('High', 0)
                        st.metric("🔴 High Risk", f"{high_risk:,}", f"{high_risk/total_samples*100:.1f}%")
                    with col3:
                        medium_risk = risk_counts.get('Medium', 0)
                        st.metric("🟡 Medium Risk", f"{medium_risk:,}", f"{medium_risk/total_samples*100:.1f}%")
                    with col4:
                        low_risk = risk_counts.get('Low', 0)
                        st.metric("🟢 Low Risk", f"{low_risk:,}", f"{low_risk/total_samples*100:.1f}%")
                    
                    # Charts
                    st.subheader("📊 Risk Distribution Charts")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart
                        bar_fig, _ = create_risk_distribution_charts(predictions_df.rename(columns={'predicted_risk': 'risk_level'}))
                        st.plotly_chart(bar_fig, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        _, pie_fig = create_risk_distribution_charts(predictions_df.rename(columns={'predicted_risk': 'risk_level'}))
                        st.plotly_chart(pie_fig, use_container_width=True)
                    
                    # Map visualization
                    if 'latitude' in predictions_df.columns and 'longitude' in predictions_df.columns:
                        st.subheader("🗺️ Interactive Risk Map")
                        
                        # Create map
                        risk_map = create_risk_map(predictions_df.rename(columns={'predicted_risk': 'risk_level'}))
                        st_folium(risk_map, width=700, height=500)
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="landslide_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Detailed results table
                    with st.expander("📋 View Detailed Results Table"):
                        st.dataframe(predictions_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please ensure your CSV file has the correct format and column names.")
    
    else:
        # Show sample data format
        st.info("📝 Please upload a CSV file to make predictions.")
        
        st.subheader("📋 Expected Data Format")
        st.markdown("Your CSV file should contain the following columns:")
        
        sample_format = pd.DataFrame({
            'rainfall': [150.5, 200.3, 120.1],
            'soil_moisture': [45.2, 65.8, 30.5],
            'elevation': [1200, 800, 2000],
            'slope': [25.5, 35.2, 15.8],
            'ndvi': [0.6, 0.3, 0.8],
            'land_use': ['forest', 'agriculture', 'urban'],
            'latitude': [30.5, 31.2, 29.8],
            'longitude': [-120.5, -119.8, -121.2]
        })
        st.dataframe(sample_format, use_container_width=True)
        
        # Download sample format
        csv_sample = sample_format.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Format",
            data=csv_sample,
            file_name="sample_format.csv",
            mime="text/csv"
        )

def show_analysis_page(sample_data):
    """Display model analysis page"""
    st.header("📊 Model Analysis & Performance")
    
    if sample_data is None:
        st.error("❌ Sample data not available. Please run setup first.")
        return
    
    # Model performance metrics
    st.subheader("🎯 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Accuracy", "~87%", "High")
    with col2:
        st.metric("🤖 Algorithm", "Random Forest", "100 trees")
    with col3:
        st.metric("📊 Features", "6", "Environmental")
    
    # Feature importance
    try:
        feature_importance = pd.read_csv('ml_model/feature_importance.csv')
        st.subheader("🔍 Feature Importance Analysis")
        st.markdown("This chart shows which environmental factors are most important for predicting landslide risk:")
        
        fig = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        with st.expander("📋 View Feature Importance Table"):
            st.dataframe(feature_importance, use_container_width=True)
            
    except FileNotFoundError:
        st.warning("⚠️ Feature importance data not found.")
    
    # Correlation analysis
    st.subheader("🔗 Feature Correlation Analysis")
    st.markdown("This heatmap shows how environmental factors relate to each other:")
    
    corr_fig = create_correlation_heatmap(sample_data)
    st.plotly_chart(corr_fig, use_container_width=True)
    
    # Risk distribution in sample data
    st.subheader("📊 Sample Data Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bar_fig, _ = create_risk_distribution_charts(sample_data)
        st.plotly_chart(bar_fig, use_container_width=True)
    
    with col2:
        _, pie_fig = create_risk_distribution_charts(sample_data)
        st.plotly_chart(pie_fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("📈 Feature Distribution Analysis")
    st.markdown("These histograms show the distribution of each environmental factor:")
    
    dist_fig = create_feature_distribution_charts(sample_data)
    st.plotly_chart(dist_fig, use_container_width=True)

def show_data_explorer_page(sample_data):
    """Display data explorer page"""
    st.header("🔍 Sample Data Explorer")
    
    if sample_data is None:
        st.error("❌ Sample data not available. Please run setup first.")
        return
    
    # Data overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Dataset Statistics:**")
        st.write(f"- **Total Samples:** {sample_data.shape[0]:,}")
        st.write(f"- **Features:** {sample_data.shape[1]}")
        st.write(f"- **Memory Usage:** {sample_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.markdown("**🎯 Risk Level Distribution:**")
        risk_dist = sample_data['risk_level'].value_counts()
        for risk, count in risk_dist.items():
            percentage = (count / len(sample_data)) * 100
            st.write(f"- **{risk}:** {count:,} ({percentage:.1f}%)")
    
    with col2:
        st.markdown("**📊 Statistical Summary:**")
        st.dataframe(sample_data.describe(), use_container_width=True)
    
    # Interactive filters
    st.subheader("🎛️ Interactive Data Filtering")
    st.markdown("Use these filters to explore different subsets of the data:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "🎯 Filter by Risk Level",
            options=sample_data['risk_level'].unique(),
            default=sample_data['risk_level'].unique()
        )
    
    with col2:
        land_use_filter = st.multiselect(
            "🏞️ Filter by Land Use",
            options=sample_data['land_use'].unique(),
            default=sample_data['land_use'].unique()
        )
    
    with col3:
        slope_range = st.slider(
            "📐 Slope Range (degrees)",
            min_value=float(sample_data['slope'].min()),
            max_value=float(sample_data['slope'].max()),
            value=(float(sample_data['slope'].min()), float(sample_data['slope'].max())),
            step=1.0
        )
    
    # Apply filters
    filtered_data = sample_data[
        (sample_data['risk_level'].isin(risk_filter)) &
        (sample_data['land_use'].isin(land_use_filter)) &
        (sample_data['slope'] >= slope_range[0]) &
        (sample_data['slope'] <= slope_range[1])
    ]
    
    st.success(f"✅ Filtered data: {filtered_data.shape[0]:,} samples (from {sample_data.shape[0]:,} total)")
    
    # Display filtered data
    st.subheader("📋 Filtered Data Preview")
    st.dataframe(filtered_data.head(20), use_container_width=True)
    
    # Download filtered data
    if len(filtered_data) > 0:
        csv_filtered = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv_filtered,
            file_name="filtered_landslide_data.csv",
            mime="text/csv"
        )
    
    # Interactive scatter plots
    st.subheader("📊 Interactive Feature Analysis")
    st.markdown("Create custom scatter plots to explore relationships between features:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("📊 X-axis", ['rainfall', 'soil_moisture', 'elevation', 'slope', 'ndvi'])
        y_axis = st.selectbox("📊 Y-axis", ['rainfall', 'soil_moisture', 'elevation', 'slope', 'ndvi'], index=1)
    
    with col2:
        color_by = st.selectbox("🎨 Color by", ['risk_level', 'land_use'])
        size_by = st.selectbox("📏 Size by", ['None'] + ['rainfall', 'soil_moisture', 'elevation', 'slope', 'ndvi'])
    
    if x_axis != y_axis and len(filtered_data) > 0:
        # Create scatter plot
        scatter_fig = px.scatter(
            filtered_data.sample(n=min(1000, len(filtered_data))),  # Sample for performance
            x=x_axis,
            y=y_axis,
            color=color_by,
            size=size_by if size_by != 'None' else None,
            title=f"{y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}",
            hover_data=['risk_level', 'land_use', 'slope', 'elevation'],
            height=500
        )
        scatter_fig.update_layout(
            xaxis_title=x_axis.replace('_', ' ').title(),
            yaxis_title=y_axis.replace('_', ' ').title()
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

if __name__ == "__main__":
    main()
