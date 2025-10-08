import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import folium
import numpy as np

def create_risk_distribution_charts(df):
    """Create bar and pie charts for risk distribution"""
    
    # Count risk levels
    risk_counts = df['risk_level'].value_counts()
    
    # Bar chart
    bar_fig = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        title="Landslide Risk Distribution (Count)",
        labels={'x': 'Risk Level', 'y': 'Number of Zones'},
        color=risk_counts.index,
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    )
    bar_fig.update_layout(showlegend=False, height=400)
    
    # Pie chart
    pie_fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Landslide Risk Distribution (Proportion)",
        color=risk_counts.index,
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    )
    pie_fig.update_layout(height=400)
    
    return bar_fig, pie_fig

def create_feature_importance_chart(feature_importance_df):
    """Create feature importance chart"""
    
    fig = px.bar(
        feature_importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance in Landslide Prediction",
        labels={'importance': 'Importance Score', 'feature': 'Features'},
        color='importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    
    numeric_cols = ['rainfall', 'soil_moisture', 'elevation', 'slope', 'ndvi']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale='RdBu',
        aspect='auto',
        text_auto=True
    )
    fig.update_layout(height=500)
    
    return fig

def create_risk_map(df):
    """Create folium map with risk zones"""
    
    # Create base map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Color mapping for risk levels
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    
    # Sample data for better performance (max 1000 points)
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df
    
    # Add markers for each zone
    for idx, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=f"Risk: {row['risk_level']}<br>Slope: {row['slope']:.1f}°<br>Rainfall: {row['rainfall']:.1f}mm",
            color=color_map[row['risk_level']],
            fill=True,
            fillColor=color_map[row['risk_level']],
            fillOpacity=0.7
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Risk Levels</b></p>
    <p><span style="color:red">●</span> High Risk</p>
    <p><span style="color:orange">●</span> Medium Risk</p>
    <p><span style="color:green">●</span> Low Risk</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_feature_distribution_charts(df):
    """Create distribution charts for features"""
    
    numeric_features = ['rainfall', 'soil_moisture', 'elevation', 'slope', 'ndvi']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=numeric_features,
        specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "histogram"}, {"type": "xy"}]]
    )
    
    for i, feature in enumerate(numeric_features):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        fig.add_trace(
            go.Histogram(x=df[feature], name=feature, showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(title_text="Feature Distributions", height=600)
    
    return fig
