import plotly.express as px
import pandas as pd
import requests
import os

def create_rwanda_map(df):
    # Group by district to count clients
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    
    # URL for Rwanda ADM2 (Districts) GeoJSON
    geojson_url = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/RWA/ADM2/geoBoundaries-RWA-ADM2.geojson"
    
    try:
        # Load GeoJSON from URL
        response = requests.get(geojson_url)
        rwanda_geojson = response.json()
        
        # Create Choropleth Map
        fig = px.choropleth(
            district_counts,
            geojson=rwanda_geojson,
            locations='district',
            featureidkey="properties.shapeName", # Match 'district' with 'shapeName' in GeoJSON
            color='client_count',
            color_continuous_scale="Viridis",
            range_color=(district_counts['client_count'].min(), district_counts['client_count'].max()),
            labels={'client_count': 'Number of Clients'},
            title='Vehicle Clients per District in Rwanda',
            template='plotly_dark'
        )
        
        # Update map to fit Rwanda
        fig.update_geos(
            visible=False, 
            resolution=50,
            showcountries=True, 
            countrycolor="RebeccaPurple",
            fitbounds="locations"
        )
        
    except Exception as e:
        # Fallback to bar chart if GeoJSON loading fails
        print(f"Error loading GeoJSON: {e}")
        fig = px.bar(district_counts, x='district', y='client_count', 
                     title='Vehicle Clients per District in Rwanda (Fallback)',
                     labels={'client_count': 'Number of Clients', 'district': 'District'},
                     template='plotly_dark',
                     color='client_count',
                     color_continuous_scale='Viridis')
    
    fig.update_layout(
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig.to_html(full_html=False)
