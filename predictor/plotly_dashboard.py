import plotly.express as px
import pandas as pd
import requests
import json
import numpy as np
import os

def simplify_coordinates(coords, precision=4):
    """Recursively simplify coordinates in GeoJSON geometry."""
    if isinstance(coords, (list, tuple)):
        if len(coords) == 2 and isinstance(coords[0], (int, float)):
            return [round(coords[0], precision), round(coords[1], precision)]
        return [simplify_coordinates(c, precision) for c in coords]
    return coords

def calculate_centroid(geojson_feature):
    """Simple centroid calculation for polygons/multipolygons."""
    try:
        geom = geojson_feature['geometry']
        if geom['type'] == 'Polygon':
            coords = np.array(geom['coordinates'][0])
        elif geom['type'] == 'MultiPolygon':
            # Use the largest polygon for MultiPolygon
            polys = geom['coordinates']
            largest_poly = max(polys, key=lambda p: len(p[0]))
            coords = np.array(largest_poly[0])
        else:
            return None, None
            
        return np.mean(coords[:, 0]), np.mean(coords[:, 1])
    except:
        return None, None

import plotly.graph_objects as go

def create_rwanda_map(df):
    # Group by district to count clients
    df['district'] = df['district'].str.strip()
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    
    # Use local GeoJSON
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geojson_path = os.path.join(BASE_DIR, 'dummy-data', 'rwanda_districts.geojson')
    
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            rwanda_geojson = json.load(f)
            
        fig = go.Figure()

        # 1. Prepare data for a BOLD drawing of the entire map
        # Extract all boundaries into a single list with None separators for efficiency
        line_lons = []
        line_lats = []
        label_data = []
        
        for feature in rwanda_geojson['features']:
            name = feature['properties']['shapeName']
            geom = feature['geometry']
            
            # Geometry extraction
            if geom['type'] == 'Polygon':
                for ring in geom['coordinates']:
                    line_lons.extend([pt[0] for pt in ring] + [None])
                    line_lats.extend([pt[1] for pt in ring] + [None])
            elif geom['type'] == 'MultiPolygon':
                for poly in geom['coordinates']:
                    for ring in poly:
                        line_lons.extend([pt[0] for pt in ring] + [None])
                        line_lats.extend([pt[1] for pt in ring] + [None])

            # Centroid for labeling
            lon, lat = calculate_centroid(feature)
            if lon:
                count = district_counts[district_counts['district'] == name]['client_count'].values
                val = int(count[0]) if len(count) > 0 else 0
                label_data.append({
                    'lon': lon, 'lat': lat, 
                    'text': f"{name}<br><b>{val}</b>"
                })

        # Add the BOLD boundaries drawing
        fig.add_trace(go.Scattergeo(
            lon=line_lons,
            lat=line_lats,
            mode='lines',
            line=dict(width=2, color='black'), # BOLD black lines
            fill='toself',
            fillcolor='rgba(245, 245, 245, 0.5)', # Subtle drawing fill
            name='Boundaries',
            hoverinfo='skip',
            showlegend=False
        ))

        # 2. Add Text Labels on top
        if label_data:
            ldf = pd.DataFrame(label_data)
            fig.add_trace(go.Scattergeo(
                lon=ldf['lon'],
                lat=ldf['lat'],
                text=ldf['text'],
                mode='text',
                textfont=dict(color="black", size=10),
                showlegend=False,
                hoverinfo='none'
            ))

        # Update layout to focus on Rwanda
        fig.update_geos(
            visible=False,
            fitbounds="geojson",
            projection_type="mercator"
        )
        
    except Exception as e:
        print(f"Error drawing map: {e}")
        # Reliable fallback
        fig = px.bar(district_counts, x='district', y='client_count', 
                     title='District Distribution (Fallback Bar Chart)',
                     template='plotly_white')
    
    fig.update_layout(
        height=800,
        title='Vehicle Clients per District in Rwanda',
        title_x=0.5,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Arial", color='black', size=14),
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig.to_html(full_html=False)
