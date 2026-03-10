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

def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def _relative_luminance(rgb):
    def channel(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

def _pick_text_color(bg_hex):
    # Return black or white depending on background luminance
    lum = _relative_luminance(_hex_to_rgb(bg_hex))
    return "#111111" if lum > 0.45 else "#ffffff"

def _color_for_value(value, vmin, vmax, colorscale):
    if vmax <= vmin:
        return colorscale[-1][1]
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    for i in range(len(colorscale) - 1):
        p0, c0 = colorscale[i]
        p1, c1 = colorscale[i + 1]
        if p0 <= t <= p1:
            # linear interpolate in RGB space
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            if p1 == p0:
                return c1
            f = (t - p0) / (p1 - p0)
            r = int(r0 + (r1 - r0) * f)
            g = int(g0 + (g1 - g0) * f)
            b = int(b0 + (b1 - b0) * f)
            return f"#{r:02x}{g:02x}{b:02x}"
    return colorscale[-1][1]

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

        # 1. Build district list for consistent coloring, including districts with zero counts
        district_names = [f['properties']['shapeName'] for f in rwanda_geojson['features']]
        counts_map = district_counts.set_index('district')['client_count'].to_dict()
        choropleth_df = pd.DataFrame({
            'district': district_names,
            'client_count': [int(counts_map.get(name, 0)) for name in district_names],
        })

        # Blue gradient colorscale based on client counts
        colorscale = [
            [0.0, "#f7fbff"],
            [0.2, "#c6dbef"],
            [0.4, "#9ecae1"],
            [0.6, "#6baed6"],
            [0.8, "#3182bd"],
            [1.0, "#08519c"],
        ]
        zmin = choropleth_df['client_count'].min()
        zmax = choropleth_df['client_count'].max() if choropleth_df['client_count'].max() > 0 else 1

        # 2. Prepare data for a BOLD drawing of the entire map
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
                    'text': f"{name}<br><b>Sales: {val}</b>"
                })

        # 3. Add colored district fills (inside the map)
        fig.add_trace(go.Choropleth(
            geojson=rwanda_geojson,
            locations=choropleth_df['district'],
            z=choropleth_df['client_count'],
            featureidkey="properties.shapeName",
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            marker_line_color="#1b1b1b",
            marker_line_width=1,
            colorbar=dict(
                title="Sales",
                thickness=12,
                len=0.5,
                x=0.98,
                y=0.5
            ),
            hovertemplate="<b>%{location}</b><br>Sales: %{z}<extra></extra>",
            showscale=True
        ))

        # 4. Add the BOLD boundaries drawing on top
        fig.add_trace(go.Scattergeo(
            lon=line_lons,
            lat=line_lats,
            mode='lines',
            line=dict(width=3, color='#111111'),
            name='Boundaries',
            hoverinfo='skip',
            showlegend=False
        ))

        # 5. Add Text Labels on top with contrast-aware color
        if label_data:
            ldf = pd.DataFrame(label_data)
            label_colors = []
            for name in ldf['text']:
                district_name = name.split('<br>')[0]
                val = int(counts_map.get(district_name, 0))
                fill_color = _color_for_value(val, zmin, zmax, colorscale)
                label_colors.append(_pick_text_color(fill_color))
            fig.add_trace(go.Scattergeo(
                lon=ldf['lon'],
                lat=ldf['lat'],
                text=ldf['text'],
                mode='text',
                textfont=dict(color=label_colors, size=10),
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
