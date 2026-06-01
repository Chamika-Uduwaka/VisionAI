import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
from shapely.geometry import Point

print("Initializing Phase 2 Spatial Engine (Gold Layer)...")

# 1. Load the Base Data
df_coords = pd.read_csv('outlet_coordinates.csv')
df_master = pd.read_csv('outlet_master.csv')
df_preds = pd.read_csv('VisionAI_predictions.csv') # Contains Predicted_Maximum_Monthly_Liters

# --- LOAD THE SILVER LAYER INSTEAD OF RECALCULATING ---
print("Loading pre-calculated historical averages from Silver Layer...")
avg_sales = pd.read_csv('silver_cleaned_transactions.csv')

# Align data types for a clean merge
print("Aligning data types for a clean merge...")
df_master['Outlet_ID'] = df_master['Outlet_ID'].astype(str)
df_coords['Outlet_ID'] = df_coords['Outlet_ID'].astype(str)
df_preds['Outlet_ID'] = df_preds['Outlet_ID'].astype(str)
avg_sales['Outlet_ID'] = avg_sales['Outlet_ID'].astype(str)

# 2. Merge everything together safely
df = df_master.merge(df_coords, on='Outlet_ID')
df = df.merge(df_preds, on='Outlet_ID', how='left')

# Drop duplicate Distributor_ID if it exists in both tables before merging
if 'Distributor_ID' in df.columns and 'Distributor_ID' in avg_sales.columns:
    avg_sales = avg_sales.drop(columns=['Distributor_ID'])
    
df = df.merge(avg_sales, on='Outlet_ID', how='left')

# 3. Filter STRICTLY for Western Province (DIST_W_01, 02, 03)
western_dists = ['DIST_W_01', 'DIST_W_02', 'DIST_W_03']
df_west = df[df['Distributor_ID'].isin(western_dists)].copy()

# Convert to GeoDataFrame and Project to EPSG:32644 (Meters)
geometry = [Point(xy) for xy in zip(df_west['Longitude'], df_west['Latitude'])]
gdf_west = gpd.GeoDataFrame(df_west, geometry=geometry, crs="EPSG:4326")
gdf_west = gdf_west.to_crs("EPSG:32644")

# --- METRIC 1: COMPETITOR DENSITY ---
print("Calculating Competitor Density (1km radius)...")
gdf_buffer = gdf_west.copy()
gdf_buffer['geometry'] = gdf_buffer.geometry.buffer(1000)

competitors = gpd.sjoin(gdf_west, gdf_buffer, how='inner', predicate='within')
comp_counts = competitors.groupby('Outlet_ID_left').size() - 1
gdf_west['Competitor_Density'] = gdf_west['Outlet_ID'].map(comp_counts).fillna(0)

# --- METRIC 2: DISTANCE-DECAY POI MODELING ---
print("Downloading OSM POIs for Western Province...")
tags = {"amenity": ["school", "hospital", "bus_station", "bank"]}
pois = ox.features_from_place("Western Province, Sri Lanka", tags)
pois = pois[pois.geometry.type == 'Point'].to_crs("EPSG:32644")

print("Calculating Exponential Distance-Decay...")
decay_lambda = 0.002
decay_scores = []

for idx, row in gdf_west.iterrows():
    dists = pois.distance(row.geometry)
    valid_dists = dists[dists <= 2000]
    score = np.sum(np.exp(-decay_lambda * valid_dists))
    decay_scores.append(score)

gdf_west['Decay_POI_Score'] = decay_scores

# Export the final Gold Layer for the optimizer and the web app
df_final = pd.DataFrame(gdf_west.drop(columns='geometry'))
df_final.to_csv('Web_App_Data.csv', index=False)
print("Spatial Engine Complete! 'Web_App_Data.csv' generated.")