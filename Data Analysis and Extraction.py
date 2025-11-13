from pathlib import Path
import geopandas as gpd
import rasterio
from shapely.geometry import box
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
from shapely.geometry import Point
import os
import rasterio.mask
from shapely.validation import make_valid
from scipy.spatial import cKDTree
import rasterio

base_dir = 'F:/WRI'
base_dir = Path('F:/WRI')

# Load the AOI and reproject to a UTM CRS (UTM Zone 37S for Kenya)
input_file = r"F:\AOI\AOI_Part\Part_3_1.geojson"
aoi_gdf = gpd.read_file(input_file)
aoi_utm = aoi_gdf.to_crs(epsg=32737)  # UTM Zone 37S (EPSG:32737) is suitable for Kenya

# Define the width and height of each grid cell in meters
cell_size = 100  # 1 m by 1 m grid cells

# Get the bounds of the AOI in the UTM projection
bounds = aoi_utm.total_bounds  # [minx, miny, maxx, maxy]
print("AOI Bounds in UTM:", bounds)

# Calculate the number of cells needed in the X and Y directions
num_cells_x = int((bounds[2] - bounds[0]) / cell_size)
num_cells_y = int((bounds[3] - bounds[1]) / cell_size)
print("Number of Cells X:", num_cells_x)
print("Number of Cells Y:", num_cells_y)

# Generate 1m x 1m grid cells in UTM coordinates
grid_cells = []
for i in range(num_cells_x):
    for j in range(num_cells_y):
        cell = box(bounds[0] + i * cell_size, bounds[1] + j * cell_size,
                   bounds[0] + (i + 1) * cell_size, bounds[1] + (j + 1) * cell_size)
        grid_cells.append(cell)

# Create a GeoDataFrame for the grid with the UTM CRS
grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=aoi_utm.crs)
grid_gdf['grid_id'] = grid_gdf.index  # Assign a unique ID to each grid cell

# Reproject the grid back to WGS 84 (EPSG:4326)
grid_gdf = grid_gdf.to_crs(epsg=4326)

# Check the structure and validity of the grid in WGS 84
print(grid_gdf)
print("Grid Geometries Valid:", grid_gdf.geometry.is_valid.all())

# Save the grid to CSV
#grid_gdf.to_csv("AOI_grid.csv", index=False)

# Initialize a dictionary to store results
results_dict = {'grid_id': grid_gdf['grid_id']}  # Start with grid IDs

# Function to calculate zonal statistics
def calculate_zonal_stats_for_grid(grid_gdf, raster_data, transform, nodata_value):
    stats = zonal_stats(grid_gdf, raster_data, affine=transform, stats='mean', nodata=nodata_value)
    return stats
    
# Raster file paths for analyses 
raster_files = {
    "irrigation": (r"F:/WRI/Irrigation Systems/Preprocessing/area equipped for irrigation expressed as percentage of total area.tif", -9),
    "livestock": (r"F:/WRI/Livestock Distribution/Preprocessing/Livestock Distribution Clip.tif", -3.40282346638528e+38),
    "energy_consumption_access": (r"F:/WRI/Energy Consumption/Preprocessing/tiersofaccess_SSA_2018.tif", -3.40282347e+38),
    "urban_access": (r"F:/WRI/Urban Accessibility wrt travel time/Urban Accessibility wrt travel time.tif", -2147483648),
    "africa_isobioclimates": (r"F:/WRI/Africa IsoBioclimates/Preprocessing/clipped_isobioclimates.tif", 999),
    "climate_zones": (r"F:/WRI/Climate Zones/Preprocessing/clipped_mean_climate_zone_1901_2020.tiff", 0),
    "soil_constraints": (r"F:/WRI/Global land area with soil constraints/Preprocessing/clipped_land area with soil constraints.tif", -32768),
    "population_distribution": (r"F:/WRI/Population Distribution/Preprocessing/predicted_population_2025.tif", -99999),
    "elevation": (r"F:/WRI/Elevation/Elevation.tif", 0), 
    "slope": (r"F:/WRI/Elevation/Slope.tif", 0),
}

# Loop through each raster file and calculate stats
for var_name, (raster_file, nodata_value) in raster_files.items():
    try:
        with rasterio.open(raster_file) as src:
            raster_data = src.read(1).astype(np.float64)
            # Replace nodata values with NaN for processing
            raster_data = np.where(raster_data == nodata_value, np.nan, raster_data)
            raster_data = np.nan_to_num(raster_data, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
            raster_crs = src.crs
            
            # Ensure grid CRS matches raster CRS
            if grid_gdf.crs != raster_crs:
                print(f'Reprojecting grid to match raster CRS: {raster_crs}')
                grid_gdf = grid_gdf.to_crs(raster_crs)
                
            print(f'Calculating zonal statistics for variable: {var_name}')
            stats = calculate_zonal_stats_for_grid(grid_gdf, raster_data, src.transform, nodata_value=np.nan)
            results_dict[f'{var_name}_mean'] = [stat['mean'] for stat in stats]
            
    except Exception as e:
        print(f'Error processing raster {var_name}: {e}')
            
# Create final DataFrame from results
results_df = pd.DataFrame(results_dict)

# Print the results DataFrame to verify all data
#print(results_df.head())  # Check the first few rows to see if all expected data is included

# Merge results with grid GeoDataFrame
grid_gdf = grid_gdf.merge(results_df, on='grid_id')

# Output the final grid_gdf
print(grid_gdf.head()) 

# Define the mapping of land use classes to their corresponding impact factors on electricity usage
land_use_factors = {
    1: 0.1,  # Water
    2: 0.2,  # Trees
    3: 0.15, # Flooded Vegetation
    4: 0.5,  # Crops
    5: 0.8,  # Built-Up Area
    6: 0.05, # Bare Ground
    7: 0.1,  # Snow/Ice
    8: 0.0,  # Clouds
    9: 0.3   # Rangelands
}

land_use_path = r"F:/WRI/Land Use Classification/Land Use Classification.tif"

# Open the land use raster
with rasterio.open(land_use_path) as land_use_raster:
    # Create a list to hold the impact factors for each grid cell
    impact_factors = []

    # Loop through each grid cell geometry
    for index, row in grid_gdf.iterrows():
        # Get the geometry of the grid cell (bounding box)
        grid_cell_geom = row['geometry']
        
        # Get the bounds of the grid cell
        minx, miny, maxx, maxy = grid_cell_geom.bounds
        
        try:
            # Clip the land use raster with the grid cell geometry (bounding box)
            window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=land_use_raster.transform)
            land_use_data = land_use_raster.read(1, window=window)
            
            if land_use_data.size > 0:
                # Flatten the land use data and get unique land use classes
                unique_land_use_classes = set(land_use_data.flatten())
                # Remove any no-data values
                unique_land_use_classes.discard(0)

                impact_factors_for_cell = []
                if unique_land_use_classes:
                    # Calculate impact factors based on unique land use classes
                    for land_use_class in unique_land_use_classes:
                        impact_factor = land_use_factors.get(land_use_class, np.nan)  # Use np.nan if no factor is found
                        impact_factors_for_cell.append(impact_factor)

                    # Calculate the mean impact factor if there are multiple classes
                    mean_impact_factor = np.nanmean(impact_factors_for_cell)
                    impact_factors.append(mean_impact_factor)
                else:
                    impact_factors.append(np.nan)  # No land use classes found
            else:
                impact_factors.append(np.nan)  # No land use data found

        except Exception as e:
            print(f"Error processing grid cell {index}: {e}")
            impact_factors.append(np.nan)

    # Assign the impact factors back to the grid GeoDataFrame
    grid_gdf['Lulc_impact_factor'] = impact_factors

    # Display the results
    print(grid_gdf[['grid_id', 'Lulc_impact_factor']])

def calculate_mean_per_polygon_Fertilizer(tiff_file, grid_gdf):
    """
    Calculate the mean value for each polygon in the grid_gdf for each band in the multi-band TIFF.
    Clip the raster to the extent of grid_gdf before calculating the mean.
    """
    
    # Open the multi-band TIFF file using rasterio
    with rasterio.open(tiff_file) as src:
        # If the CRS does not match, reproject the grid_gdf to match raster CRS
        if grid_gdf.crs != src.crs:
            grid_gdf = grid_gdf.to_crs(src.crs)
        
        # List to store the results
        results = []
        
        # Get the number of bands in the raster file
        num_bands = src.count
        
        # Get the actual band names from the TIFF metadata and rename as requested
        band_names = []
        for i in range(num_bands):
            band_name = src.descriptions[i] if src.descriptions[i] else f'Band_{i+1}'  # Default to Band_# if no description
            # Remove any `.tif` from the description and add the prefix
            band_name = band_name.replace('.tif', '')
            new_band_name = f"fertilizer_{band_name}_mean"
            band_names.append(new_band_name)
        
        # Iterate over each polygon in the grid_gdf
        for idx, row in grid_gdf.iterrows():
            polygon = [row['geometry']]
            
            try:
                # Mask the raster with the polygon geometry using rasterio.mask
                out_image, out_transform = rasterio.mask.mask(src, polygon, crop=True)
                
                if out_image.size == 0 or np.all(np.isnan(out_image)):
                    band_means = [np.nan] * num_bands  # Return NaN for all bands if no valid data
                else:
                    out_image[out_image == 0] = np.nan  # Replace 0 with NaN
                    out_image = np.ma.masked_invalid(out_image)  # Mask out invalid (NaN) values
                    
                    # Calculate the mean value for each band
                    band_means = [np.mean(out_image[band_idx, :, :]) if out_image[band_idx, :, :].size > 0 else np.nan for band_idx in range(num_bands)]
                
                # Append the results for this polygon
                results.append([row['grid_id']] + band_means)
            except Exception as e:
                print(f"Error processing polygon {row['grid_id']}: {e}")
                band_means = [np.nan] * num_bands  # Return NaN for all bands in case of error
                results.append([row['grid_id']] + band_means)
        
        # Convert results to a DataFrame
        df = pd.DataFrame(results, columns=['grid_id'] + band_names)
        
        df.replace('--', np.nan, inplace=True)
        
        # Append the new columns to the original grid_gdf DataFrame (without replacing any existing columns)
        grid_gdf = grid_gdf.join(df.set_index('grid_id'), on='grid_id')
        
        return grid_gdf

def calculate_mean_per_polygon_Yield_Potential(tiff_file, grid_gdf):
    """
    Calculate the mean value for each polygon in the grid_gdf for each band in the multi-band TIFF.
    Clip the raster layer to the extent of grid_gdf before calculating the mean.
    """
    
    # Open the multi-band TIFF file using rasterio
    with rasterio.open(tiff_file) as src:
        # If the CRS does not match, reproject the grid_gdf to match raster CRS
        if grid_gdf.crs != src.crs:
            grid_gdf = grid_gdf.to_crs(src.crs)
        
        # Get the number of bands in the raster file
        num_bands = src.count
        
        # Get the actual band names from the TIFF metadata and rename as requested
        band_names = []
        for i in range(num_bands):
            band_name = src.descriptions[i] if src.descriptions[i] else f'Band_{i+1}'
            # Remove any `.tif` from the description and add the prefix
            band_name = band_name.replace('.tif', '')
            new_band_name = f"yield_{band_name}_mean"
            band_names.append(new_band_name)
        
        # Iterate over each polygon in the grid_gdf
        results = []  # List to store the results for each polygon
        for idx, row in grid_gdf.iterrows():
            polygon = [row['geometry']]
            
            try:
                # Mask the raster with the polygon geometry using rasterio.mask
                out_image, out_transform = rasterio.mask.mask(src, polygon, crop=True)
                if out_image.size == 0 or np.all(np.isnan(out_image)):
                    band_means = [np.nan] * num_bands  # Return NaN for all bands if no valid data
                else:
                    out_image[out_image == 0] = np.nan  # Replace 0 with NaN
                    out_image = np.ma.masked_invalid(out_image)  # Mask out invalid (NaN) values
                    
                    # Calculate the mean value for each band
                    band_means = [np.mean(out_image[band_idx, :, :]) if out_image[band_idx, :, :].size > 0 else np.nan for band_idx in range(num_bands)]
                
                # Append the results for this polygon
                results.append([row['grid_id']] + band_means)
            except Exception as e:
                print(f"Error processing polygon {row['grid_id']}: {e}")
                band_means = [np.nan] * num_bands  # Return NaN for all bands in case of error
                results.append([row['grid_id']] + band_means)
        
        # Convert results to a DataFrame
        df = pd.DataFrame(results, columns=['grid_id'] + band_names)
        
        df.replace('--', np.nan, inplace=True)
        # Append the new columns to the original grid_gdf DataFrame (without replacing any existing columns)
        grid_gdf = grid_gdf.join(df.set_index('grid_id'), on='grid_id')
        
        return grid_gdf


def calculate_mean_per_polygon_Crop_Type_Distribution(tiff_file, grid_gdf):
    """
    Calculate the mean value for each polygon in the grid_gdf for each band in the multi-band TIFF.
    Clip the raster layer to the extent of grid_gdf before calculating the mean.
    """
    
    # Open the multi-band TIFF file using rasterio
    with rasterio.open(tiff_file) as src:
        # If the CRS does not match, reproject the grid_gdf to match raster CRS
        if grid_gdf.crs != src.crs:
            grid_gdf = grid_gdf.to_crs(src.crs)
        
        # Get the number of bands in the raster file
        num_bands = src.count
        
        # Get the actual band names from the TIFF metadata and rename as requested
        band_names = []
        for i in range(num_bands):
            band_name = src.descriptions[i] if src.descriptions[i] else f'Band_{i+1}'  
            # Remove any `.tif` from the description and add the prefix
            band_name = band_name.replace('.tif', '')
            new_band_name = f"crop_type_{band_name}_mean"
            band_names.append(new_band_name)
        
        # List to store the results
        results = []
        
        # Iterate over each polygon in the grid_gdf
        for idx, row in grid_gdf.iterrows():
            polygon = [row['geometry']]
            
            try:
                # Mask the raster with the polygon geometry using rasterio.mask
                out_image, out_transform = rasterio.mask.mask(src, polygon, crop=True)
                
                if out_image.size == 0 or np.all(np.isnan(out_image)):
                    band_means = [np.nan] * num_bands  # Return NaN for all bands if no valid data
                else:
                    out_image[out_image == 0] = np.nan  # Replace 0 with NaN
                    out_image = np.ma.masked_invalid(out_image)  # Mask out invalid (NaN) values
                    
                    # Calculate the mean value for each band
                    band_means = [np.mean(out_image[band_idx, :, :]) if out_image[band_idx, :, :].size > 0 else np.nan for band_idx in range(num_bands)]
                
                # Append the results for this polygon
                results.append([row['grid_id']] + band_means)
            except Exception as e:
                print(f"Error processing polygon {row['grid_id']}: {e}")
                band_means = [np.nan] * num_bands  # Return NaN for all bands in case of error
                results.append([row['grid_id']] + band_means)
        
        # Convert results to a DataFrame
        df = pd.DataFrame(results, columns=['grid_id'] + band_names)
        df.replace('--', np.nan, inplace=True)
        # Append the new columns to the original grid_gdf DataFrame (without replacing any existing columns)
        grid_gdf = grid_gdf.join(df.set_index('grid_id'), on='grid_id')
        
        return grid_gdf
    
Fertilizer=r"F:/WRI/Fertilizer Application/Fertilizer Application.tif"
grid_gdf = calculate_mean_per_polygon_Fertilizer(Fertilizer, grid_gdf)
print("Fertilizer Data")

Yield_Potential = r"F:/WRI/Yield Potential/Yield Potential.tif"
grid_gdf = calculate_mean_per_polygon_Yield_Potential(Yield_Potential, grid_gdf)
print("Yield Potential Data")
    
Crop_Type_Distribution = r"F:/WRI/Crop Type Distribution/Crop Type Distribution.tif"
grid_gdf = calculate_mean_per_polygon_Crop_Type_Distribution(Crop_Type_Distribution, grid_gdf)
print("Crop Type Distribution Data")

# Load Agro-Processing Facilities CSV file
agro_processing_csv_path = base_dir / 'Agro-Processing Facilities' / 'Agro-Processing Facilities_with_coordinates.csv'
agro_processing_df = pd.read_csv(agro_processing_csv_path)

# Create Point geometries from latitude and longitude in the CSV
geometry = [Point(xy) for xy in zip(agro_processing_df['Longitude'], agro_processing_df['Latitude'])]
gdf_agro_processing = gpd.GeoDataFrame(agro_processing_df, geometry=geometry)
gdf_agro_processing.set_crs(epsg=4326, inplace=True)

# Ensure the CRS of both GeoDataFrames are the same
grid_gdf = grid_gdf.to_crs(epsg=4326)

# Spatial join to check if Agro-Processing Facilities are inside the grid cells
join_within = gpd.sjoin(grid_gdf, gdf_agro_processing, predicate='contains', how='left')

# Initialize the grid result column with 0, and convert it to float type
grid_gdf['agro_processing_in_grid'] = 0.0

# Assign 1 where Agro-Processing Facilities are found in the grid cell
grid_gdf.loc[~join_within.index_right.isna(), 'agro_processing_in_grid'] = 1.0

# Reproject to a CRS that uses meters before applying the buffer (e.g., EPSG:3857)
grid_gdf_proj = grid_gdf.to_crs(epsg=3857)

# Create a buffer (e.g., 5 km) around each grid cell to check for nearby facilities
buffer_distance = 5000  # 5 km buffer in meters
grid_gdf_proj['buffered_geometry'] = grid_gdf_proj.geometry.buffer(buffer_distance)

# Spatial join to check if Agro-Processing Facilities are within the buffer (nearby)
gdf_agro_processing_proj = gdf_agro_processing.to_crs(epsg=3857)
join_nearby = gpd.sjoin(grid_gdf_proj.set_geometry('buffered_geometry'), gdf_agro_processing_proj, predicate='intersects', how='left')

# Assign 0.5 where Agro-Processing Facilities are nearby but not inside the grid
grid_gdf_proj.loc[~join_nearby.index_right.isna() & (grid_gdf_proj['agro_processing_in_grid'] == 0), 'agro_processing_in_grid'] = 0.5

# Reproject back to EPSG:4326 and update original grid_gdf
grid_gdf_proj = grid_gdf_proj.to_crs(epsg=4326)

# Load Road and Railway data
roads_gdf = gpd.read_file(base_dir / 'Road Network' / 'kenya_roads'/'Kenya_roads_version2.shp')  # Load roads shapefile
railways_gdf = gpd.read_file(base_dir / 'Railway Networks' / 'KEN_Rails.shp')  # Load railways shapefile

roads_gdf = roads_gdf.to_crs(epsg=4326)
railways_gdf = railways_gdf.to_crs(epsg=4326)

roads_gdf = roads_gdf.clip(grid_gdf_proj)
railways_gdf = railways_gdf.clip(grid_gdf_proj)

# Spatial join to check if roads and railways are inside the grid cells
grid_gdf_proj['road_in_grid'] = 0.0
grid_gdf_proj['railway_in_grid'] = 0.0

# Spatial join for roads
join_roads = gpd.sjoin(grid_gdf_proj, roads_gdf, predicate='intersects', how='left')

# Check if 'index_left' exists, otherwise use the original grid_gdf_proj's index
if 'index_left' not in join_roads.columns:
    join_roads['index_left'] = join_roads.index

# Group by index_left (the grid cell index) and check if there's any intersection (non-null index_right)
road_intersects = join_roads.groupby('index_left', group_keys=False).apply(
    lambda x: ~x['index_right'].isna().any()
).reset_index()
road_intersects.columns = ['index_left', 'intersects_road']

# Set road_in_grid to 1 for intersecting grids
grid_gdf_proj.loc[grid_gdf_proj.index.isin(road_intersects[road_intersects['intersects_road']].index_left), 'road_in_grid'] = 1.0

# Spatial join for railways
join_railways = gpd.sjoin(grid_gdf_proj, railways_gdf, predicate='intersects', how='left')

# Check if 'index_left' exists, otherwise use the original grid_gdf_proj's index
if 'index_left' not in join_railways.columns:
    join_railways['index_left'] = join_railways.index

# Group by index_left (the grid cell index) and check if there's any intersection (non-null index_right)
railway_intersects = join_railways.groupby('index_left', group_keys=False).apply(
    lambda x: ~x['index_right'].isna().any()
).reset_index()
railway_intersects.columns = ['index_left', 'intersects_railway']

# Set railway_in_grid to 1 for intersecting grids
grid_gdf_proj.loc[grid_gdf_proj.index.isin(railway_intersects[railway_intersects['intersects_railway']].index_left), 'railway_in_grid'] = 1.0

grid_gdf_proj.drop(columns=['buffered_geometry'], inplace=True)

# Reproject back to EPSG:4326 if needed
grid_gdf = grid_gdf_proj.to_crs(epsg=4326)


# Display the final grid DataFrame with agro-processing, road, and railway information
print(grid_gdf[['grid_id', 'agro_processing_in_grid', 'road_in_grid', 'railway_in_grid']].head())

# Load the water source locations shapefile
water_source_locations = gpd.read_file(os.path.join(base_dir, 'Water Source Locations', 'Preprocessing', 'Water Source Locations.shp'))

# Ensure the CRS of both GeoDataFrames match
# Reproject water sources to the CRS of grid_gdf_proj
water_source_locations_proj = water_source_locations.to_crs(grid_gdf_proj.crs)

# Choose a suitable projected CRS for accurate distance calculations
utm_crs = 'EPSG:3857'

# Project both GeoDataFrames to UTM
grid_gdf_proj_utm = grid_gdf_proj.to_crs(utm_crs)
water_source_locations_proj_utm = water_source_locations_proj.to_crs(utm_crs)

# Perform a spatial join for water sources
join_water_sources = gpd.sjoin(grid_gdf_proj_utm, water_source_locations_proj_utm, predicate='intersects', how='left')

# Mark grid cells with water sources as 'no agriculture' (0)
grid_gdf_proj_utm['agriculture_feasibility'] = 1.0  # Default: suitable for agriculture
grid_gdf_proj_utm.loc[~join_water_sources['index_right'].isna(), 'agriculture_feasibility'] = 0.0  # No agriculture in cells with water sources

# Calculate distance to the nearest water source for nearby cells
# Function to calculate distance to the nearest water source
def calculate_distance_to_nearest_water(grid_geom, water_geoms):
    # Ensure water_geoms is not empty
    if water_geoms.empty:
        return np.nan  # No water sources available
    distances = water_geoms.distance(grid_geom)
    return distances.min()

# Apply distance calculation for all grid cells
grid_gdf_proj_utm['distance_to_water'] = grid_gdf_proj_utm.geometry.apply(
    lambda geom: calculate_distance_to_nearest_water(geom, water_source_locations_proj_utm.geometry)
)

# Update agriculture feasibility based on proximity to water sources
# Define a threshold distance (e.g., 500 meters) within which agriculture feasibility is influenced by proximity to water
threshold_distance = 500
grid_gdf_proj_utm.loc[
    (grid_gdf_proj_utm['distance_to_water'] <= threshold_distance) & 
    (grid_gdf_proj_utm['agriculture_feasibility'] == 1), 
    'agriculture_feasibility'] = 0.5  # Partial suitability for nearby cells

# Reproject back to original CRS (EPSG:4326) 
grid_gdf = grid_gdf_proj_utm.to_crs(epsg=4326)

# Display the final grid DataFrame with agriculture feasibility
print(grid_gdf[['grid_id', 'agriculture_feasibility', 'distance_to_water']].head())

# Load and reproject data
def load_reproject_and_clip(file_path, target_crs, grid_gdf):
    gdf = gpd.read_file(file_path)
    gdf = gdf.to_crs(target_crs)
    return gdf

# Ensure invalid geometries are fixed
def fix_invalid_geometries(gdf):
    if not gdf.is_valid.all():
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: make_valid(geom) if geom and not geom.is_valid else geom)
    return gdf

# Function to build KD-Tree for fast distance calculation
def build_kd_tree(geom_series):
    valid_coords = [
        (geom.centroid.x, geom.centroid.y) for geom in geom_series
        if geom.is_valid and isinstance(geom.centroid, Point)
    ]
    if not valid_coords:
        raise ValueError("No valid centroids found to build KD-Tree. Check input geometries.")
    return cKDTree(np.array(valid_coords))

# Function to calculate minimum distance using KD-Tree
def calculate_min_distance_kdtree(geom, kdtree):
    if not geom.is_valid:
        return np.inf  # Invalid geometry, return infinite distance
    distance, _ = kdtree.query([geom.centroid.x, geom.centroid.y])
    return distance

# Load and reproject infrastructure layers
target_crs = "EPSG:3857"  # Projected CRS
grid_gdf_proj = grid_gdf.to_crs(target_crs)  # Ensure correct CRS transformation

# Load infrastructure layers
electricity_distribution_lines = load_reproject_and_clip(
    os.path.join(base_dir, 'Electricity Distribution Lines', 'Preprocessing', 'Electricity Distribution Lines.shp'),
    target_crs, grid_gdf)
electricity_substations = load_reproject_and_clip(
    os.path.join(base_dir, 'Electricity Substations', 'Primary_Substations.shp'),
    target_crs, grid_gdf)
electricity_transmission_lines = load_reproject_and_clip(
    os.path.join(base_dir, 'Electricity Transmission Lines', 'kenya-electricity-transmission-network', 'Kenya Electricity Transmission Network.shp'),
    target_crs, grid_gdf)
distribution_transformers = load_reproject_and_clip(
    os.path.join(base_dir, 'Distribution Transformers', 'distribution-transformer', 'Distribution Transformer', 'Distribution_Transformers.shp'),
    target_crs, grid_gdf)

# Handle invalid geometries
electricity_distribution_lines = fix_invalid_geometries(electricity_distribution_lines)
electricity_substations = fix_invalid_geometries(electricity_substations)
electricity_transmission_lines = fix_invalid_geometries(electricity_transmission_lines)
distribution_transformers = fix_invalid_geometries(distribution_transformers)

# Buffer for faster calculations later
buffer_size_distribution = 1000  # meters
electricity_distribution_lines_buffered = electricity_distribution_lines.geometry.buffer(buffer_size_distribution)
electricity_distribution_lines_buffered = electricity_distribution_lines_buffered[~electricity_distribution_lines_buffered.isna()]

# Build KD-Trees for faster distance calculations
distribution_tree = build_kd_tree(electricity_distribution_lines_buffered)
substations_tree = build_kd_tree(electricity_substations.geometry)
transmission_tree = build_kd_tree(electricity_transmission_lines.geometry)
transformers_tree = build_kd_tree(distribution_transformers.geometry)

# Calculate minimum distances to each electricity feature using KD-Tree
grid_gdf_proj['distance_to_distribution'] = grid_gdf_proj.geometry.apply(
    lambda geom: calculate_min_distance_kdtree(geom, distribution_tree) if geom.is_valid else np.inf
)
grid_gdf_proj['distance_to_substations'] = grid_gdf_proj.geometry.apply(
    lambda geom: calculate_min_distance_kdtree(geom, substations_tree) if geom.is_valid else np.inf
)
grid_gdf_proj['distance_to_transmission'] = grid_gdf_proj.geometry.apply(
    lambda geom: calculate_min_distance_kdtree(geom, transmission_tree) if geom.is_valid else np.inf
)
grid_gdf_proj['distance_to_transformers'] = grid_gdf_proj.geometry.apply(
    lambda geom: calculate_min_distance_kdtree(geom, transformers_tree) if geom.is_valid else np.inf
)

# Set thresholds and update electricity influence based on proximity
distribution_threshold = 500
substation_threshold = 1000
transmission_threshold = 1500
transformer_threshold = 800

grid_gdf_proj['electricity_influence'] = 1.0
grid_gdf_proj.loc[grid_gdf_proj['distance_to_distribution'] <= distribution_threshold, 'electricity_influence'] *= 0.5
grid_gdf_proj.loc[grid_gdf_proj['distance_to_substations'] <= substation_threshold, 'electricity_influence'] *= 0.5
grid_gdf_proj.loc[grid_gdf_proj['distance_to_transmission'] <= transmission_threshold, 'electricity_influence'] *= 0.5
grid_gdf_proj.loc[grid_gdf_proj['distance_to_transformers'] <= transformer_threshold, 'electricity_influence'] *= 0.5

# Reproject back to EPSG 4326 for final display
grid_gdf = grid_gdf_proj.to_crs("EPSG:4326")

# Display final results
print(grid_gdf[['grid_id', 'electricity_influence', 'distance_to_distribution', 'distance_to_substations', 'distance_to_transmission', 'distance_to_transformers']].head())

# Load urban accessibility data
urban_accessibility = gpd.read_file(os.path.join(base_dir, 'Urban Accessibility', 'ke_urban', 'ke_urban.shp'))

# Load regulatory zones
regulatory_zones = gpd.read_file(os.path.join(base_dir, 'Regulatory Zones', 'ke_protected-areas', 'ke_protected-areas.shp'))

# Set a target projected CRS for the grid GeoDataFrame
target_crs = "EPSG:3857" 
grid_gdf_proj = grid_gdf_proj.to_crs(target_crs)

# Ensure regulatory zones and urban accessibility have the same CRS
regulatory_zones = regulatory_zones.to_crs(target_crs)
urban_accessibility = urban_accessibility.to_crs(target_crs)

#cCheck if any grid cell is within the regulatory zones
grid_gdf_proj['in_regulatory_zone'] = grid_gdf_proj.geometry.apply(
    lambda geom: regulatory_zones.intersects(geom).any()
)

# Step 4: Check if any grid cell is within urban areas
grid_gdf_proj['in_urban_area'] = grid_gdf_proj.geometry.apply(
    lambda geom: urban_accessibility.intersects(geom).any()
)

# Calculate distance to urban areas for non-urban grid cells
def calculate_distance_to_urban(grid_geom, urban_geoms):
    if urban_geoms.empty:
        return np.nan  # No urban area available
    distances = urban_geoms.distance(grid_geom)
    return distances.min()

# Calculate distances only for grid cells that are not in regulatory zones and are not in urban areas
grid_gdf_proj['distance_to_urban'] = np.where(
    grid_gdf_proj['in_regulatory_zone'] | grid_gdf_proj['in_urban_area'], 
    np.nan,  # No distance calculation for grid cells in regulatory zones or urban areas
    grid_gdf_proj.geometry.apply(lambda geom: calculate_distance_to_urban(geom, urban_accessibility.geometry))
)

# Convert boolean columns to integer (0 and 1)
grid_gdf_proj['in_regulatory_zone'] = grid_gdf_proj['in_regulatory_zone'].astype(int)
grid_gdf_proj['in_urban_area'] = grid_gdf_proj['in_urban_area'].astype(int)

# Reproject back to original CRS (EPSG:4326)
grid_gdf = grid_gdf_proj.to_crs(epsg=4326)

# Display the final grid DataFrame
print(grid_gdf[['grid_id','in_regulatory_zone', 'in_urban_area', 'distance_to_urban']].head())

# Set the path for the Market Prices CSV file
market_prices_csv_path = base_dir / 'Market Prices' / 'Market Prices.csv'

# Read the CSV file into a DataFrame
market_prices_df = pd.read_csv(market_prices_csv_path)

# Create Point geometries from latitude and longitude
geometry = [Point(xy) for xy in zip(market_prices_df['Longitude'], market_prices_df['Latitude'])]

# Create a GeoDataFrame from the DataFrame
gdf_market_prices = gpd.GeoDataFrame(market_prices_df, geometry=geometry)

# Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
gdf_market_prices.set_crs(epsg=4326, inplace=True)

# Reproject to a suitable projected CRS (e.g., UTM)
gdf_market_prices = gdf_market_prices.to_crs(epsg=3857)  # Web Mercator
grid_gdf_proj = grid_gdf_proj.to_crs(epsg=3857)  # Ensure grid is also in the same CRS

#  Calculate distance to nearest market for each grid cell
def calculate_distance_to_nearest_market(grid_geom, market_geoms):
    if market_geoms.empty:
        return np.inf  # No markets available
    distances = market_geoms.distance(grid_geom)
    return distances.min()

grid_gdf_proj['distance_to_market'] = grid_gdf_proj.geometry.apply(
    lambda geom: calculate_distance_to_nearest_market(geom, gdf_market_prices.geometry)
)

# Step 2: Assign weights based on average market price
max_price = gdf_market_prices['Avg_Price'].max()
min_price = gdf_market_prices['Avg_Price'].min()

gdf_market_prices['normalized_price'] = (gdf_market_prices['Avg_Price'] - min_price) / (max_price - min_price)

# Merge normalized price back into the grid GeoDataFrame
def assign_market_price_weight(row):
    if row['distance_to_market'] == np.inf:
        return 0.0  # No influence if no market is available
    nearest_market_idx = gdf_market_prices.geometry.distance(row.geometry).idxmin()
    nearest_market = gdf_market_prices.iloc[nearest_market_idx]
    return nearest_market['normalized_price']

# Assign market price influence based on distance to market
grid_gdf_proj['market_price_influence'] = grid_gdf_proj.apply(assign_market_price_weight, axis=1)

# Reproject back to original CRS (EPSG:4326)
grid_gdf = grid_gdf_proj.to_crs(epsg=4326)

# Display the final grid DataFrame with updated demand
print(grid_gdf[['grid_id','distance_to_market', 'market_price_influence']].head())

# Load the arable land suitability data
arable_land_suitability = gpd.read_file(os.path.join(base_dir, 'Arable Land Suitability', 'ke_agriculture', 'ke_agriculture.shp'))

# Define the demand mapping based on unique agricultural activities
demand_mapping = {
    'Isolated (in natural vegetation or other) Rainfed herbaceous crop (field density 10-20% polygon area)': 1.0,  
    'Scattered (in natural vegetation or other) Rainfed herbaceous crop (field density 20-40% of polygon area)': 1.2,
    'Rainfed herbaceous crop': 1.5,
    'Rainfed tree crop': 1.3,
    'Irrigated herbaceous crop': 2.0,
    'Scattered (in natural vegetation or other) Rainfed tree crop (field density 20-40% of polygon area)': 1.4,
    'Scattered (in natural vegetation or other) Rainfed shrub crop (field density 20-40% of polygon area)': 1.3,
    'Rainfed shrub crop': 1.1,
    'Forest plantation - undifferentiated': 0.8,
    'Rice fields': 1.6,
}

# Map demand factors to a new column in the arable_land_suitability dataframe
arable_land_suitability['arable_land_adjusted_demand'] = arable_land_suitability['AGRICULTUR'].map(demand_mapping)

# Ensure both GeoDataFrames have the same CRS
if grid_gdf_proj.crs != arable_land_suitability.crs:
    arable_land_suitability = arable_land_suitability.to_crs(grid_gdf_proj.crs)
    
arable_land_suitability = arable_land_suitability.clip(grid_gdf_proj)

# Perform the spatial join after CRS alignment
grid_gdf_proj = gpd.sjoin(
    grid_gdf_proj,
    arable_land_suitability[['geometry', 'arable_land_adjusted_demand']],
    how='left',
    predicate='intersects'
)

grid_gdf_proj['arable_land_adjusted_demand'] = grid_gdf_proj['arable_land_adjusted_demand'].fillna(0)
grid_gdf_proj.drop(columns=['index_right'], inplace=True)

# Reproject back to original CRS (EPSG:4326)
grid_gdf = grid_gdf_proj.to_crs(epsg=4326)

# Print the resulting DataFrame for verification
print(grid_gdf[['grid_id', 'arable_land_adjusted_demand']].head())


biomass_residue_mgmt = gpd.read_file(os.path.join(base_dir, 'Biomass Residue Management', 'Biomass Residue Management.shp'))
# Inspect unique values in the AGRICULTUR column
#print(biomass_residue_mgmt)

unique_activities = biomass_residue_mgmt['code1_desc'].unique()
#print(unique_activities)

relevant_code1_desc = [
    'Trees Plantation - Large Fields, Rainfed Permanent',
    'Needle Leaved Evergreen Forest Plantation',
    'Forest Plantation, Broad Leaved Evergreen, Rainfed Permanent'
]

#Reproject biomass_residue_mgmt to match grid_gdf_proj's CRS if needed
if biomass_residue_mgmt.crs != grid_gdf_proj.crs:
    biomass_residue_mgmt = biomass_residue_mgmt.to_crs(grid_gdf_proj.crs)
    
biomass_residue_mgmt = biomass_residue_mgmt.clip(grid_gdf_proj)

# Calculate biomass_demand_factor with conditional logic
max_logres_t = biomass_residue_mgmt['logres_t'].max()

biomass_residue_mgmt['biomass_demand_factor'] = biomass_residue_mgmt.apply(
    lambda row: (row['logres_t'] / max_logres_t) if (
        row['logres_t'] > 0 and row['code1_desc'] in relevant_code1_desc and max_logres_t > 0
    ) else 0, axis=1
)

#  Proceed with the spatial join
grid_gdf_proj = gpd.sjoin(grid_gdf_proj,biomass_residue_mgmt[['geometry', 'biomass_demand_factor']], how='left', predicate='intersects',lsuffix='left', rsuffix='bio')

# Replace NaN values with 0 in biomass_demand_factor column
grid_gdf_proj['biomass_demand_factor'] = grid_gdf_proj['biomass_demand_factor'].fillna(0)

grid_gdf_proj.drop(columns=['index_bio'], inplace=True)

# Reproject back to original CRS (EPSG:4326)
grid_gdf = grid_gdf_proj.to_crs(epsg=4326)

# Verify the updated DataFrame
print(grid_gdf[['grid_id', 'biomass_demand_factor']].head())

# Load farm size distribution shapefile
farm_size_distribution = gpd.read_file(os.path.join(base_dir, 'Farm Size Distribution', 'ke_crops_size', 'ke_crops_size.shp'))

# Define farm size categories and electricity demand factors using AREA_SQKM_ as the farm size
small_farm_threshold = farm_size_distribution['AREA_SQKM_'].quantile(0.33)
medium_farm_threshold = farm_size_distribution['AREA_SQKM_'].quantile(0.66)

def calculate_electricity_demand(row):
    if row['AREA_SQKM_'] <= small_farm_threshold:
        return 0.5  # Low demand for small farms
    elif row['AREA_SQKM_'] <= medium_farm_threshold:
        return 1.0  # Medium demand for medium farms
    else:
        return 1.5  # High demand for large farms

# Apply electricity demand factor based on farm size
farm_size_distribution['farm_size_distribution_factor'] = farm_size_distribution.apply(calculate_electricity_demand, axis=1)

# Verify the output
#print(farm_size_distribution[['AREA_SQKM_', 'farm_size_distribution_factor']].head())

# Ensure the geometry is included for the spatial join
farm_size_distribution_relevant = farm_size_distribution[['AREA_SQKM_', 'farm_size_distribution_factor', 'geometry']]

# Check CRS of both GeoDataFrames
#print(f"Grid CRS: {grid_gdf_proj.crs}")
#print(f"Farm Size Distribution CRS: {farm_size_distribution.crs}")

# Reproject the farm_size_distribution to match the grid CRS if they are different
if grid_gdf_proj.crs != farm_size_distribution.crs:
    farm_size_distribution_relevant = farm_size_distribution_relevant.to_crs(grid_gdf_proj.crs)

farm_size_distribution_relevant = farm_size_distribution_relevant.clip(grid_gdf_proj)

# Perform spatial join to add farm size distribution factors to the grid GeoDataFrame
grid_gdf_proj = gpd.sjoin(grid_gdf_proj, farm_size_distribution_relevant, how='left', predicate='intersects', lsuffix='left', rsuffix='farm')

grid_gdf_proj['farm_size_distribution_factor'] = grid_gdf_proj['farm_size_distribution_factor'].fillna(0)
# Verify the updated DataFrame
grid_gdf_proj.drop(columns=['index_farm', 'AREA_SQKM_', ], inplace=True)

grid_gdf = grid_gdf_proj.to_crs(epsg=4326)

print(grid_gdf[['grid_id', 'farm_size_distribution_factor']].head())

# Load the soil characteristics GeoDataFrame
soil_characteristics = gpd.read_file(os.path.join(base_dir, 'Soil Characteristics', 'Soil Characteristics.shp'))

# Define impact mappings for drainage, depth, and soil type
drainage_impact_mapping = {
    'well drained': 0.2,
    'very poorly drained': 0.8,
    'imperfectly drained to poorly drained': 0.6,
    'imperfectly drained': 0.5,
    'excessively drained to well drained': 0.3,
    'poorly drained': 0.7,
    'moderately well drained': 0.4,
    'excessively drained': 0.9,
    'varying drainage condition': 0.5,
    '9999': np.nan,  # Assuming '9999' represents missing data
    'imperfectly drained to well drained': 0.5,
    'poorly drained to moderately well drained': 0.65,
    'excessively drained': 0.9,
    'well drained to moderately well drained': 0.3,
    'poorly drained to well drained': 0.6,
    'moderately well drained to well drained': 0.3,
    'excessively drained to moderately well drained': 0.8,
    'well drained to imperfectly': 0.35,
    'well drained to imperfectly drained': 0.35,
    'excessively drained to imperfectly drained': 0.75,
    None: np.nan,  # For null values
    'poorly drained to very poorly drained': 0.75,
    'moderately well drained to imperfectly drained': 0.45,
    'imperfectly drained to very poorly drained': 0.7,
    'imperfectly drained to excessively drained': 0.65,
    'excessively drained to poorly drained': 0.8,
    'poorly drained to excessively drained': 0.75,
    'well drained to poorly drained': 0.5,
    'moderately well drained to poorly drained': 0.6,
    'imperfectly drained to moderately well drained': 0.55
}

depth_impact_mapping = {
    'very shallow to moderately deep': 0.4,
    'very deep': 0.7,
    'deep to very deep': 0.6,
    'moderately deep to deep': 0.5,
    'shallow to moderately deep': 0.4,
    'shallow to deep': 0.3,
    'deep': 0.6,
    'extremely deep': 0.8,
    'shallow': 0.3,
    'very shallow to extremely deep': 0.5,
    'moderately deep to very deep': 0.6,
    'shallow to very deep': 0.7,
    'deep to extremely deep': 0.7,
    'moderately deep': 0.5,
    'very shallow to shallow': 0.3,
    'varying': 0.5,
    'very shallow to deep': 0.4,
    '9999': np.nan,  # Assuming '9999' represents missing data
    'very deep to extremely deep': 0.75,
    'shallow to moderately deeep': 0.4,  # Assuming typo, matching 'shallow to moderately deep'
    'shallow to extremely deep': 0.6,
    'dusky red to dark reddish brown': np.nan,  # Assuming color description, no impact value
    'moderately deep to extremely deep': 0.7,
    None: np.nan  # For null values
}


soil_demand_map = {
    'Acrisols and Cambisols and Lithosols': 0.4,
    'Ferralsols': 0.6,
    'Fluvisols and Solonchaks': 0.8,
    'Ferralsols with Acrisols': 0.7,
    'Gleysols and Vertisols': 0.9,
    'Histosols and Lithosols': 0.2,
    'Cambisols and Luvisols': 0.6,
    'Regosols': 0.3,
    '9999': 0,
    'Cambisols, Acrisols and Ferralsols and Gleysols with Vertisols and Histosols': 0.7,
    'Vertisols and Gleysols': 0.9,
    'Planosols with Histosols': 0.5,
    'Xerosols': 0.4,
    'Cambisols and Arenosols': 0.6,
    'Gleysols': 0.8,
    'Acrisols': 0.5,
    'Planosols, Gleysols, Solonetz, Vertisols and Fluvisols': 0.9,
    'Lithosols': 0.2,
    'Arenosols': 0.4,
    'Nitisols': 0.8,
    'Solonchaks': 0.9,
    'Luvisols': 0.7,
    'Planosols': 0.5,
    'Histosols': 0.3,
    'Nitisols with Cambisols and Acrisols': 0.8,
    'Vertisols and Rendzinas': 0.7,
    'Phaeozems and Luvisols': 0.7,
    'Planosols with Solonetz': 0.5,
    'Phaeozems and Planosols': 0.6,
    'Nitisols and Luvisols': 0.8,
    'Solonetz and Planosols and Vertisols': 0.6,
    'Vertisols with Gleysols and Solonetz': 0.9,
    'Luvisols and Acrisols and Cambisols with Lithosols': 0.7,
    'Xerosols with Arenosols': 0.4,
    'Acrisols and Luvisols with Ferralsols': 0.6,
    'Regosols with Rankers, Cambisols and Lithosols': 0.5,
    'Phaeozems': 0.7,
    'Ironstone with Lithosols, Vertisols and Gleysols': 0.8,
    'Gleysols, Fluvisols, Cambisols, Vertisols': 0.7,
    'Vertisols': 0.9,
    'Luvisols': 0.7,
    'Cambisols and Lithosols': 0.6,
    'Nitisols and Andosols': 0.8,
    'Ferralsols with Cambisols, Lithosols and Gleysols': 0.9,
    'Acrisols with Arenosols': 0.4,
    'Solonetz': 0.3,
    'Ferralsols with Acrisols and Cambisols': 0.6,
    'Phaeozems with Luvisols and Phaeozems': 0.7,
    'Acrisols with Ferralsols': 0.6,
    'Nitisols and Phaeozems': 0.8,
    'Regosols with Lithosols and Cambisols': 0.5,
    'Fluvisols': 0.7,
    'Nitisols with Cambisols, Acrisols and Luvisols': 0.8,
    'Luvisols and Acrisols': 0.7,
    'Rankers and Lithosols': 0.6,
    'Luvisols with Ferralsols and Arenosols': 0.7,
    'Regosols with Cambisols and Lithosols': 0.6,
    'Ferralsols and Cambisols': 0.6,
    'Lithosols with Cambisols and Regosols': 0.5,
    'None': 0,
    'Planosols with Gleysols and Vertisols': 0.9,
    'Nitisols with Luvisols': 0.8,
    'Solonetz and Luvisols': 0.6,
    'Phaeozems with Nitisols': 0.8,
    'Nitisols and Cambisols': 0.8,
    'Regosols and Cambisols': 0.7,
    'Cambisols and Phaeozems': 0.6,
    'Luvisols with Acrisols and Ferralsols': 0.7,
    'Solonchaks': 0.9,
    'Lithosols and Xerosols': 0.4,
    'Lithosols and Cambisols': 0.5,
    'Lava': 0.2,
    'Cambisols and Acrisols': 0.6,
    'Ferralsols with Cambisols': 0.6,
    'Arenosols with Ferralsols, Luvisols, Planosols and Vertisols': 0.7,
    'Acrisols and Cambisols': 0.4,
    'Ferralsols with Acrisols and Arenosols': 0.7,
    'Solonetz with Xerosols, Lithosols': 0.5,
    'Phaeozems with Luvisols': 0.7,
    'Phaeozems with Cambisols and Vertisols': 0.7,
    'Gleysols, Planosols and Histosols': 0.5,
    'Lithosols with Camobisols, Regosols and Rankers': 0.6,
    'Planosols with Vertisols and Phaeozems': 0.7,
    'Phaeozems with Luvisols and Lithosols': 0.7,
    'Regosols with Solonetz': 0.5,
    'Phaeozems and Rankers': 0.6,
    'Luvisols with Phaeozems and Vertisols': 0.7,
    'Luvisols and Cambisols': 0.7,
    'Cambisols with Lithosols, Luvisols and Regosols': 0.6,
    'Arenosols': 0.4,
    'Luvisols with Lithosols': 0.6,
    'Planosols with Planosols': 0.5,
    'Phaeozems and Nitisols': 0.8,
    'Cambisols and Xerosols': 0.5,
    'Gleysols with Vertisols, Gleysols, Histosols, Rankers and Lithosols': 0.8,
    'Cambisols and Ferralsols': 0.6,
    'Gleysols and Histosols': 0.7,
    'Vertisols and Luvisols': 0.9,
    'Phaeozems with Regosols, Cambisols and Lithosols': 0.7,
    'Acrisols with Arenosols and Luvisols': 0.6,
    'Acrisols and Lithosols': 0.5,
    'Andosols and Cambisols': 0.6,
    'Vertisols and Solonchaks': 0.7,
    'Nitisols with Luvisols and Cambisols': 0.8,
    'Lithosols with Regosols, Cambisols, Arenosols, Luvisols and Acrisols': 0.7,
    'Luvisols with Arenosols': 0.6,
    'Fluvisols with Arenosols and Vertisols': 0.8,
    'Phaeozems with Cambisols, Fluvisols and Vertisols': 0.7,
    'Planosols and Vertisols': 0.6,
    'Fluvisols with Vertisols': 0.7,
    'Cambisols with Nitisols and Regosols': 0.7,
    'Solonetz with Rendzinas and Phaeozems': 0.6,
    'Acrisols with Cambisols and Acrisols and Gleysols': 0.6,
    'Acrisols with Planosols': 0.5,
    'Nitisols with Cambisols, Vertisols and Cambisols': 0.7,
    'Acrisols and Gleysols': 0.5,
    'Acrisols and Ferralsols': 0.7,
    'Cambisols with Lithosols and Rankers': 0.6,
    'Acrisols with Cambisols': 0.5,
    'Lithosols and Phaeozems': 0.6,
    'Acrisols with Solonchaks': 0.4,
}

# Define functions to map impact values
def map_drainage_to_impact(drainage_type):
    return drainage_impact_mapping.get(drainage_type, None)

def map_depth_to_impact(depth_type):
    return depth_impact_mapping.get(depth_type, None)

def map_soil_demand(soil_type):
    return soil_demand_map.get(soil_type, None)

# Apply mappings to the soil characteristics GeoDataFrame
soil_characteristics['drainage_impact'] = soil_characteristics['DRAINAGE'].apply(map_drainage_to_impact)
soil_characteristics['depth_impact'] = soil_characteristics['DEPTH'].apply(map_depth_to_impact)
soil_characteristics['soil_type_impact'] = soil_characteristics['SOILCLASS'].apply(map_soil_demand)

if grid_gdf_proj.crs != soil_characteristics.crs:
    soil_characteristics = soil_characteristics.to_crs(grid_gdf_proj.crs)

soil_characteristics = soil_characteristics.clip(grid_gdf_proj)

# Perform spatial join with all impact columns
grid_gdf_proj = gpd.sjoin(
    grid_gdf_proj,
    soil_characteristics[['geometry', 'drainage_impact', 'depth_impact', 'soil_type_impact']],
    how='left',
    predicate='intersects',
    lsuffix='left',
    rsuffix='soil'
)

# Drop unnecessary columns and reproject to WGS 84 if needed
grid_gdf_proj.drop(columns=['index_soil'], inplace=True)
grid_gdf = grid_gdf_proj.to_crs(epsg=4326)

# Calculate mean for impact columns while keeping all other columns
impact_means = grid_gdf.groupby('grid_id')[['drainage_impact', 'depth_impact', 'soil_type_impact']].transform('mean')

# Update the original DataFrame with the calculated mean values
grid_gdf[['drainage_impact', 'depth_impact', 'soil_type_impact']] = impact_means

# Drop duplicate rows for each grid_id, keeping the first occurrence and updated mean values
grid_gdf = grid_gdf.drop_duplicates(subset=['grid_id'])
grid_gdf = grid_gdf.applymap(lambda x: np.nan if x == '--' else x)
grid_gdf = grid_gdf.replace('--', np.nan)
# Display the result
print("Spatial join with all impact scores completed.")

# Assuming grid_gdf is your GeoDataFrame
#column_names = grid_gdf.columns.tolist()  # Get all column names as a list
#print(column_names)  # Print the list of column names
# Create a DataFrame from the list of column names
#column_names_df = pd.DataFrame(column_names, columns=['Layer'])
# Save the DataFrame to a CSV file
#column_names_df.to_csv('Demand_Weights.csv', index=False)
#print("Column names saved to 'Demand_Weights.csv'")

#Save to CSV
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_csv = f"F:\\AOI\\Data\\grid_statistics_{base_name}.csv"
grid_gdf.to_csv(output_csv, index=False)
print(f"Data saved to {output_csv}")
grid_gdf.to_csv("test.csv", index=False)
