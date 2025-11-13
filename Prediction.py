import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn.preprocessing import MinMaxScaler
import joblib
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import time

# Define paths
input_dir = r"E:\AOI\Data\500"  # Directory for input CSV files
output_dir = r"E:/AOI/Predicted_Raster"  # Directory for output raster files
model_path = r"E:/Code/tuned_electricity_demand_model.pkl"  # Saved model path
weights_path = r"E:/Code/Demand_Weights.csv"  # Path to weights file

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the trained model
model = joblib.load(model_path)

# Load weights for 'estimated_electricity_demand'
weights_df = pd.read_csv(weights_path)
weights_dict = dict(zip(weights_df['Layer'], weights_df['Weight']))

# Initialize MinMaxScaler (will be refitted per input data to match training workflow)
scaler = MinMaxScaler()

# Process each CSV file
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        input_file = os.path.join(input_dir, filename)
        print(f"\nProcessing file: {filename}")
        start_time = time.time()

        # Load the CSV file
        data = pd.read_csv(input_file)

        # Convert 'geometry' to GeoDataFrame
        data['geometry'] = data['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(data, geometry='geometry')
        gdf.set_crs("EPSG:4326", inplace=True)
        gdf.fillna(0, inplace=True)

        # Reproject to a CRS with meters for rasterization (EPSG:32737)
        gdf = gdf.to_crs(epsg=32737)

        # Identify numeric columns for scaling
        numeric_columns = gdf.select_dtypes(include=['float64', 'int64']).columns
        numeric_columns = numeric_columns[numeric_columns != 'grid_id']

        # Scale numeric columns
        scaled_data = scaler.fit_transform(gdf[numeric_columns])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns, index=gdf.index)

        # Calculate 'estimated_electricity_demand'
        scaled_df['estimated_electricity_demand'] = scaled_df.apply(
            lambda row: sum(row[layer] * weights_dict.get(layer, 0) for layer in weights_dict if layer in row),
            axis=1
        )

        # Prepare features for prediction
        features = scaled_df.drop(columns=['estimated_electricity_demand'])

        # Predict using the loaded model
        y_pred = model.predict(features)
        
        y_pred_rescaled = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        # Add predictions to the original GeoDataFrame
        gdf['predicted_electricity_demand'] = y_pred_rescaled

        # Define rasterization parameters
        cell_size = 500  # Resolution in meters
        minx, miny, maxx, maxy = gdf.total_bounds
        width = int((maxx - minx) / cell_size)
        height = int((maxy - miny) / cell_size)

        # Define transform for rasterization
        transform = from_origin(minx, maxy, cell_size, cell_size)

        # Rasterize predicted electricity demand
        raster = rasterize(
            ((geom, value) for geom, value in zip(gdf.geometry, gdf['predicted_electricity_demand'])),
            out_shape=(height, width),
            transform=transform,
            fill=-9999,  # Default value for NoData
            dtype='float32'
        )

        # Define output raster file path
        output_raster_path = os.path.join(output_dir, f"predicted_{os.path.splitext(filename)[0]}.tif")

        # Save the raster to a GeoTIFF file
        with rasterio.open(
            output_raster_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=gdf.crs,
            transform=transform,
            nodata=-9999,
            compress='LZW'  # Apply compression to reduce file size
        ) as dst:
            dst.write(raster, 1)

        print(f"Raster saved to: {output_raster_path}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")