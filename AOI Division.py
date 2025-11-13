import geopandas as gpd
import math
from shapely.geometry import box

# Load your AOI shapefile
aoi_gdf = gpd.read_file(r"F:\AOI\Makueni County.shp")

# Calculate the bounding box of the AOI
minx, miny, maxx, maxy = aoi_gdf.total_bounds

# Desired number of parts (increase to reduce polygon size)
num_parts = 10

# Calculate rows and columns to approximate the number of parts
rows = int(math.sqrt(num_parts))
cols = int(math.ceil(num_parts / rows))

# Recalculate width and height of each grid cell
width = (maxx - minx) / cols
height = (maxy - miny) / rows

# Create a list to store each grid part
grid_parts = []

# Generate grid cells and split the AOI
for i in range(cols):
    for j in range(rows):
        # Define the bounding box of the grid cell
        cell_minx = minx + i * width
        cell_maxx = minx + (i + 1) * width
        cell_miny = miny + j * height
        cell_maxy = miny + (j + 1) * height
        cell_box = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
        
        # Intersect the cell with AOI and get the resulting geometry
        cell_gdf = gpd.clip(aoi_gdf, cell_box)
        
        # Save the non-empty cells to a GeoJSON file
        if not cell_gdf.empty:
            cell_gdf.to_file(f"F:\\AOI\\AOI_Part\\Part_{i}_{j}.geojson", driver="GeoJSON")
            grid_parts.append(cell_gdf)

print("Division and saving of GeoJSON files completed.")
