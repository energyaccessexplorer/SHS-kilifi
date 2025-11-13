import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time

# Directory path where the CSV files are stored
directory_path = r'F:\AOI\Data'

# List to hold all DataFrames
df_list = []

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        # Read the CSV file and append to the list
        df_list.append(pd.read_csv(file_path))

# Concatenate all DataFrames in the list into one DataFrame
df = pd.concat(df_list, ignore_index=True)

# Display the shape of the concatenated DataFrame
print(f"Concatenated DataFrame shape: {df.shape}")

# Convert 'geometry' column to geometry objects
df['geometry'] = df['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs("EPSG:4326", inplace=True)
gdf.fillna(0, inplace=True)

# Load weights and calculate 'estimated_electricity_demand'
weights_df = pd.read_csv(r"F:/Code/Demand_Weights.csv")
weights_dict = dict(zip(weights_df['Layer'], weights_df['Weight']))

# Identify numeric columns for scaling and check for missing weights
numeric_columns = gdf.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = numeric_columns[numeric_columns != 'grid_id']  # Exclude ID columns
missing_weights = [col for col in numeric_columns if col not in weights_dict]
if missing_weights:
    print(f"Warning: Missing weights for columns: {missing_weights}")

# Scale numeric columns
scaler = MinMaxScaler()
start_time_scaling = time.time()
scaled_data = scaler.fit_transform(gdf[numeric_columns])
scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns, index=gdf.index)
print(f"Feature scaling took {time.time() - start_time_scaling:.2f} seconds")

# Calculate 'estimated_electricity_demand'
scaled_df['estimated_electricity_demand'] = scaled_df.apply(
    lambda row: sum(row[layer] * weights_dict.get(layer, 0) for layer in weights_dict if layer in row),
    axis=1
)

# Concatenate scaled data back to GeoDataFrame
gdf = pd.concat([gdf.drop(columns=numeric_columns), scaled_df], axis=1)

# Split data into features (X) and target (y)
X = gdf.drop(columns=['geometry', 'grid_id', 'estimated_electricity_demand'])  # Drop non-feature columns
y = gdf['estimated_electricity_demand']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and test sets")

# Reduce memory usage during hyperparameter tuning (optional)
X_train_sampled = X_train.sample(frac=0.1, random_state=42)
y_train_sampled = y_train.loc[X_train_sampled.index]

# Define hyperparameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10]
}

# Initialize Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)

# Perform RandomizedSearchCV
print("Performing hyperparameter tuning...")
random_search = RandomizedSearchCV(
    gb_model, 
    param_distributions=param_grid, 
    n_iter=20, 
    cv=3, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    random_state=42
)

start_time_tuning = time.time()
random_search.fit(X_train_sampled, y_train_sampled)  # Using sampled data for tuning
print(f"Hyperparameter tuning completed in {time.time() - start_time_tuning:.2f} seconds")

# Best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_
print(f"Best parameters: {best_params}")

# Evaluate the tuned model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTuned Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.10f}")
print(f"Mean Squared Error (MSE): {mse:.10f}")
print(f"R-squared (R2): {r2:.10f}")

# Save the tuned model
model_path = r"F:\Code\tuned_electricity_demand_model.pkl"
joblib.dump(best_model, model_path)
print(f"Tuned model saved to {model_path}")  
