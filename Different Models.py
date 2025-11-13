import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
from shapely import wkt
import time

# Load data
df = pd.read_csv("F:/AOI/grid_statistics_Makueni County.csv") #Grid Size 500 m
df['geometry'] = df['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs("EPSG:4326", inplace=True)  # Set coordinate reference system
gdf.fillna(0, inplace=True)  # Replace missing values with 0

# Load weights and calculate 'estimated_electricity_demand'
weights_df = pd.read_csv("F:/Code/Demand_Weights.csv")
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

# Initialize models
models = {
    "K-Neighbors Regression": KNeighborsRegressor(n_neighbors=5),
    "Support Vector Regression": SVR(kernel='rbf', C=1, epsilon=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42)
}

# Perform hyperparameter tuning using RandomizedSearchCV for faster execution
tuned_models = {}
for model_name in ["Gradient Boosting", "XGBoost"]:
    if model_name == "Gradient Boosting":
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    elif model_name == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    
    random_search = RandomizedSearchCV(models[model_name], param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error', random_state=42)
    start_time_random_search = time.time()
    random_search.fit(X_train, y_train)
    tuned_models[model_name] = random_search.best_estimator_
    print(f"{model_name} randomized search took {time.time() - start_time_random_search:.2f} seconds")

# Update the models dictionary with tuned models
models.update(tuned_models)

# Train and evaluate models
results = {}
feature_importances = {}

for model_name, model in models.items():
    # Train model with cross-validation
    start_time_train = time.time()
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    model.fit(X_train, y_train)
    print(f"{model_name} training took {time.time() - start_time_train:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results[model_name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "CV MAE": -cross_val_scores.mean()}
    
    # Store feature importance for ensemble models
    if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        feature_importances[model_name] = model.feature_importances_

# Display results
print("\nModel Performance Metrics:")
for model_name, metrics in results.items():
    print(f"{model_name}: MAE={metrics['MAE']:.10f}, MSE={metrics['MSE']:.10f}, RMSE={metrics['RMSE']:.10f}, CV MAE={metrics['CV MAE']:.10f}")
