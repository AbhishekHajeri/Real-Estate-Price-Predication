import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the original dataset
file_path = 'Housing-dataset.csv'  # Update to the path of your original dataset
data = pd.read_csv(file_path)

# Rename columns for clarity if needed
data.columns = [
    'ID',
    'Transaction_Date',
    'House_Age',
    'Distance_to_MRT',
    'Number_of_Stores',
    'Latitude',
    'Longitude',
    'House_Price_per_Unit_Area'
]

# Select the features to scale (exclude Transaction_Date and House_Price_per_Unit_Area)
features_to_scale = ['House_Age', 'Distance_to_MRT', 'Number_of_Stores', 'Latitude', 'Longitude']

# Apply Min-Max Scaling to the selected features
scaler = MinMaxScaler()
data_scaled = data.copy()  # Create a copy to preserve original data
data_scaled[features_to_scale] = scaler.fit_transform(data[features_to_scale])
print(data_scaled)

# Save the scaler for use in app.py
scaler_save_path = 'model/preprocessing_scaler.pkl'
with open(scaler_save_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"Scaler saved to {scaler_save_path}")

