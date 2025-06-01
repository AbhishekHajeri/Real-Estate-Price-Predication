import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = 'Housing-dataset.csv'  # Update this to the correct path
data = pd.read_csv(file_path)

# Display dataset information
print("Dataset Overview:")
print(data.info())
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Rename columns for clarity
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

# Drop the 'ID' column as it is not relevant for modeling
data.drop(columns=['ID'], inplace=True)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove duplicate rows
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
if duplicates > 0:
    data.drop_duplicates(inplace=True)
    print("Duplicate rows removed.")

# Statistical summary of the dataset
print("\nStatistical Summary:")
print(data.describe())

# Scale numerical features using standard Scaling
scaler = MinMaxScaler()
scaled_features = ['House_Age', 'Distance_to_MRT', 'Number_of_Stores', 'Latitude', 'Longitude']
data[scaled_features] = scaler.fit_transform(data[scaled_features])
print("\nFeatures scaled using Min-Max Scaling.")

scaler_save_path='preprocessing_scaler.pkl'
with open(scaler_save_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Saved the preprocessing_scaler.pkl file")

# Log-transform skewed features (e.g., Distance_to_MRT)
data['Distance_to_MRT'] = np.log1p(data['Distance_to_MRT'])
print("\n'Log1p' transformation applied to 'Distance_to_MRT'.")

# Correlation Heatmap: Identify redundant features
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Target Variable Analysis
plt.figure(figsize=(8, 5))
sns.histplot(data['House_Price_per_Unit_Area'], kde=True, color='blue')
plt.title('Distribution of House Price per Unit Area')
plt.xlabel('House Price per Unit Area')
plt.ylabel('Frequency')
plt.show()

# Save the preprocessed dataset for further use
data.to_csv('preprocessed_real_estate.csv', index=False)
print("\nPreprocessed dataset saved as 'preprocessed_real_estate.csv'")

# Display the first few rows of the preprocessed data
print("\nFirst 5 rows of the preprocessed dataset:")
print(data.head())

# Detect outliers using Z-score
threshold = 3  # Define a Z-score threshold
z_scores = np.abs(zscore(data))
outliers = np.where(z_scores > threshold)

print(f"Number of outliers detected: {len(outliers[0])}")

# Remove outliers
data = data[(z_scores < threshold).all(axis=1)]
print("Outliers removed.")

# Drop the target variable for unsupervised learning
features = data.drop(columns=['House_Price_per_Unit_Area'])

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Define the autoencoder architecture
input_dim = scaled_data.shape[1]  # Number of input features

autoencoder = Sequential([
    Dense(64, input_dim=input_dim, activation='relu'),
    Dense(32, activation='relu'),
    Dense(8, activation='relu'),  # Latent space representation
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(input_dim, activation='sigmoid')  # Output layer
])

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Train the autoencoder
history = autoencoder.fit(
    scaled_data, scaled_data, 
    epochs=50, batch_size=32, validation_split=0.2, verbose=1
)

# Visualize the training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
