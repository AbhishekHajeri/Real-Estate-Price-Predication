import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
file_path = 'preprocessed_real_estate.csv'  # Ensure this file exists in the same directory
data = pd.read_csv(file_path)

# Split features and target
transaction_date = data['Transaction_Date'].values.reshape(-1, 1)  # Keep it as-is
X = data.drop(columns=['House_Price_per_Unit_Area'])
y = data['House_Price_per_Unit_Area']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the individual regressors
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning for RandomForestRegressor and GradientBoostingRegressor
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

# GridSearchCV for RandomForest
grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"Best Random Forest Parameters: {grid_rf.best_params_}")

# GridSearchCV for GradientBoosting
grid_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_
print(f"Best Gradient Boosting Parameters: {grid_gb.best_params_}")

# Combine regressors using Voting Regressor
voting_model = VotingRegressor([('lr', lr), ('rf', best_rf), ('gb', best_gb)])

# Train the Voting Regressor
voting_model.fit(X_train, y_train)

# Evaluate the model on test data
predictions = voting_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Voting Regressor MSE: {mse:.4f}")
print(f"Voting Regressor R² Score: {r2:.4f}")

# Cross-validation on Voting Regressor
cv_scores = cross_val_score(voting_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_mse = -np.mean(cv_scores)
print(f"Cross-Validated MSE: {mean_cv_mse:.4f}")

# Save the model and scaler for Flask app
model_save_path = 'model/voting_model.pkl'

with open(model_save_path, 'wb') as model_file:
    pickle.dump(voting_model, model_file)

print(f"Model saved at: {model_save_path}")
