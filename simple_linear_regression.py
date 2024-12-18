import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load the California Housing Dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # Adding target variable

# Target variable
y = df['MedHouseVal'].values

# Predicting 'MedHouseVal' using 'MedInc' (Median Income)
x_simple = df['MedInc'].values

# Mean of x and y
mean_x = np.mean(x_simple)
mean_y = np.mean(y)

# Calculate slope (b1) and intercept (b0)
b1 = np.sum((x_simple - mean_x) * (y - mean_y)) / np.sum((x_simple - mean_x) ** 2)
b0 = mean_y - b1 * mean_x

# Predictions
y_pred_simple = b0 + b1 * x_simple

# Output Results
print("Simple Linear Regression Results:")
print(f"Slope (b1): {b1:.4f}")
print(f"Intercept (b0): {b0:.4f}")