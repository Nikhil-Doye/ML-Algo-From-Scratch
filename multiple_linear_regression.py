import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load the California Housing Dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # Adding target variable

# Target variable
y = df['MedHouseVal'].values

X = df.drop(columns=['MedHouseVal']).values  # Independent variables
n_samples, n_features = X.shape

# Add a column of ones to X for the intercept term
X_with_intercept = np.hstack([np.ones((n_samples, 1)), X])

# Solve for coefficients using the Normal Equation: B = (X^T X)^(-1) X^T y
B = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

# Predictions
y_pred_multiple = X_with_intercept @ B

# Output Results
print("Multiple Linear Regression Results:")
print("Coefficients (including intercept):")
print(B)