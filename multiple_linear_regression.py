import numpy as np
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])

# Adding intercept
X = np.c_[np.ones(X.shape[0]), X]

# Coefficients
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# Predictions
y_pred = X @ beta

print(f"Coefficients: {beta}")
print(f"Predicted values: {y_pred}")

mse = np.mean((y - y_pred) ** 2)
print("Mean Squared Error:", mse)
# R-squared
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2_score = 1 - (ss_residual / ss_total)
print("R² Score:", r2_score)