import numpy as np

x = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([2, 4, 5, 4, 5])  # Dependent variable

# Number of observations
n = len(x)

# Calculating slope (b1) and intercept (b0)
mean_x = np.mean(x)
mean_y = np.mean(y)

numerator = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sum((x - mean_x) ** 2)

b1 = numerator / denominator
b0 = mean_y - b1 * mean_x

# Predictions
y_pred = b0 + b1 * x

print(f"Slope (b1): {b1}")
print(f"Intercept (b0): {b0}")
print(f"Predicted values: {y_pred}")