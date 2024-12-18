import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_binary_classification_data(n_samples=1000, n_features=4, random_state=42):
    """Generate synthetic data for binary classification tasks."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=random_state)
    return X, y

def load_iris_classification_data():
    """Load the Iris dataset for multi-class classification tasks."""
    data = load_iris()
    return data.data, data.target

def split_and_standardize_data(X, y, test_size=0.2, random_state=42):
    """Split the data into train/test sets and standardize the features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test