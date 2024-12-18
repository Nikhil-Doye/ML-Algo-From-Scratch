import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_processing import generate_binary_classification_data, split_and_standardize_data
from logistic_regression.binary_logistic_regression import BinaryLogisticRegression
from utils.metrics import evaluate_classification_model

def binary_classification_demo():
    """Run a demo for binary logistic regression."""
    # Generate synthetic data
    X, y = generate_binary_classification_data()
    X_train, X_test, y_train, y_test = split_and_standardize_data(X, y)

    # Train binary logistic regression model
    model = BinaryLogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate the model
    evaluate_classification_model(y_test, predictions)

if __name__ == "__main__":
    print("Binary Classification Demo:")
    binary_classification_demo()