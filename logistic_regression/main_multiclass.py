import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_processing import load_iris_classification_data, split_and_standardize_data
from logistic_regression.multiclass_logistic_regression import MultiClassLogisticRegression
from utils.metrics import evaluate_classification_model

def multi_class_classification_demo():
    """Run a demo for multi-class logistic regression using the Iris dataset."""
    # Load Iris dataset
    X, y = load_iris_classification_data()
    X_train, X_test, y_train, y_test = split_and_standardize_data(X, y)

    # Train multi-class logistic regression model
    model = MultiClassLogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate the model
    evaluate_classification_model(y_test, predictions)

if __name__ == "__main__":
    print("Multi-Class Classification Demo:")
    multi_class_classification_demo()