import numpy as np
from logistic_regression.binary_logistic_regression import BinaryLogisticRegression

class MultiClassLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.classifiers = []

    def fit(self, X, y):
        """Train the multi-class logistic regression model using the One-vs-Rest strategy."""
        self.unique_classes = np.unique(y)
        for cls in self.unique_classes:
            binary_y = np.where(y == cls, 1, 0)
            clf = BinaryLogisticRegression(self.learning_rate, self.epochs)
            clf.fit(X, binary_y)
            self.classifiers.append((cls, clf))

    def predict(self, X):
        """Predict class labels for input data."""
        predictions = []
        for cls, clf in self.classifiers:
            preds = clf.sigmoid(np.dot(X, clf.weights) + clf.bias)
            predictions.append(preds)
        predictions = np.array(predictions).T
        return self.unique_classes[np.argmax(predictions, axis=1)]