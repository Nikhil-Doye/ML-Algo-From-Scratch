from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_classification_model(y_true, y_pred):
    """Evaluate a classification model and print accuracy, confusion matrix, and classification report."""
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)