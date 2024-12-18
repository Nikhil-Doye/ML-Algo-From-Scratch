# ML Algorithms and Neural Networks from Scratch

This repository consists of implementations of machine learning algorithms and neural network architectures built from scratch using Python. No external libraries like Scikit-learn or TensorFlow/Keras are being used for the core algorithms, though NumPy might be used for numerical operations.

## Repository Structure

```
ML-Algorithms-From-Scratch/
|-- simple_linear_regression.py
|-- multiple_linear_regression.py
|-- logistic_regression.py
|-- decision_tree.py
|-- random_forest.py
|-- kmeans.py
|-- neural_network/
|   |-- perceptron.py
|   |-- feedforward_nn.py
|   |-- backpropagation.py
|   |-- cnn.py
|   |-- rnn.py
|-- utils/
|   |-- data_processing.py
|   |-- metrics.py
|   |-- visualizations.py
|-- README.md
|-- requirements.txt
|-- LICENSE
```

## Example Algorithms to Include

### Machine Learning
1. **Linear Regression**
   - Implementation of simple and multiple linear regression using the least squares method.

2. **Logistic Regression**
   - Binary and multi-class logistic regression with gradient descent optimization.

3. **Decision Tree**
   - Implementation of a CART-based decision tree for regression and classification.

4. **Random Forest**
   - Building an ensemble of decision trees using bagging.

5. **K-Means Clustering**
   - Unsupervised learning algorithm to partition data into k clusters.

### Neural Networks
1. **Perceptron**
   - Implementation of a single-layer perceptron for binary classification.

2. **Feedforward Neural Network**
   - Fully connected neural network with multiple layers.

3. **Backpropagation**
   - Gradient descent-based optimization for training feedforward networks.

4. **Convolutional Neural Network (CNN)**
   - Implementation of basic CNN layers: convolution, pooling, and fully connected layers.

5. **Recurrent Neural Network (RNN)**
   - Implementation of a simple RNN with forward propagation.

## Utilities

1. **Data Processing**
   - Feature scaling, one-hot encoding, and train-test splitting.

2. **Metrics**
   - Functions for calculating accuracy, precision, recall, F1 score, mean squared error, etc.

3. **Visualizations**
   - Plotting training curves, decision boundaries, and cluster assignments.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ML-Algorithms-From-Scratch.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run an algorithm, e.g., linear regression:
```bash
python linear_regression.py
```

## Contributions

Contributions are welcome! Please follow the guidelines mentioned in `CONTRIBUTING.md` (to be created).

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any questions, please open an issue or contact me at [doye.n@northeastern.edu].
