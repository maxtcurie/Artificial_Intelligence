import numpy as np
from scipy.linalg import pinv
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix


class ExtremeLearningMachine:
    def __init__(self, input_size, hidden_size, activation_function='sigmoid', sparsity=0.95):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_function = activation_function

        # Create sparse random weights matrix W (with sparsity control)
        self.W = sparse_random(self.hidden_size, self.input_size, density=1 - sparsity, format='csr')  # Sparse matrix
        self.b = np.random.randn(self.hidden_size)  # biases for the hidden layer

    def activation(self, X):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-X))  # Sigmoid activation
        elif self.activation_function == 'relu':
            return np.maximum(0, X)  # ReLU activation
        elif self.activation_function == 'tanh':
            return np.tanh(X)  # Tanh activation
        else:
            raise ValueError("Unsupported activation function")

    def fit(self, X_train, y_train):
        # Step 1: Compute the hidden layer output matrix
        # Multiply the sparse matrix W with X_train
        H = self.activation(X_train @ self.W.T + self.b)  # Matrix multiplication (X_train @ W.T)

        # Step 2: Compute the output weights using the Moore-Penrose pseudoinverse
        self.beta = np.dot(pinv(H), y_train)

    def predict(self, X_test):
        # Compute the hidden layer output for test data
        H_test = self.activation(X_test @ self.W.T + self.b)

        # Compute the predictions
        return np.dot(H_test, self.beta)


# Example usage:

# Generate some random data
np.random.seed(42)
X_train = np.random.randn(100, 10)  # 100 samples, 10 features
y_train = np.random.randn(100, 1)  # 100 samples, 1 target

# Initialize ELM with 10 input nodes and 20 hidden nodes
# Here we choose sparsity as 90% (which means W will be 90% sparse, or 10% non-zero elements)
elm = ExtremeLearningMachine(input_size=10, hidden_size=20, activation_function='sigmoid', sparsity=0.9)

# Train the model
elm.fit(X_train, y_train)

# Make predictions on new data
X_test = np.random.randn(10, 10)  # 10 new samples
predictions = elm.predict(X_test)

print(predictions)
