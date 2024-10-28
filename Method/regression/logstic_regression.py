#https://chatgpt.com/share/e/66fc9c59-b20c-8003-98e1-6fcf61608dd5
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function (binary cross-entropy)
def compute_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Gradient descent for logistic regression
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    # Number of samples and features
    n_samples, n_features = X.shape
    
    # Initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0
    
    # Gradient descent
    for _ in range(iterations):
        # Linear model
        linear_model = np.dot(X, weights) + bias
        
        # Apply sigmoid function to get predictions
        y_pred = sigmoid(linear_model)
        
        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias

# Prediction function
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return [1 if i > 0.5 else 0 for i in y_pred]

# Example usage
if __name__ == "__main__":
    # Create dummy data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 0, 1, 1])  # Labels
    
    # Train logistic regression model
    weights, bias = logistic_regression(X, y, learning_rate=0.1, iterations=1000)
    
    # Print weights and bias
    print("Weights:", weights)
    print("Bias:", bias)
    
    # Make predictions
    predictions = predict(X, weights, bias)
    print("Predictions:", predictions)
    print("Real:       ", y)
