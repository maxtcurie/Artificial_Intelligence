import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, alpha=0):
        self.weight = None
        self.bias = None
        self.alpha = alpha

    def fit(self, X, y):
        X = np.vstack((X, np.ones(len(X)))).T  # Add a column of ones for the bias term
        num_features = X.shape[1]

        # Compute the closed-form solution with L2 regularization
        X_transpose_X = np.dot(X.T, X)
        regularization = self.alpha * np.eye(num_features)  # Identity matrix scaled by alpha
        self.weight, self.bias = np.linalg.inv(X_transpose_X + regularization).dot(X.T).dot(y)

    def predict(self, X):
        return self.weight * X + self.bias

# Toy dataset with noise
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Add Gaussian noise to X
noise = np.random.normal(0, 1, X.shape)
X_with_noise = X + noise

# Create and fit the linear regression model
model = LinearRegression(alpha=0.1)
model.fit(X_with_noise, y)
predictions = model.predict(X_with_noise)

# Scatter plot of actual data points with noise and overlaying the predicted line
plt.scatter(X_with_noise, y, label='Actual with Noise')
plt.plot(X_with_noise, predictions, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Noise')
plt.legend()
plt.show()
