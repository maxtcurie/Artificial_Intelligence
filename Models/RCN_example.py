import numpy as np
from pyrcn.echo_state_network import ESNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate a combined sine and cosine wave
x = np.linspace(0, 50, 1000) # 1000 points from 0 to 50
y = np.sin(x) * np.cos(x * 0.5) + np.sin(x * 0.3)

# Create sequences
sequence_length = 20
predict_length = 5
X = []
Y = []
for i in range(len(y) - sequence_length-predict_length):
    X.append(y[i:i+sequence_length])
    Y.append(y[i+sequence_length:i+sequence_length+predict_length])

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

# Flatten X for training
X_flattened = X.reshape(X.shape[0], -1)  # Flattening each sequence into a single array

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened, Y, test_size=0.3, random_state=42)

# Create an Echo State Network Regressor
esn = ESNRegressor(random_state=42)

# Train the ESN
esn.fit(X_train, y_train)

# Predict using the ESN
y_pred = esn.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values', color='blue')
plt.plot(y_pred, label='Predicted Values', color='red', linestyle='dashed')
plt.title(f'Comparison of True and Predicted Values (MSE: {mse:.4f})')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
plt.show()


# Predict using the ESN
y_pred = esn.predict(X)

# Calculate Mean Squared Error
mse = mean_squared_error(Y, y_pred)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Y[:,-3], label='True Values', color='blue')
plt.plot(y_pred[:,-3], label='Predicted Values', color='red', linestyle='dashed')
plt.title(f'Comparison of True and Predicted Values (MSE: {mse:.4f})')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
plt.show()
