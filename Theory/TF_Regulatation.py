import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate toy dataset
X = np.linspace(-5, 5, 100)
y = 2 * X + np.random.normal(0, 1, X.shape)

# Split dataset into training and testing sets
X_train, y_train = X[:80], y[:80]
X_test, y_test = X[80:], y[80:]

# Reshape the data
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1], kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Fit the model without regularization
history_without_reg = model.fit(X_train, y_train, epochs=100, verbose=0)

# Fit the model with L2 regularization
history_with_reg = model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

# Evaluate the model on the test set
loss_without_reg = model.evaluate(X_test, y_test)
predictions_without_reg = model.predict(X_test)

# Reset the model weights
model.set_weights(model.get_weights())

# Compile the model again
model.compile(optimizer='adam', loss='mse')

# Fit the model with L2 regularization
history_with_reg = model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)], validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss_with_reg = model.evaluate(X_test, y_test)
predictions_with_reg = model.predict(X_test)

# Sort X_test and predictions for proper plotting
sorted_indices = np.argsort(X_test.flatten())
X_test_sorted = X_test[sorted_indices]
predictions_without_reg_sorted = predictions_without_reg[sorted_indices]
predictions_with_reg_sorted = predictions_with_reg[sorted_indices]

# Plot the results
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test_sorted, predictions_without_reg_sorted, color='red', label='Without Regularization')
plt.plot(X_test_sorted, predictions_with_reg_sorted, color='blue', label='With Regularization')
plt.plot(X_test, 2 * X_test, color='black', linestyle='--', label='True Relationship')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with and without Regularization')
plt.legend()
plt.show()

print("Loss without regularization:", loss_without_reg)
print("Loss with regularization:", loss_with_reg)
