import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate mock time series data
def generate_mock_data(n_samples, n_timesteps, n_features):
    X = np.random.rand(n_samples, n_timesteps, n_features)
    y = np.random.rand(n_samples, n_features)
    return X, y

# Parameters
n_samples = 1000
n_timesteps = 10
n_features = 1

# Generate data
X, y = generate_mock_data(n_samples, n_timesteps, n_features)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dense(units=n_features))

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Predict using the trained model
predictions = model.predict(X)

print("Predictions:", predictions)
