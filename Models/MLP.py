import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

# Define the MLP model
def build_mlp(input_dim, output_dim, hidden_layers, dropout_rate=0.2):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(output_dim, activation='relu'))
    return model

# Define input and output dimensions


# Build the MLP model
mlp_model = build_mlp(input_dim, output_dim, hidden_layers)

# Create dummy data
dummy_data = np.random.random((1, input_dim))

# Build the model by passing data through it
mlp_model(dummy_data)

# Compile the model
mlp_model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

# Print the summary
mlp_model.summary()
