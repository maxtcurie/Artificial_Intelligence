import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

# Define the Encoder
def build_encoder(input_dim, encoding_dim):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(encoding_dim, activation='sigmoid'))
    return model

# Define the Decoder
def build_decoder(encoding_dim, output_dim):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(encoding_dim,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    return model

# Define input and output dimensions
input_dim = 784  # Example dimension, replace with actual input dimension
encoding_dim = 32  # Example encoding dimension
output_dim = input_dim  # Output dimension should match input dimension for autoencoders

# Build encoder and decoder
encoder = build_encoder(input_dim, encoding_dim)
decoder = build_decoder(encoding_dim, output_dim)

# Combine encoder and decoder into an autoencoder model
autoencoder = models.Sequential([encoder, decoder])

# Create dummy data
dummy_data = np.random.random((1, input_dim))

# Build the model by passing data through it
autoencoder(dummy_data)

# Compile the model
autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

# Print the summary
autoencoder.summary()
