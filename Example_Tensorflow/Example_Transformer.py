import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class SpectrogramTransformerAutoencoder(tf.keras.Model):
    def __init__(self, n_freq, n_time, n_channel, d_model=128, num_heads=4, num_layers=4):
        super(SpectrogramTransformerAutoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')
        ])

        self.positional_encoder = PositionalEncoder(d_model)

        self.transformer_encoder = tf.keras.layers.TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            name="transformer_encoder"
        )

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2DTranspose(n_channel, kernel_size=3, strides=1, padding='same', activation='relu')
        ])

    def call(self, inputs):
        x = self.encoder(inputs)
        x = tf.transpose(x, perm=[2, 0, 1, 3])  # Reshape for transformer input
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x = tf.transpose(x, perm=[1, 2, 0, 3])  # Reshape for decoder input
        x = self.decoder(x)
        return x

class PositionalEncoder(layers.Layer):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.position_enc = self._get_positional_encoding(max_len)

    def _get_positional_encoding(self, max_len):
        position = tf.expand_dims(tf.range(0, max_len, dtype=tf.float32), axis=1)
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.d_model))
        pos_enc = tf.zeros((max_len, self.d_model), dtype=tf.float32)
        pos_enc[:, 0::2] = tf.sin(position * div_term)
        pos_enc[:, 1::2] = tf.cos(position * div_term)
        pos_enc = tf.expand_dims(pos_enc, axis=0)
        return pos_enc

    def call(self, x):
        seq_len = tf.shape(x)[2]
        x = x + self.position_enc[:, :seq_len]
        return x


# Generate synthetic dataset
def generate_spectrograms(n_samples, n_freq, n_time, n_channel):
    dataset = []
    for _ in range(n_samples):
        spectrogram = np.random.rand(n_freq, n_time, n_channel)
        dataset.append(spectrogram)
    return np.array(dataset)

# Define constants and hyperparameters
n_samples = 1000
n_freq = 128
n_time = 256
n_channel = 3
d_model = 128
num_heads = 4
num_layers = 4
batch_size = 32
epochs = 10

# Generate dataset
dataset = generate_spectrograms(n_samples, n_freq, n_time, n_channel)

# Create model
model = SpectrogramTransformerAutoencoder(n_freq, n_time, n_channel, d_model, num_heads, num_layers)

# Compile model
model.compile(optimizer='adam', loss='mse')

# Convert dataset to TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices(dataset)
dataset = dataset.batch(batch_size)

# Train model
model.fit(dataset, epochs=epochs)




# Select a few samples from the dataset
num_samples_to_plot = 5
samples = dataset.take(num_samples_to_plot)

# Reconstruct the spectrograms using the trained model
reconstructed_samples = model.predict(samples)

# Plot the original and reconstructed spectrograms
fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 10))
fig.tight_layout()

for i in range(num_samples_to_plot):
    # Plot original spectrogram
    axes[i, 0].imshow(samples[i].numpy().squeeze(), cmap='jet', origin='lower')
    axes[i, 0].set_title('Original Spectrogram')

    # Plot reconstructed spectrogram
    axes[i, 1].imshow(reconstructed_samples[i].squeeze(), cmap='jet', origin='lower')
    axes[i, 1].set_title('Reconstructed Spectrogram')

plt.show()
