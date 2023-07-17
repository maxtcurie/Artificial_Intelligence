import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.activations import gelu
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Create a custom positional encoding layer
class PositionalEncoding(layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]

        position_enc = np.array([
            [pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
            for pos in range(seq_length)
        ])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # Apply sine to even indices
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # Apply cosine to odd indices

        position_enc = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        position_enc = tf.expand_dims(position_enc, 0)

        return inputs + position_enc

# Create the Transformer model
def transformer_model(input_shape, vocab_size, num_blocks=1, d_model=128, num_heads=8, dff=512, dropout_rate=0.1):
    # Input layer
    inputs = Input(shape=input_shape)

    # Positional encoding
    x = PositionalEncoding()(inputs)

    # Transformer blocks
    for _ in range(num_blocks):
        # Multi-head self-attention
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward neural network
        ffn = tf.keras.Sequential([
            layers.Dense(dff, activation=gelu),
            layers.Dense(d_model)
        ])
        x = ffn(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Output layer
    outputs = Dense(vocab_size, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define the input shape and vocabulary size
input_shape = (512, 512)
vocab_size = 10000

# Create an instance of the Transformer model
transformer = transformer_model(input_shape, vocab_size)

# Compile the model
optimizer = Adam(learning_rate=1e-4)
loss = SparseCategoricalCrossentropy(from_logits=False)
metric = SparseCategoricalAccuracy()
transformer.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Print the model summary
transformer.summary()


import numpy as np

# Generate random training data
num_samples = 1000
input_shape = (512, 512)
vocab_size = 10000

# Generate random input sequences
X_train = np.random.randint(0, vocab_size, size=(num_samples,) + input_shape)

# Generate random labels (0 for negative, 1 for positive)
y_train = np.random.randint(0, 2, size=(num_samples,))

# Generate random test data
num_samples_test = 10

# Generate random input sequences for testing
X_test = np.random.randint(0, vocab_size, size=(num_samples_test,) + input_shape)

# Make predictions on test data
y_pred = transformer.predict(X_test)

# Convert predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Display the predicted labels
print("Predicted Labels:", y_pred_labels)
