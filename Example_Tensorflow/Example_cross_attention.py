import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow_addons.layers import MultiHeadAttention

# Input configuration
n_time = 10
n_dim1 = 32
n_dim2 = 64
n_dim3 = 48
m_dim = 32

# Input
input1 = Input(shape=(n_time, n_dim1))
input2 = Input(shape=(n_time, n_dim2))
input3 = Input(shape=(n_time, n_dim3))

# Multihead Attention for each pair
attention12 = MultiHeadAttention(head_size=n_dim1, num_heads=2)
attention13 = MultiHeadAttention(head_size=n_dim1, num_heads=2)
attention23 = MultiHeadAttention(head_size=n_dim2, num_heads=2)

# Attention for each pair
output12 = attention12([input1, input2])
output13 = attention13([input1, input3])
output23 = attention23([input2, input3])

# Concatenate results
concatenated = Concatenate(axis=-1)([output12, output13, output23])

# Feed Forward
dense = Dense(m_dim)
output = dense(concatenated)

# Build model
model = Model(inputs=[input1, input2, input3], outputs=output)
model.compile(loss="mean_squared_error", optimizer="adam")

print(model.summary())

# Generating random training data
import numpy as np

n_samples = 1000

x1_train = np.random.rand(n_samples, n_time, n_dim1)
x2_train = np.random.rand(n_samples, n_time, n_dim2)
x3_train = np.random.rand(n_samples, n_time, n_dim3)

y_train = np.random.rand(n_samples, n_time, m_dim)

# Training
model.fit([x1_train, x2_train, x3_train], y_train, epochs=10, batch_size=32)

# Testing
n_samples_test = 100

x1_test = np.random.rand(n_samples_test, n_time, n_dim1)
x2_test = np.random.rand(n_samples_test, n_time, n_dim2)
x3_test = np.random.rand(n_samples_test, n_time, n_dim3)

y_test = np.random.rand(n_samples_test, n_time, m_dim)

print("Testing set performance:")
model.evaluate([x1_test, x2_test, x3_test], y_test)

# Prediction
print("Predictions:")
predictions = model.predict([x1_test, x2_test, x3_test])
print(predictions)
