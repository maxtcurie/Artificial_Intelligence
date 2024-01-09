#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.backend import random_normal, mean, log, exp

# Generate synthetic curve data
x = np.linspace(0, 5, 100)
y = np.array([i*np.sin(x) for i in np.random.rand(1000)])

print(np.shape(y))

# VAE Parameters
input_dim = 100
latent_dim = 4
intermediate_dim = 64
batch_size = 1024
epochs = 1000

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = random_normal(shape=(batch, dim))
    return z_mean + exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(input_dim)
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(inputs, x_decoded_mean)

# Loss: Reconstruction loss + KL divergence
xent_loss = input_dim * mean_squared_error(inputs, x_decoded_mean)
kl_loss = - 0.5 * mean(1 + z_log_var - tf.square(z_mean) - exp(z_log_var), axis=-1)
vae_loss = mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train VAE
vae.fit(y, y, epochs=epochs, batch_size=batch_size)

# Predict (fit curve)
y_pred = vae.predict(y)

# Now y_pred can be plotted against x to see how well the curve has been fitted.


# In[14]:


i_plot=500
# Plot the original and predicted curves
plt.figure(figsize=(10, 6))
plt.plot(x, y[i_plot,:], label='Original Curve')
plt.plot(x, y_pred[i_plot,:], label='VAE Fitted Curve', linestyle='dashed')
plt.legend()
plt.title("Original vs VAE Fitted Curve")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

