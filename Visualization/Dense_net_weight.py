#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_dense_NN(lines, dots):
    plt.clf()
    for line in lines:
        plt.plot(line[0], line[1], color='blue', alpha=line[2])
    for dot in dots:
        plt.scatter(dot[0], dot[1], color='red', s=dot[2] * 500)
    plt.show()

def weights_to_lines_and_dots(weights):
    lines = []
    dots = []

    for i in range(int(len(weights) / 2)):
        weight = weights[i * 2]
        weight_shape = np.shape(weight)

        # Skip if the weight is not a 2D array (e.g., bias vector)
        if len(weight_shape) != 2:
            continue

        (nx1, nx2) = weight_shape

        dot_weights = np.sum(np.abs(weight), axis=1)
        dot_weights /= np.sum(dot_weights)

        dot_num = len(dot_weights)

        for j in range(dot_num):
            dots.append([i, dot_num / 2 - j, dot_weights[j]])

        weight = np.abs(weight) / np.sum(np.abs(weight))

        for j in range(nx1):
            for k in range(nx2):
                y1 = nx1 / 2 - j
                y2 = nx2 / 2 - k
                line_weight = weight[j, k]
                lines.append([[i, i + 1], [y1, y2], line_weight])

    nx2 = weight_shape[1] if len(weight_shape) == 2 else 1
    for i in range(nx2):
        dots.append([len(weights) / 2, nx2 / 2 - i, 1 / nx2])

    return lines, dots


def first_weight(weights):
    weight = weights[0]
    dot_weights = np.sum(np.abs(weight), axis=1)
    dot_weights /= np.sum(dot_weights)
    return dot_weights





# In[ ]:


# Load model
model = tf.keras.models.load_model('/scratch/gpfs/mc5076/models/hi2Tetmp.h5')

# Extract weights
weights = model.get_weights()

# Convert weights to visualization format
lines, dots = weights_to_lines_and_dots(weights)

# Plot the neural network visualization
plot_dense_NN(lines, dots)

