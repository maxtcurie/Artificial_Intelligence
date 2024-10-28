import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

# 1. Create a dataset
data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

# 2. Normalize the dataset
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(data)
normalized_data = normalizer(data)

print("Original Data:")
print(data)
print("\nNormalized Data:")
print(normalized_data)

# 3. Save and load the normalizer
normalizer.save('normalizer')
loaded_normalizer = tf.keras.models.load_model('normalizer')

# 4. Use the loaded normalizer on a new dataset
new_data = np.array([
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0]
])

normalized_new_data = loaded_normalizer(new_data)
print("\nNew Data:")
print(new_data)
print("\nNormalized New Data:")
print(normalized_new_data)
