import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow_addons.layers import MultiHeadAttention
from itertools import combinations

def create_cross_attention_model(n_time=10, n_dims=[32, 64, 48], m_dim=32, num_heads=2):
    # Generate inputs based on the dimensions specified in n_dims
    inputs = [Input(shape=(n_time, dim)) for dim in n_dims]

    # Initialize the attention outputs list
    attention_outputs = []

    # Generate all unique pairs of inputs and apply the attention layer to each pair
    for i, j in combinations(range(len(n_dims)), 2):
        attention = MultiHeadAttention(head_size=n_dims[i], num_heads=num_heads)
        output = attention([inputs[i], inputs[j]])
        attention_outputs.append(output)

    # Concatenate all attention outputs
    concatenated = Concatenate(axis=-1)(attention_outputs)

    # Feed Forward
    dense = Dense(m_dim)
    output = dense(concatenated)

    # Build model
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model
