import numpy as np

def self_attention(inputs):
    # Get the input shape
    batch_size, seq_length, input_dim = inputs.shape

    # Compute the query, key, and value matrices
    query = np.random.randn(input_dim, input_dim)
    key = np.random.randn(input_dim, input_dim)
    value = np.random.randn(input_dim, input_dim)

    # Compute the attention scores
    attention_scores = np.matmul(np.matmul(inputs, query), np.transpose(key, axes=(0, 2, 1)))
    attention_scores /= np.sqrt(input_dim)

    # Apply softmax activation to get attention weights
    attention_weights = np.softmax(attention_scores, axis=-1)

    # Compute the weighted sum of values
    weighted_sum = np.matmul(attention_weights, value)

    return weighted_sum

# Generate random input tensor
batch_size = 2
seq_length = 5
input_dim = 10
inputs = np.random.randn(batch_size, seq_length, input_dim)

# Apply self-attention
output = self_attention(inputs)

print("Input shape:", inputs.shape)
print("Output shape:", output.shape)
