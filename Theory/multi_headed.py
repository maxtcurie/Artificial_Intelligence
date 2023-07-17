import numpy as np

def scaled_dot_product_attention(query, key, value):
    # Compute attention scores
    attention_scores = np.matmul(query, key.T)
    attention_scores /= np.sqrt(query.shape[-1])

    # Apply softmax to get attention weights
    attention_weights = np.softmax(attention_scores, axis=-1)

    # Compute weighted sum of values
    weighted_sum = np.matmul(attention_weights, value)

    return weighted_sum, attention_weights

def multihead_attention(inputs, num_heads):
    # Get input shape
    batch_size, seq_length, input_dim = inputs.shape

    # Split input into heads
    head_dim = input_dim // num_heads
    inputs = inputs.reshape(batch_size, seq_length, num_heads, head_dim)

    # Compute query, key, and value matrices for each head
    query = np.random.randn(num_heads, head_dim, head_dim)
    key = np.random.randn(num_heads, head_dim, head_dim)
    value = np.random.randn(num_heads, head_dim, head_dim)

    # Apply attention mechanism for each head
    outputs = []
    attention_weights = []
    for i in range(num_heads):
        output, attention = scaled_dot_product_attention(
            np.matmul(inputs[:, :, i, :], query[i]),
            np.matmul(inputs[:, :, i, :], key[i]),
            np.matmul(inputs[:, :, i, :], value[i])
        )
        outputs.append(output)
        attention_weights.append(attention)

    # Concatenate outputs of all heads
    outputs = np.concatenate(outputs, axis=-1)
    attention_weights = np.stack(attention_weights, axis=-1)

    return outputs, attention_weights

# Generate random input tensor
batch_size = 2
seq_length = 5
input_dim = 10
num_heads = 2
inputs = np.random.randn(batch_size, seq_length, input_dim)

# Apply multi-headed self-attention
outputs, attention_weights = multihead_attention(inputs, num_heads)

print("Input shape:", inputs.shape)
print("Output shape:", outputs.shape)
print("Attention weights shape:", attention_weights.shape)
