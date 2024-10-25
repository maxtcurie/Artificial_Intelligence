import numpy as np

# Helper function for softmax
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Multi-head self-attention mechanism
def multi_head_attention(Q, K, V, d_k, num_heads):
    batch_size, seq_length, d_model = Q.shape
    
    # Initialize weight matrices for each head
    W_q = np.random.randn(num_heads, d_model, d_k)
    W_k = np.random.randn(num_heads, d_model, d_k)
    W_v = np.random.randn(num_heads, d_model, d_k)
    
    # Linear projections
    Q_proj = np.dot(Q, W_q)  # (batch_size, seq_length, num_heads, d_k)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)
    
    # Split the queries, keys, and values for each head
    Q_proj = Q_proj.reshape(batch_size, seq_length, num_heads, d_k)
    K_proj = K_proj.reshape(batch_size, seq_length, num_heads, d_k)
    V_proj = V_proj.reshape(batch_size, seq_length, num_heads, d_k)
    
    # Scaled dot-product attention
    scores = np.einsum('bhqd, bhkd -> bhqk', Q_proj, K_proj) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    
    # Apply attention weights to values
    attention_output = np.einsum('bhqk, bhvd -> bhqd', attention_weights, V_proj)
    
    # Concatenate heads
    attention_output = attention_output.reshape(batch_size, seq_length, num_heads * d_k)
    
    # Final linear transformation
    W_o = np.random.randn(num_heads * d_k, d_model)
    output = np.dot(attention_output, W_o)
    
    return output

# Position encoding
def positional_encoding(seq_length, d_model):
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    pos_encoding = pos * angle_rates
    
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    
    return pos_encoding

# Feed-forward layer
def feed_forward(x, d_ff, d_model):
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros((1, d_ff))
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros((1, d_model))
    
    return np.dot(np.maximum(0, np.dot(x, W1) + b1), W2) + b2

# Layer normalization
def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

# Transformer Encoder block
class TransformerBlock:
    def __init__(self, d_model, d_ff, num_heads):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
    
    def __call__(self, x):
        # Multi-head self-attention
        attn_output = multi_head_attention(x, x, x, self.d_k, self.num_heads)
        attn_output = layer_norm(attn_output + x)
        
        # Feed-forward network
        ff_output = feed_forward(attn_output, self.d_ff, self.d_model)
        ff_output = layer_norm(ff_output + attn_output)
        
        return ff_output

# Transformer encoder
class TransformerEncoder:
    def __init__(self, num_layers, d_model, d_ff, num_heads, seq_length):
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.layers = [TransformerBlock(d_model, d_ff, num_heads) for _ in range(num_layers)]
        self.pos_encoding = positional_encoding(seq_length, d_model)
    
    def encode(self, x):
        # Add position encoding
        x += self.pos_encoding
        
        # Pass through all the layers
        for layer in self.layers:
            x = layer(x)
        
        return x

# Example usage
if __name__ == "__main__":
    seq_length = 10
    d_model = 512
    d_ff = 2048
    num_heads = 8
    num_layers = 6
    batch_size = 32

    # Random input sequence
    x = np.random.randn(batch_size, seq_length, d_model)
    
    # Create the transformer encoder
    encoder = TransformerEncoder(num_layers, d_model, d_ff, num_heads, seq_length)
    
    # Encode the input sequence
    encoded_output = encoder.encode(x)
    
    print("Encoded output shape:", encoded_output.shape)
