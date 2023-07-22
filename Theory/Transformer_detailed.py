import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers

class TransformerModel(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, dff, rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(input_vocab_size, d_model, num_heads, num_encoder_layers, dff, rate)
        self.decoder = Decoder(target_vocab_size, d_model, num_heads, num_decoder_layers, dff, rate)
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training):
        enc_output = self.encoder(inputs, training)
        dec_output = self.decoder(targets, enc_output, training)
        final_output = self.final_layer(dec_output)
        return final_output

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, d_model, num_heads, num_layers, dff, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(input_vocab_size, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training)
        return x

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output, _ = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

class Decoder(tf.keras.layers.Layer):
    def __init__(self, target_vocab_size, d_model, num_heads, num_layers, dff, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(target_vocab_size, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, inputs, enc_output, training):
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, training)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        return x, attention_weights

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = layers.MultiHeadAttention(num_heads, d_model)
        self.mha2 = layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs, enc_output, training):
        attn1, block1 = self.mha1(inputs, inputs, inputs)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        attn2, block2 = self.mha2(enc_output, enc_output, out1)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3, block1, block2

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

# Example usage
input_vocab_size = 10000
target_vocab_size = 20000
d_model = 256
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4
dff = 512

input_sequence_length = 100
target_sequence_length = 150
batch_size = 32

inputs = tf.random.uniform((batch_size, input_sequence_length), dtype=tf.int32, minval=0, maxval=input_vocab_size)
targets = tf.random.uniform((batch_size, target_sequence_length), dtype=tf.int32, minval=0, maxval=target_vocab_size)

model = TransformerModel(input_vocab_size, target_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, dff)
output = model(inputs, targets, training=True)

print(output.shape)
