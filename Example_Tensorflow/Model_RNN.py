import tensorflow as tf 

#linear regression
def RNN():
	model = tf.keras.models.Sequential([
					tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
								input_shape=[None]),
					tf.keras.layers.SimpleRNN(40, return_sequences=True),
					tf.keras.layers.SimpleRNN(40),
					tf.keras.layers.Dense(1),
					tf.keras.layers.Lambda(lambda x: x * 100.0)
				])

	return model