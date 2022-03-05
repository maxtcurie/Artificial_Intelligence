import tensorflow as tf 

def LSTM():
	# Model Definition with LSTM
	model = tf.keras.Sequential([
	    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
	    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	    tf.keras.layers.Dense(6, activation='relu'),
	    tf.keras.layers.Dense(1, activation='sigmoid')
	])
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()

	return model