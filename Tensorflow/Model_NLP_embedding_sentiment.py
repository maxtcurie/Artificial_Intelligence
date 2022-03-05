import tensorflow as tf 

#https://youtu.be/xtPXjvwCt64
def NPL_embedding():
	vocab_size = 10000
	embedding_dim = 16
	max_length = 120

	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(6, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')
		])
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
