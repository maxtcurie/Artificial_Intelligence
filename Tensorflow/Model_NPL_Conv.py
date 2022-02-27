import tensorflow as tf 

#https://youtu.be/xtPXjvwCt64
def NPL_embedding_Conv1D():
	vocab_size = 10000
	embedding_dim = 16
	max_length = 120

	# Model Definition with Conv1D
	model = tf.keras.Sequential([
	    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
	    tf.keras.layers.Conv1D(128, 5, activation='relu'),
	    tf.keras.layers.GlobalAveragePooling1D(),
	    tf.keras.layers.Dense(6, activation='relu'),
	    tf.keras.layers.Dense(1, activation='sigmoid')
	])
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
