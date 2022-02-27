import tensorflow as tf 

#linear regression
def LR():
	model=tf.keras.Sequential([
				tf.keras.layers.Dense(units=1,input_shape=[1])
				])

	return model