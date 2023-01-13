import tensorflow as tf 

#linear regression
def Dropout():
	model=tf.keras.models.Sequential([
		'''
		convolutional layer: 	64 filters, 
								size of the filter: 3X3

		'''
		tf.keras.layers.Conv2D(64,(3,3),activation='relu',\
								input_shape=(28,28,1)),
		'''
		https://youtu.be/8oOgPUO-TBY
		Max pooling:	take the max of the 2X2 box as the value 
						to reduce infomation size
		'''
		tf.keras.layers.MaxPooling2D(2,2),
		tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
		tf.keras.layers.MaxPooling2D(2,2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128,activation='relu'),
		
		#drop 20% of neuron
		tf.keras.layers.Dropout(0.2),

		tf.keras.layers.Dense(10, activation='softmax')
		]
		)
	return model