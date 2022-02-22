import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

model = tf.keras.models.Sequential([
						tf.keras.layers.Conv2D(16,(3,3),activation='relu',
												input_shape=(300,300,3)), 
						tf.keras.layers.MaxPooling2D(2,2),
						tf.keras.layers.Conv2D(32,(3,3),activation='relu'), 
						tf.keras.layers.MaxPooling2D(2,2),
						tf.keras.layers.Conv2D(64,(3,3),activation='relu'), 
						tf.keras.layers.MaxPooling2D(2,2),
						tf.keras.layers.Flatten(),
						tf.keras.layers.Dense(512,activation='relu'),
						tf.keras.layers.Dense(1,activation='sigmoid'),
						]
						)
model.compile(loss='binary_crossentropy',
				optimizer=RMSprop(lr=0.001),
				metrics=['accuracy']
				)
#total 1024 images
model.fit(
			train_generator,
			#load images in 8 batches, 256/batch
			steps_per_epoch=8,
			epochs=15,
			validation_data=validation_generator,
			#handeling images in 8 batches, 32/batch
			validation_steps=8,
			#the less for progress bar
			verbose=2)