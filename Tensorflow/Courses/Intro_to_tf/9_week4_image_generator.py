from tensorflow.keras.preprocessing.image import ImageDataGenerator 


train_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(\
				#directory of the trainning images
				train_dir,
				#uniform the image size
				target_size=(300,300),
				#load the image by batch size
				batch_size=128,
				#class mode 1 or 0
				class_mode='binary')
