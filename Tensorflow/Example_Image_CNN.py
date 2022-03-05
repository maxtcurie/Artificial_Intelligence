import os
import zipfile 
import wget #pip install wget
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt

# Note: This is a very large dataset and will take some time to download

#pre-processing


url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path='./tmp'
file_name=path+'/cats_and_dogs_filtered'
local_zip = file_name+'.zip'

TRAINING_DIR = "./tmp/cats_and_dogs_filtered/train/"
TESTING_DIR =  "./tmp/cats_and_dogs_filtered/validation/"

#check(create) the path
try: 
    os.mkdir(path)
except:
    pass

#download file
if not os.path.exists(file_name+'.zip'):
    print('Beginning download cat and dog images for training')
    wget.download(url, file_name+'.zip')


# Unzip the archive
if not os.path.exists(file_name):
  print('unziping')
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall(path)
  zip_ref.close()
  print('finished unzip')

# Test your split_data function

# Define paths


# GRADED FUNCTION: train_val_generators
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  train_datagen = ImageDataGenerator( rescale=1./255.,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

  # Pass in the appropiate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=20,
                                                      class_mode='binary',
                                                      target_size= (150, 150))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator( rescale=1./255.)

  # Pass in the appropiate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size= (150, 150))
  ### END CODE HERE
  return train_generator, validation_generator

# Test your generators
train_generator, validation_generator = train_val_generators(TRAINING_DIR, TESTING_DIR)

class myCallback(tf.keras.callbacks.Callback):
 def on_epoch_end(self,epoch,log={}):
   if(log.get('val_accuracy')>0.99):
     print('\naccuracy is good enough so cancelling training!')
     self.model.stop_training=True

# GRADED FUNCTION: create_model
def create_model():
  # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
  # USE AT LEAST 3 CONVOLUTION LAYERS

  ### START CODE HERE
  
  model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),  
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2), 
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    #tf.keras.layers.Dense(4096, activation='relu'), 
    tf.keras.layers.Dense(512, activation='relu'), 
    #tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
  ])

  model.compile(
              optimizer='Adam',
              #optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy']
              )
    
  ### END CODE HERE

  return model

print(dir(validation_generator))
print(validation_generator.labels)
input()
# Get the untrained model
model = create_model()
callbacks = myCallback()
# Train the model
# Note that this may take some time.
history = model.fit(train_generator,
                    epochs=100,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=validation_generator)

                    #,callbacks=[callbacks])

from Post_plot_learning_rate import plot_hist
plot_hist(history)

#save the trained model
model.save('./tmp/cat_dog_CNN.h5')
print('Model Saved!')
