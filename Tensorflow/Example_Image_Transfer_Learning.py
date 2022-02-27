import os
import zipfile
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
IMG_SIZE = (160, 160)
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
  train_datagen = ImageDataGenerator( rescale=1./255. ,
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
                                                      target_size= IMG_SIZE)

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator( rescale=1./255.)

  # Pass in the appropiate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size= IMG_SIZE)
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
  if not os.path.exists('tmp'):
    os.mkdir('tmp')

  if os.path.exists('./tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'):
    pass
  else:
    #download model
    import wget #pip install wget
    print('Beginning download pre_trained_model')
    
    url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    wget.download(url, './tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
  

  #load pre trained model 
  from tensorflow.keras.applications.inception_v3 import InceptionV3

  local_weight_file='./tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
  pre_trained_model=InceptionV3(
                                input_shape=IMG_SIZE+(3,),
                                #the model has dense layer before convolutional layer,
                                # to set it to false, one will ge to convolutional layer as the first(top) layer 
                                include_top=False,
                                #set weight to None to not use the built-in weight for the NN
                                weights='imagenet'
                                )
  
  #lock the model so that it cannot be trained
  for layer in pre_trained_model.layers:
    layer.trainable=False
  
  pre_trained_model.summary()
    ### START CODE HERE
  #creat apropriate last layer to connect
  #mix7: output of convolution 7X7
  last_layer=pre_trained_model.get_layer('mixed7')
  last_output=last_layer.output
  
  #conect with the pretrained model
  x=tf.keras.layers.Flatten()(last_output)
  x=tf.keras.layers.Dense(1024,activation='relu')(x)
  #x=tf.keras.layers.Dropout(0.2)(x)
  x=tf.keras.layers.Dense(1,activation='sigmoid')(x)
  
  model=tf.keras.Model(pre_trained_model.input,x)
  
  model.compile(optimizer='Adam',
                #optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
  
  return model


# Get the untrained model
model = create_model()

callbacks = myCallback()
# Train the model
# Note that this may take some time.
history = model.fit(train_generator,
                    epochs=200,
                    verbose=1,
                    callbacks=[callbacks],
                    validation_data=validation_generator)

from Post_plot_learning_rate import plot_hist
plot_hist(history)

model.save('./tmp/cat_dog_transfer_learning.h5')