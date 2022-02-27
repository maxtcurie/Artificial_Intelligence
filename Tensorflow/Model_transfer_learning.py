import tensorflow as tf 
import os 

def Transfer_learning_model():
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
  x=tf.keras.layers.Dense(1,activation='sigmoid')(x)
  
  model=tf.keras.Model(pre_trained_model.input,x)
  
  model.compile(optimizer='Adam',
                #optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
  
  return model



