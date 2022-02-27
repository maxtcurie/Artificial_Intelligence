#https://www.tensorflow.org/tutorials/keras/save_and_load




#save the trained model
import os 
path='tmp'
try: 
    os.mkdir(path)
except:
    pass
model.save('./tmp/linear_regression.h5')
print('Model Saved!')

import tensorflow as tf
#load model
savedModel=tf.keras.models.load_model('gfgModel.h5')
