#import libraries
import tensorflow as tf 

train_portion=0.8

#pre process
import numpy as np 

import os 
if not os.path.exists('./tmp/checkpoint'):
    os.mkdir('./tmp/checkpoint')
checkpoint_path='./tmp/checkpoint/checkpoint'
batch_size=10

xs=np.arange(100,dtype=float)*0.01
ys=xs*2.+2.

index=int(train_portion*len(xs))

x_train=xs[:index]
y_train=ys[:index]
x_val=xs[index:]
y_val=ys[index:]


#create model
model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1,input_shape=[1])
                ])

#*create callback function (optional)
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,log={}):

        #print(log.get.keys())
        #print(log.get('epoch'))
        if(log.get('val_loss')<0.000001):
            print('loss<0.000001, stop training!')
            self.model.stop_training=True

callbacks=myCallback()

#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')


#compile model
model.compile(optimizer='sgd',\
            loss='mean_squared_error')


#run the model
history = model.fit(x_train,y_train,epochs=10,\
                 callbacks=[cp_callback,callbacks],\
                 validation_data=(x_val, y_val)
                 )

a=input('Countinue? (1)yes (0)no\n')
if int(a)==1:
    print('load checkpoint')
    model.load_weights(checkpoint_path)
    history = model.fit(x_train,y_train,epochs=90,\
                 callbacks=[cp_callback,callbacks],\
                 validation_data=(x_val, y_val)
                 )

#check the prediction
test_set=[0.5]
test_set=np.array(test_set,dtype=float)
[prediction]=model.predict(test_set)
print('prediction='+str(prediction))
print('actual=    '+str(test_set*2.+2.))

#post processing
import matplotlib.pyplot as plt

plt.clf()
plt.plot(history.history["loss"])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.grid()
plt.show()
