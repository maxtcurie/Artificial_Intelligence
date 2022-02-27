#import libraries
import tensorflow as tf 

train_portion=0.8

#pre process
import numpy as np 

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
        if(log.get('val_loss')<0.004):
            print('\nLoss is low so cancelling training!')
            self.model.stop_training=True

callbacks=myCallback()

#compile model
model.compile(optimizer='sgd',\
            loss='mean_squared_error')


#run the model
history = model.fit(x_train,y_train,epochs=500,\
                 callbacks=callbacks,\
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
