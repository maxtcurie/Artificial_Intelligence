#import libraries
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt


train_portion=0.8
define_loss=10**(-5)
n=100000

x1_s=np.arange(n,dtype=float)/n
x2_s=np.random.rand(n)
x3_s=np.random.rand(n)
ys=x1_s*2.+2.

xs=np.zeros((n,3))
xs[:,0]=x1_s
xs[:,1]=x2_s
xs[:,2]=x3_s

plt.clf()
plt.plot(x1_s,ys,label='x1')
plt.scatter(x2_s,ys,color='orange',s=5,label='x2',alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

index=int(train_portion*len(xs))

print(np.shape(xs))

x_train=xs[:index,:]
y_train=ys[:index]
x_val=xs[index:,:]
y_val=ys[index:]



#create model
model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1,\
                    activation='linear',\
                    input_shape=[3])
                #tf.keras.layers.Dense(units=1,\
                #    activation='linear')
                ])

#*create callback function (optional)
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,log={}):
        if(log.get('val_loss')<define_loss):
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
test_set=[[0.5,0.2,0.2]]
test_set=np.array(test_set,dtype=float)
[prediction]=model.predict(test_set)
print('prediction='+str(prediction))
print('actual=    '+str(test_set[0]*2.+2.))

#post processing
import matplotlib.pyplot as plt

plt.clf()
plt.plot(history.history["loss"])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.grid()
plt.show()

model.save('./Save/linear_regression.h5')
print('Model Saved!')


