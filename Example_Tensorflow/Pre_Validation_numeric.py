import numpy as np 

train_portion=0.8

xs=np.arange(100,dtype=float)*0.01
ys=xs*2.+2.

index=int(train_portion*len(xs))

x_train=xs[:index]
y_train=ys[:index]
x_val=xs[index:]
y_val=ys[index:]

validation_data=(x_val, y_val)