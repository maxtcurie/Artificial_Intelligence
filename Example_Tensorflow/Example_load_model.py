import tensorflow as tf

Model=tf.keras.models.load_model('./linear_regression.h5')

Model.summary()

#print(Model.__dict__)

import numpy as np 

test_set=[0.5,0.6,0.8]
test_set=np.array(test_set,dtype=float)

prediction=Model.predict(test_set).T

print('prediction='+str(prediction))
print('actual=    '+str(test_set*2.+2.))