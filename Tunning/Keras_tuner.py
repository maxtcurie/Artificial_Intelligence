#from: https://youtu.be/6Nf1x7qThR8 

import tensorflow as tf
import keras_tuner as kt
import pandas as pd


hp_epochs=10
epochs=50
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


X_train = X_train / 255.0
X_test = X_test / 255.0


def model_builder(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

  #hp_activation = hp.Choice('activation', values=['relu'])
  hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=1000, step=100)
  #hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=1000, step=100)
  #hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(tf.keras.layers.Dense(units=hp_layer_1, activation='relu'))
  model.add(tf.keras.layers.Dense(units=50, activation='softmax'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), #hp_learning_rate
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
  
  return model



tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=hp_epochs,
                     factor=3,
                     directory='dir',
                     project_name='x')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#https://keras.io/api/callbacks/
log_callback=tf.keras.callbacks.TensorBoard(log_dir='./logs'),

tuner.search(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[stop_early,log_callback])


import sys

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

tuner.results_summary(num_trials=10**10)

sys.stdout = orig_stdout
f.close()
 
  

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)

history = model.fit(X_train, y_train, epochs=50, validation_split=0.2,
                    callbacks=[stop_early])

df=pd.DataFrame(history.history)

