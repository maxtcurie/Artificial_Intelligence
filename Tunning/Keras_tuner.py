#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

all_X_norm=np.random.rand((1000,1,52))
all_y_norm=np.random.rand((1000,108))

X_shape=all_X_norm.shape[1:]
y_shape=all_y_norm.shape[1:]

# Split data into train and validation sets
X_test = all_X_norm[::10,...]  # Every 10th step
X_train_val = np.delete(all_X_norm, np.arange(0, all_X_norm.shape[0], 10), axis=0)  # All other steps

X_val = X_train_val[::10,...]  # Every 10th step
X_train = np.delete(X_train_val, np.arange(0, X_train_val.shape[0], 10), axis=0)  # All other steps


y_test = all_y_norm[::10,...]  # Every 10th step
y_train_val = np.delete(all_y_norm, np.arange(0, all_y_norm.shape[0], 10), axis=0)  # All other steps

y_val = y_train_val[::10,...]  # Every 10th step
y_train = np.delete(y_train_val, np.arange(0, y_train_val.shape[0], 10), axis=0)  # All other steps


def model_builder(hp):
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    hp_layers_size = hp.Int('layers_size', min_value=128, max_value=512, step=128)
    hp_num_layers = hp.Int('num_layers', min_value=1, max_value=5, step=1)
    hp_activation = hp.Choice('activation', values=['leaky_relu','relu', 'tanh', 'sigmoid', 'linear'])
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1, 52)))
    
    for _ in range(hp_num_layers):
        model.add(tf.keras.layers.Dense(units=hp_layers_size, activation=hp_activation))
        model.add(tf.keras.layers.Dropout(rate=hp_dropout_rate))
    
    model.add(tf.keras.layers.Dense(108))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), 
                  loss='mse', 
                  metrics=['mse'])
    
    return model
    
# Initialize the tuner
tuner = kt.tuners.BayesianOptimization(
    model_builder,
    objective='val_mse',
    max_trials=30,
    executions_per_trial=2,
    directory='tuner_results',
    project_name='bayesian_tuning_2'
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=6, restore_best_weights=True)

# Search for best hyperparameters
tuner.search(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping], batch_size=2048, epochs=20)

# Extract trial information and save as CSV
trials = tuner.oracle.trials

results_list = []

for trial_id, trial in trials.items():
    trial_info = trial.hyperparameters.values
    trial_info['trial_id'] = trial.trial_id
    # Accessing the metrics for the trial and fetching the best value for 'val_mse'
    trial_info['val_mse'] = trial.metrics.metrics['val_mse'].get_best_value()
    results_list.append(trial_info)

df = pd.DataFrame(results_list)
df.to_csv('tuner_results.csv', index=False)


print(df)

