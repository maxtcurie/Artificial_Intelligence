import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

omega = 1.

# Define the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # input layer for time 't'
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1)  # output layer for position 'u'
])

# Define the loss function
def custom_loss(y_true, y_pred):
    t = y_true[:, 0]  # Extract t values
    u_obs = y_true[:, 1]  # Extract observed u values
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape1.watch(t)
            tape2.watch(t)
            u = model(t)
        du_dt = tape2.gradient(u, t)
    d2u_dt2 = tape1.gradient(du_dt, t)

    L_data = tf.reduce_mean(tf.square(u - u_obs))
    L_physics = tf.reduce_mean(tf.square(d2u_dt2 + omega**2 * u))
    return L_data + L_physics

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss=custom_loss)

# Train the model
t_data = np.arange(0, 3, 0.01)  # your array of time points where you have observations
u_obs = np.sin(omega * t_data)  # your array of observed positions at those time points

# This is a small modification: concatenate t_data and u_obs
# This will ensure that the custom loss function receives both t and u_obs as y_true
train_data = np.vstack([t_data, u_obs]).T
model.fit(t_data, train_data, epochs=10000)

# Predict (curve fitting)
t_new = np.arange(0, 1.5, 0.01)  # new time points
u_pred = model.predict(t_new)

plt.clf()
plt.plot(t_new, u_pred)
plt.plot(t_new, np.sin(t_new))
plt.show()
