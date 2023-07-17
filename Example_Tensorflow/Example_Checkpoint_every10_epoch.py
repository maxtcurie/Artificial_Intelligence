import os 
if not os.path.exists('./tmp/checkpoint'):
    os.mkdir('./tmp/checkpoint')
checkpoint_path='./tmp/checkpoint/checkpoint'


import tensorflow as tf

# Define a mock model
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Define a mock optimizer
optimizer = tf.keras.optimizers.Adam()

# Create a custom callback to save the weights every 10 epochs
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model, optimizer):
        super(CustomModelCheckpoint, self).__init__()
        self.model = model
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            self.model.save_weights(checkpoint_path.format(epoch=epoch+1))
            print(f'Saved model weights for epoch {epoch+1}')
            
            # Release references to the model and optimizer
            self.model = None
            self.optimizer = None

# Create the custom callback instance
cp_callback = CustomModelCheckpoint(model, optimizer)

# Train the model with the custom callback
model.compile(optimizer=optimizer, loss='binary_crossentropy')
model.fit(x_train, y_train, epochs=epochs, callbacks=[cp_callback])
