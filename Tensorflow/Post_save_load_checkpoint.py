#https://www.tensorflow.org/tutorials/keras/save_and_load

#create checkpoint
import os 
if not os.path.exists('./tmp/checkpoint'):
    os.mkdir('./tmp/checkpoint')
checkpoint_path='./tmp/checkpoint/checkpoint'
batch_size=10


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

model.load_weights(checkpoint_path)