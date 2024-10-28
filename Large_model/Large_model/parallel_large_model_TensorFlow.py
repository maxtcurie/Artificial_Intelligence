import tensorflow as tf
from tensorflow.keras import datasets

# Define your model parts
def create_model_part1():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    return model

def create_model_part2():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(256,)))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

# Assign parts of a model to different GPUs
with tf.device('/GPU:0'):
    model_part1 = create_model_part1()
    
with tf.device('/GPU:1'):
    model_part2 = create_model_part2()

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 784)).astype('float32') / 255

# Define loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

for epoch in range(5):
    print(f"Start of epoch {epoch}")

    for step in range(100):  # Assume there are 100 steps in an epoch
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            with tf.device('/GPU:0'):
                part1_outputs = model_part1(train_images[step*32:(step+1)*32])

            with tf.device('/GPU:1'):
                logits = model_part2(part1_outputs)
                # Compute the loss value for this batch
                loss_value = loss_fn(train_labels[step*32:(step+1)*32], logits)
                
        print(f"Loss at step {step}: {loss_value}")

        # Get gradients of loss wrt the weights
        gradients_part1 = tape.gradient(loss_value, model_part1.trainable_weights)
        gradients_part2 = tape.gradient(loss_value, model_part2.trainable_weights)

        # Update the weights of our linear layer
        with tf.device('/GPU:0'):
            optimizer.apply_gradients(zip(gradients_part1, model_part1.trainable_weights))
        with tf.device('/GPU:1'):
            optimizer.apply_gradients(zip(gradients_part2, model_part2.trainable_weights))
