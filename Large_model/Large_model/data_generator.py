import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert the labels to one-hot encoded vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the data generator for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Create the data generator iterator for training data
train_generator = train_datagen.flow(
    x_train,
    y_train,
    batch_size=32
)

# Define the data generator for testing (validation)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Create the data generator iterator for testing data
test_generator = test_datagen.flow(
    x_test,
    y_test,
    batch_size=32
)

# Define your model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the data generator
model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 32,  # Number of batches per epoch
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(x_test) // 32  # Number of batches for validation
)
