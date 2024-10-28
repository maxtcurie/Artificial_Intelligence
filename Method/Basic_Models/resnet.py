import tensorflow as tf
from tensorflow.keras import layers, models

def resnet_block(input_layer, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_layer])
    x = layers.Activation('relu')(x)
    return x

# Create the model
inputs = tf.keras.Input(shape=(224, 224, 3))

# Initial Conv Layer
x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

# ResNet blocks
x = resnet_block(x, 64, 3)
x = resnet_block(x, 64, 3)

x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
x = resnet_block(x, 128, 3)
x = resnet_block(x, 128, 3)

x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
x = resnet_block(x, 256, 3)
x = resnet_block(x, 256, 3)

x = layers.Conv2D(512, 3, strides=2, padding='same')(x)
x = resnet_block(x, 512, 3)
x = resnet_block(x, 512, 3)

# Global Average Pooling and Dense Layer
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1000, activation='softmax')(x)

# Create and compile the model
model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
