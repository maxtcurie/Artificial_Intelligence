import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define your model architecture
def build_model():
    # Define input layers
    input1 = keras.Input(shape=(32,), name='input1')
    input2 = keras.Input(shape=(64,), name='input2')

    # Concatenate the input layers
    concatenated = keras.layers.Concatenate()([input1, input2])

    # Hidden layers
    hidden1 = keras.layers.Dense(64, activation='relu')(concatenated)
    hidden2 = keras.layers.Dense(64, activation='relu')(hidden1)

    # Output layers
    output1 = keras.layers.Dense(1, activation='sigmoid', name='output1')(hidden2)
    output2 = keras.layers.Dense(1, activation='sigmoid', name='output2')(hidden2)

    # Create the model
    model = keras.Model(inputs=[input1, input2], outputs=[output1, output2])

    return model

# Build the model
model = build_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate dummy data for training
x1_train = np.random.random((1000, 32))
x2_train = np.random.random((1000, 64))
y1_train = np.random.randint(0, 2, size=(1000, 1))
y2_train = np.random.randint(0, 2, size=(1000, 1))

# Generate dummy data for validation
x1_val = np.random.random((200, 32))
x2_val = np.random.random((200, 64))
y1_val = np.random.randint(0, 2, size=(200, 1))
y2_val = np.random.randint(0, 2, size=(200, 1))

# Train the model with validation data
model.fit({'input1': x1_train, 'input2': x2_train},
          {'output1': y1_train, 'output2': y2_train},
          validation_data=({'input1': x1_val, 'input2': x2_val},
                           {'output1': y1_val, 'output2': y2_val}),
          epochs=10, batch_size=32)

# Generate dummy data for testing
x1_test = np.random.random((100, 32))
x2_test = np.random.random((100, 64))
y1_test = np.random.randint(0, 2, size=(100, 1))
y2_test = np.random.randint(0, 2, size=(100, 1))

# Make predictions on the test data
predictions = model.predict([x1_test, x2_test])

# Compare predictions with the actual labels
for i in range(len(predictions[0])):
    print("Actual Label (Output1):", y1_test[i], "| Predicted Probability (Output1):", predictions[0][i])
    print("Actual Label (Output2):", y2_test[i], "| Predicted Probability (Output2):", predictions[1][i])
