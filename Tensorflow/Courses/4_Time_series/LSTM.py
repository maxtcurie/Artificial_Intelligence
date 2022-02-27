model=keras.models.Sequential([
          keras.layers.Lambda(
                      lambda x: tf.expand_dims(x,axis=-1),
                      input_shape=[None]),
          keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
          keras.layers.Dense(1)
          ])