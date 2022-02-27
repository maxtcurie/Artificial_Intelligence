model=keras.models.Sequential([
          keras.layers.Conv1D(filters=32,kernel_size=5,
                              strides=1,padding='causal',
                              activation='relu',
                              input_shape=[None,1])
          keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
          keras.layers.Dense(1)
          ])