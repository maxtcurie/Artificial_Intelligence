model=keras.models.Sequential([
          keras.layers.SimpleRNN(20,return_sequences=True,
                            input_shape=[None,1]),
          keras.layers.SimpleRNN(20),
          keras.layers.Dense(1)
          ])