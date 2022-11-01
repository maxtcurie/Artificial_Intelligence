model=keras.models.Sequential([
          keras.layers.Lambda(
                      lambda x: tf.expand_dims(x,axis=-1),
                      input_shape=[None]),
          keras.layers.SimpleRNN(20,return_sequences=True,
                            input_shape=[None,1]),
          keras.layers.SimpleRNN(20),
          keras.layers.Dense(1)
          #scale the output by 100
          keras.layers.Lambda(lambda x: x*100.)
          ])