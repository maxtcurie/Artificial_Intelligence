lr_scheduel=rf.keras.callbacks.LearningRateScheduler(
                    lambda: epoch: 1e-8 * 10**(epoch/20)
                    )


optimizer=tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=['mae'])

history=model.fit(train_set,epochs=100,callbacks=[lr_scheduel])
