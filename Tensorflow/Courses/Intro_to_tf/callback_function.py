import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self,epoch,log={}):
		if(log.get('loss')<0.4):
			print('\nLoss is low so cancelling training!')
			self.model.stop_training=True