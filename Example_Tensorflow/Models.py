
class model_gen():
	def __init__(self):
		return 0 

	def model_LN(self):
		from Model_LN import LN
		return LN()

	def model_CNN_2D(self):
		from Model_CNN_1D import CNN_2D
		return CNN_2D()

	def model_CNN_1D(self):
		from Model_CNN_1D import CNN_1D
		return CNN_1D()

	def model_RNN(self):
		from Model_RNN import RNN
		return RNN()

	



from Models import model_gen

model_gen=model_gen()
model=model_gen.model_CNN_2D()
model.summary()