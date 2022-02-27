#https://www.tensorflow.org/tutorials/keras/save_and_load




#save the trained model
model.save('gfgModel.h5')
print('Model Saved!')
 
#load model
savedModel=load_model('gfgModel.h5')
