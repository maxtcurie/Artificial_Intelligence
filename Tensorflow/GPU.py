import tensorflow as tf
#conda create -n tf_gpu python==3.8
#conda activate tf_gpu

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))