import numpy as np 
from goolge.colab import files
from keras.preprocessing import image 

#upload the files
uploaded = files.upload() 

for fn in uploaded.keys():
	path='/content/'+fn
	img=image.load_img(path,target_size=(300,300))

	x=image.img_to_array(img)

	#https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
	#The following is equivalent to x[axis, :] 
	x=np.expand_dims(x,axis=0)

	#stacking multiple arrays, 
	#in this case np.shape from(300,300) to (300,300,1)
	#https://www.w3resource.com/numpy/manipulation/vstack.php#:~:text=The%20vstack()%20function%20is,with%20up%20to%203%20dimensions.
	images=np.vstavk([x])

	classes=model.predict(images,batch_size=10)
	print(classes[0])
	if classes[0]>0.5:
		print(fn+' is a human')
	else:
		print(fn+' is a horse')