import tensorflow as tf 
from scipy import misc
import matplotlib.pyplot as plt 
import numpy as np 

#load the ascent image
ascent_image=misc.ascent()

#plot the image
#plt.grid(False)
#plt.gray()
#plt.axis('off')
#plt.imshow(ascent_image)
#plt.show()



(size_x,size_y)=np.shape(ascent_image)
print('[size_x,size_y]='+str([size_x,size_y]))



#Apply convolution filter
filter_=[
		[0,1,0],\
		[1,-4,1],\
		[0,1,0]
		]

weight=1
image_filtered=np.zeros((size_x,size_y),dtype=float)
for x in range(1,size_x-1):
	for y in range(1,size_y-1):
		image_block_3_by_3=ascent_image[x-1:x+2,y-1:y+2]
		#multiply each element
		convolution=weight*np.sum( np.multiply(image_block_3_by_3,filter_) )
		if (convolution<0):
			convolution=0
		elif(convolution>255):
			convolution=255
		image_filtered[x,y]=convolution



#Apply n by n pool
pool_size=2
image_max_pooled=np.zeros((int(size_x/pool_size),int(size_y/pool_size)),dtype=float)
image_avg_pooled=np.zeros((int(size_x/pool_size),int(size_y/pool_size)),dtype=float)
for x in range(int(size_x/pool_size)):
	for y in range(int(size_y/pool_size)):
		image_block_n_by_n=image_filtered[pool_size*x:pool_size*x+pool_size,\
											 pool_size*y:pool_size*y+pool_size]
		#max pooling
		image_max_pooled[x,y]=np.max(image_block_n_by_n)
		#average_pooling
		image_avg_pooled[x,y]=np.mean(image_block_n_by_n)

fig, ax=plt.subplots(nrows=1,ncols=4) 

ax[0].imshow(ascent_image,cmap='gray')
ax[0].set_title('original')

ax[1].imshow(image_filtered,cmap='gray')
ax[1].set_title('filtered')

ax[2].imshow(image_max_pooled,cmap='gray')
ax[2].set_title('max pooled')

ax[3].imshow(image_avg_pooled,cmap='gray')
ax[3].set_title('average pooled')

plt.show()