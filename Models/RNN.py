import numpy as np 

#W_out=inverse(R*R.T+epsilon*I)*(D*R.T)

def init_weight(input_dim,res_size,K_in,K_rec,insca,spra,bisca):
	#K_in: input
	#K_rec: reccuent
	#bias: 

	if K_in==-1: #fully connected input-> 
		W_in=insca*(np.random.rand(res_size,input_dim)*2-1) #random range from [-insca,insca]
	else: