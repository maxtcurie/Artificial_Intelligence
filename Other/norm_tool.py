import numpy as np 
import pandas as pd

def normalize(data_set_list,log_list=0,csv_name='./norm.csv'):
	if log_list==0:
		log_list=[0]*len(data_set_list)
	offset=[]
	scale=[]
	data_set_norm_list=[]
	for i in range(len(data_set_list)):
		data_set=data_set_list[i]
		#doing log
		if log_list[i]==1:
			#print(data_set)
			data_set_norm=np.log(data_set)
			'''
			nan_index=np.argwhere(np.isnan(data_set_norm))
			#print(nan_index)
			for index in nan_index:
				data_set_norm[index]=0
			'''
			inf_index=np.argwhere(np.isinf(data_set_norm))
			for index in inf_index:
				data_set_norm[index]=np.nan
			
		else:
			data_set_norm=data_set
		offset.append(-1.*np.nanmin(data_set_norm))
		scale.append(1./(np.nanmax(data_set_norm)-np.nanmin(data_set_norm)))
		data_set_norm=(data_set_norm+offset[-1])*scale[-1]
		data_set_norm_list.append(data_set_norm)

	d = {'factor':scale,\
		'offset':offset,\
		'log':log_list}
	df_norm=pd.DataFrame(d, columns=['factor','offset','log'])   #construct the panda dataframe
	df_norm.to_csv(csv_name,index=False)
	data_set_norm_list=np.array(data_set_norm_list,dtype='float')
	return data_set_norm_list

