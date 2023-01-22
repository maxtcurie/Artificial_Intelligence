import pandas as pd
import matplotlib.pyplot as plt

# add this block to the tunning
# after tuner.search
'''
import sys

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

tuner.results_summary(num_trials=10**10)

sys.stdout = orig_stdout
f.close()
'''
#**************start of user block****************

f1=open("out.txt","r")
hyper_paramerter_name_list=['layer_1']
hyper_paramerter_dtype_list=['int']
#**************end of user block****************

lines=f1.readlines()

d={}
d['trial_id']=[]
d['Score']=[]

for name in hyper_paramerter_name_list: 
	d[name]=[]
	
i=-1
for line in lines:
	if 'Trial summary' in line:
		i=i+1
		d['trial_id'].append(i)
		continue

	tmp=line[:-1].split(':')

	for name in hyper_paramerter_name_list: 
		if name in tmp[0]:
			d[name].append(tmp[1])

	if 'Score' in tmp[0]:
			d['Score'].append(float(tmp[1]))
	

keys=list(d)
for key in keys:
	if len(d[key])!=i+1:
		d.pop(key)

df=pd.DataFrame(d)

convert_dict = {}
for (name,type_) in zip(hyper_paramerter_name_list,hyper_paramerter_dtype_list):
	convert_dict[name]=type_
df=df.astype(convert_dict)

print(df.dtypes)

print(df)

plt.clf()
plt.scatter(df['layer_1'],df['Score'])
plt.xlabel('layer_1')
plt.ylabel('Score')
plt.show()