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

f1=open("out.txt","r")

lines=f1.readlines()

d={}
i=-1
d['trial_id']=[]
d['layer']=[]
d['layer_num']=[]
d['activation']=[]
d['learning_rate']=[]
d['Score']=[]

for line in lines:
	if 'Trial summary' in line:
		i=i+1
		d['trial_id'].append(i)
		d['layer'].append([])
		d['layer_num'].append([])

		continue

	tmp=line.split(':')

	if 'activation' in tmp[0]:
		d['activation'].append(tmp[1])
	if 'layer_' in tmp[0]:
		d['layer'][i].append(int(tmp[1]))
		d['layer_num'][i].append(int(tmp[0][6:]))
	if 'learning_rate' in tmp[0]:
		d['learning_rate'].append(float(tmp[1]))
	if 'Score' in tmp[0]:
		d['Score'].append(float(tmp[1]))


keys=list(d)
for key in keys:
	if len(d[key])!=i+1:
		d.pop(key)


df=pd.DataFrame(d)
print(df)

layer_1=[df['layer'][i][0] for i in range(len(df))]

plt.clf()
plt.scatter(layer_1,d['Score'])
plt.xlabel('layer_1')
plt.ylabel('Score')
plt.show()