from general_read_tunning_output_function import read_tunning_output
import matplotlib.pyplot as plt

#**************start of user block****************
file_name="out.txt"
hyper_paramerter_name_list=['layer_1']
hyper_paramerter_dtype_list=['int']
#**************end of user block****************

df=read_tunning_output(file_name,hyper_paramerter_name_list,hyper_paramerter_dtype_list)

print(df)

plt.clf()
plt.scatter(df['layer_1'],df['Score'])
plt.xlabel('layer_1')
plt.ylabel('Score')
plt.show()