# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt

mu_list=[   [1.,2.],\
            [5.,5.],\
            [10.,8.]\
            ]

sigma_list=[   [1.,1.],\
            [1.,1.],\
            [0.5,5.]\
            ]

def test_data(mu_list,sigma_list,show_plot=False):
    mu_list=np.array(mu_list)
    sigma_list=np.array(sigma_list)
    (n_center,dim)=np.shape(mu_list)

    output=[]
    label=[]
    for i in range(n_center):
        a=[np.random.normal(loc=mu_list[i,:],scale=sigma_list[i,:])\
            for j in range(1000)]
        b=[i]*1000
        if i==0:
            output=a 
            label=b
        else:
            output=np.concatenate((output,a),axis=0)
            label=np.concatenate((label,b),axis=0)
    test_dataset=np.array(output)
    test_label=np.array(label)
    #print(np.shape(output))
    if show_plot:
        plt.clf()
        plt.scatter(output[:,0],output[:,1])
        plt.show()
    return test_dataset,test_label


def plot_2D(cluster_label,test_dataset,compare_label,save_fig=False):
    fig, ax=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True) 

    #print(cluster_label)
    n_clusters=np.arange(np.min(cluster_label),np.max(cluster_label)+1)
    for i in n_clusters:
        index_tmp=(cluster_label == i)
        print(np.shape((test_dataset[index_tmp])[:,0]))
        ax[0].scatter(  (test_dataset[index_tmp])[:,0],\
                        (test_dataset[index_tmp])[:,1],\
                        alpha=0.5)
    #ax[0].set_xlabel(plot_name_list[0])
    #ax[0].set_ylabel(plot_name_list[1])
    ax[0].set_title('clustering (n='+str(len(n_clusters))+')')
    
    print(compare_label)
    n_clusters=np.arange(np.min(compare_label),np.max(compare_label)+1)
    for i in n_clusters:
        index_tmp=(compare_label == i)
        ax[1].scatter(  (test_dataset[index_tmp])[:,0],\
                        (test_dataset[index_tmp])[:,1],\
                        alpha=0.5)
    #ax[1].set_xlabel(plot_name_list[0])
    ax[1].set_title('label')

    #ax[1].set_xlim(0,0.5)
    #ax[1].set_ylim(0,0.5)

    plt.show()

    return 0



test_dataset,test_label=test_data(mu_list,sigma_list,show_plot=True)

print(np.shape(test_dataset))

cluster_param = OPTICS(min_samples=100).fit(test_dataset)

plot_2D(cluster_param.labels_,test_dataset,test_label,save_fig=False)

print('kmeans.labels_')
print(cluster_param.labels_)

print('kmeans.cluster_centers_')
print(cluster_param.cluster_centers_)

print('kmeans.predict([[0, 0], [12, 3]])')
print(cluster_param.predict([[0, 0], [12, 3]]))


