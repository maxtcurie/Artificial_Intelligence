#from: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans
import numpy as np

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
    for i in range(n_center):
        a=[np.random.normal(loc=mu_list[i,:],scale=sigma_list[i,:])\
            for j in range(1000)]
        if i==0:
            output=a 
        else:
            output=np.concatenate((output,a),axis=0)
    test_dataset=np.array(output)
    print(np.shape(output))
    if show_plot:
        plt.clf()
        plt.scatter(output[:,0],output[:,1])
        plt.show()
    return test_dataset


def plot_2D(cluster_label,compare_label,df,plot_name_list,save_fig=False):
    fig, ax=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True) 

    print(cluster_label)
    n_clusters=np.arange(np.min(cluster_label),np.max(cluster_label)+1)
    for i in n_clusters:
        index_tmp=(cluster_label == i)
        ax[0].scatter(  df[plot_name_list[0]][index_tmp],\
                        df[plot_name_list[1]][index_tmp],\
                        alpha=0.5)
    ax[0].set_xlabel(plot_name_list[0])
    ax[0].set_ylabel(plot_name_list[1])
    ax[0].set_title('clustering (n='+str(len(n_clusters))+')')
    
    print(compare_label)
    n_clusters=np.arange(np.min(compare_label),np.max(compare_label)+1)
    for i in n_clusters:
        index_tmp=(compare_label == i)
        ax[1].scatter(  df[plot_name_list[0]][index_tmp],\
                        df[plot_name_list[1]][index_tmp],\
                        alpha=0.5)
    ax[1].set_xlabel(plot_name_list[0])
    ax[1].set_title('label')

    #ax[1].set_xlim(0,0.5)
    #ax[1].set_ylim(0,0.5)

    plt.show()

    return 0



test_data(mu_list,sigma_list,show_plot=False)

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print('kmeans.labels_')
print(kmeans.labels_)

print('kmeans.cluster_centers_')
print(kmeans.cluster_centers_)

print('kmeans.predict([[0, 0], [12, 3]])')
print(kmeans.predict([[0, 0], [12, 3]]))


