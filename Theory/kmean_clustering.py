#refered from: https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/programming/hxhn9/k-means

import numpy as np
import matplotlib.pyplot as plt

from utility import plot_progress_kMeans

def kMeans_init_centroids(X, K):
    #This function initializes K centroids 
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids


def compute_centroids(X, idx, K):
    #Returns the new centroids: new_center=avg(cluster)
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    for k in range(K):
        filter_arr = (idx == k)
        x_list=X[filter_arr]
        centroids[k]=1./float(len(x_list))*np.sum(x_list,axis=0)

    return centroids


def find_closest_centroids(X, centroids):
    #Computes the index of the centroid memberships for every example
    #centroids: center of the cluster
    K = centroids.shape[0]
    
    idx = np.zeros(X.shape[0], dtype=int)

    for m in range(X.shape[0]):
        x=X[m,:]
        dis=np.linalg.norm(x-centroids,axis=1)
        idx[m]=np.argmin(dis)
        
    return idx


def run_kMeans(X, initial_centroids, feature_weight=1,max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    
    m, n = X.shape  #sample_num, feature
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    
    if feature_weight==1:
        feature_weight=[1.]*K

    # Run K-Means
    for i in range(max_iters):
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        #get std_dev=1/n*sum(x**2.)**0.5
        std_dev=np.zeros((K,n))
        #plt.clf()
        for j in range(K):
            dis_tmp=1./len(X[(idx == j)])*\
                    np.sum((X[(idx == j)]-centroids[j])**2.,axis=0)
            std_dev[j,:]=dis_tmp
            #print(dis_tmp)
            #plt.scatter((X[(idx == j)])[:,0],(X[(idx == j)])[:,1])
            #plt.scatter(centroids[j][0],centroids[j][1])
        #plt.show()

        std_dev_sum=std_dev


        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 

    return centroids, std_dev, idx



def kMeans(test_dataset,cluster_num,\
            feature_weight=1,max_iters=100,\
            plot_progress=False):
    initial_centroids=kMeans_init_centroids(test_dataset, cluster_num)
    center, std_dev, label=run_kMeans(test_dataset, initial_centroids, \
                                    feature_weight=feature_weight,\
                                    max_iters=max_iters, \
                                    plot_progress=plot_progress)
    print('std='+str(std_dev))
    return center, label

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



mu_list=[   [1.,2.],\
            [5.,5.],\
            [10.,8.]\
            ]

sigma_list=[   [1.,1.],\
            [1.,1.],\
            [0.5,0.5]\
            ]

cluster_num=3

test_dataset=test_data(mu_list,sigma_list,show_plot=False)

kMeans(test_dataset,cluster_num,feature_weight=1,max_iters=20,plot_progress=True)