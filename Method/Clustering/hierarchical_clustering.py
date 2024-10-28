import torch
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Step 1: Generate some example data
torch.manual_seed(0)
# Random 2D data points
data = torch.randn(20, 2)

# Step 2: Calculate pairwise distances
def pairwise_distance(data):
    # Expand data to calculate pairwise differences
    n = data.shape[0]
    data_exp = data.unsqueeze(1).repeat(1, n, 1)
    diff = data_exp - data_exp.transpose(0, 1)
    
    # Calculate Euclidean distance (squared)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
    return dist

# Calculate the distance matrix
distance_matrix = pairwise_distance(data)

# Step 3: Hierarchical clustering with single linkage
def hierarchical_clustering(dist_matrix):
    # Convert the distance matrix to numpy for scipy linkage
    dist_numpy = dist_matrix.numpy()
    
    # Flatten the upper triangular part of the distance matrix (needed for linkage)
    triu_indices = torch.triu_indices(dist_numpy.shape[0], dist_numpy.shape[1], offset=1)
    condensed_dist = dist_numpy[triu_indices[0], triu_indices[1]]
    
    # Perform hierarchical clustering using scipy linkage
    Z = sch.linkage(condensed_dist, method='single')  # single linkage
    return Z

# Perform clustering
Z = hierarchical_clustering(distance_matrix)

# Step 4: Plot dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(Z, labels=[str(i) for i in range(len(data))])
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
