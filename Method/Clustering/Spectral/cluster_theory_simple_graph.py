#%%
import numpy as np
from numpy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
#%%
def spectral_clustering(S,k,plot=True):
    # Step 1: Compute the similarity matrix
    '''
        1---|    |---6
        |   3----4   |
        2---|    |---5
    '''


    S  = [[0, 1, 1, 0, 0, 0],
          [1, 0, 1, 0, 0, 0],
          [1, 1, 0, 1, 0, 0],
          [0, 0, 1, 0, 1, 1],
          [0, 0, 0, 1, 0, 1],
          [0, 0, 0, 1, 1, 0]]
    S=np.array(S)
    D = np.diag(S.sum(axis=1))
    # Step 3: Compute the Laplacian matrix
    L = D - S

    # Step 4: Compute the first k eigenvectors of the Laplacian
    eigvals, eigvecs = eigh(L)
    print('eigvals')
    print(eigvals)

    print('eigvecs')
    print(eigvecs.T)

    U = eigvecs[:, :k]

    if plot:
        plt.clf()
        plt.plot(eigvals)
        plt.title('eigvals')
        plt.show()

        plt.clf()
        plt.plot(eigvecs)
        plt.title('eigvecs')
        plt.show()
    # Step 5: Normalize the rows
    U_normalized = U / np.sqrt(np.sum(U ** 2, axis=1, keepdims=True))

    # Step 6: K-Means on normalized eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U_normalized)
    labels = kmeans.labels_

    return labels
#%%
def generate_ring_data(num_points=200, inner_radius=5, outer_radius=10):
    # Generate random angles
    angles = 2 * np.pi * np.random.rand(num_points)

    # Random radii, uniformly sampled between inner and outer radius
    radii = np.random.uniform(inner_radius, outer_radius, num_points)

    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    return np.column_stack((x, y))
#%%
def plot_clusters(data, labels):
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title('2D Ring-Shaped Cluster Sample Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
#%%
def visualize_graph_from_adjacency_matrix(adj_matrix):
    # Convert adjacency matrix to a NetworkX graph
    G = nx.from_numpy_array(np.array(adj_matrix))

    # Draw the graph
    nx.draw(G, with_labels=True, node_color='lightgreen', font_weight='bold', node_size=500)

    # Display the graph
    plt.show()
#%%
def plot_graph_with_spectral_clustering(adj_matrix, labels):
    # Convert adjacency matrix to a NetworkX graph
    G = nx.from_numpy_array(np.array(adj_matrix))

    # Get unique labels and assign a color to each cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    # Create a color mapping based on labels
    node_colors = [colors(label) for label in labels]

    # Draw the graph
    nx.draw(G, with_labels=True, node_color=node_colors, font_weight='bold', node_size=500, cmap='viridis')

    # Display the graph
    plt.show()


#%%
# Main execution
def main():

    '''
        1---|    |---6
        |   3----4   |
        2---|    |---5
    '''


    S  = [[0, 1, 1, 0, 0, 0],
          [1, 0, 1, 0, 0, 0],
          [1, 1, 0, 1, 0, 0],
          [0, 0, 1, 0, 1, 1],
          [0, 0, 0, 1, 0, 1],
          [0, 0, 0, 1, 1, 0]]
    S=np.array(S)

    # Call the function to visualize the graph
    visualize_graph_from_adjacency_matrix(S)

    k=2
    # Perform spectral clustering
    labels = spectral_clustering(S,k)

    # Call the function to plot the graph with clustering results
    plot_graph_with_spectral_clustering(S, labels)

if __name__ == "__main__":
    main()
#%%
