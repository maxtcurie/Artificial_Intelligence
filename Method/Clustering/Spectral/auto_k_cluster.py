import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels

# Function to compute similarity matrix
def compute_similarity(data, sigma=1.0):
    return pairwise_kernels(data, metric='rbf', gamma=1 / (2 * sigma ** 2))

# Function to compute Laplacian matrix
def compute_laplacian(similarity_matrix, normalize=True):
    D = np.diag(similarity_matrix.sum(axis=1))
    if normalize:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diagonal(D)))
        return np.eye(len(D)) - D_inv_sqrt @ similarity_matrix @ D_inv_sqrt
    else:
        return D - similarity_matrix

# Function to estimate the number of clusters using the eigenvalue gap
def estimate_clusters_eigenvalue_gap(laplacian):
    eigenvalues, _ = np.linalg.eigh(laplacian)
    gaps = np.diff(eigenvalues)
    k = np.argmax(gaps) + 1
    return k, eigenvalues

# Function to find the optimal number of clusters using silhouette score
def find_optimal_k(U, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(U)
        score = silhouette_score(U, labels)
        scores.append(score)
    optimal_k = np.argmax(scores) + 2
    return optimal_k, scores

# Spectral clustering function
def spectral_clustering(data, n_clusters, sigma=1.0):
    similarity_matrix = compute_similarity(data, sigma)
    laplacian = compute_laplacian(similarity_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    U = eigenvectors[:, :n_clusters]
    U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(U_normalized)
    return labels, laplacian, U

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

# Generate synthetic data
data1 =generate_ring_data(num_points=400, inner_radius=5, outer_radius=15)
data2 =generate_ring_data(num_points=400, inner_radius=40, outer_radius=50)

data=np.concatenate([data1,data2],axis=0)

# Demonstration Script
sigma = 1.0
similarity_matrix = compute_similarity(data, sigma)
laplacian = compute_laplacian(similarity_matrix)
k_eigenvalue, eigenvalues = estimate_clusters_eigenvalue_gap(laplacian)

# Plot Eigenvalue Spectrum
plt.figure(figsize=(10, 5))
plt.plot(eigenvalues, marker='o', label='Eigenvalues')
plt.title("Eigenvalue Spectrum")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.axvline(x=k_eigenvalue, color='r', linestyle='--', label=f'Estimated k={k_eigenvalue}')
plt.legend()
plt.show()

# Perform spectral clustering
labels, laplacian, U = spectral_clustering(data, k_eigenvalue, sigma)

# Find optimal k using silhouette scores
optimal_k_silhouette, scores = find_optimal_k(U, max_k=10)

# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), scores, marker='o', label='Silhouette Scores')
plt.title("Silhouette Scores vs Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.axvline(x=optimal_k_silhouette, color='r', linestyle='--', label=f'Optimal k={optimal_k_silhouette}')
plt.legend()
plt.show()

# Scatter plot of clusters
plt.figure(figsize=(10, 5))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
plt.title("Spectral Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
