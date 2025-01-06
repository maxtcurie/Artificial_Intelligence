import numpy as np


def silhouette_score_numpy(data, labels):
    """
    Compute silhouette scores for clustering using only NumPy.

    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - labels: numpy array of shape (n_samples,), cluster labels for each point

    Returns:
    - overall_score: The mean silhouette score for all data points.
    - individual_scores: Silhouette score for each data point.
    """
    unique_labels = np.unique(labels)
    n_samples = len(data)
    individual_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Points in the same cluster as the current point
        same_cluster = data[labels == labels[i]]
        other_clusters = unique_labels[unique_labels != labels[i]]

        # Intra-cluster distance (a(i))
        a_i = np.mean(np.linalg.norm(same_cluster - data[i], axis=1)) if len(same_cluster) > 1 else 0

        # Inter-cluster distances (b(i))
        b_i = np.min([
            np.mean(np.linalg.norm(data[labels == cluster] - data[i], axis=1))
            for cluster in other_clusters
        ]) if len(other_clusters) > 0 else 0

        # Silhouette score for point i
        individual_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    overall_score = np.mean(individual_scores)
    return overall_score, individual_scores


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.rand(10, 2)  # Random dataset of 10 points in 2D
    labels = np.random.randint(0, 3, size=10)  # Random cluster labels (3 clusters)

    overall_score, individual_scores = silhouette_score_numpy(data, labels)
    print(f"Overall Silhouette Score: {overall_score}")
    print(f"Individual Silhouette Scores: {individual_scores}")
