import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

num_components = 3

# Step 1: Generate a synthetic dataset
np.random.seed(42)
n_samples = 100
# Create correlated features
feature1 = np.random.normal(0, 1, n_samples)
feature2 = np.random.normal(0, 1, n_samples)
feature3 = 1 * feature1 
feature4 = 0.5 * feature1+0.5*feature2 

data = np.vstack([feature1, feature2, feature3, feature4]).T
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3','feature4'])

# Step 2: Standardize the data
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)

# Step 3: Compute the Covariance Matrix
cov_matrix = np.cov(df_standardized, rowvar=False)

corr_matrix = df.corr()

print('cov_matrix:')
print(cov_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# Step 4: Compute the Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort Eigenvalues and Eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Step 6: Project the Data onto Principal Components

selected_eigenvectors = sorted_eigenvectors[:, :num_components]
projected_data = np.dot(df_standardized, selected_eigenvectors)
df_pca = pd.DataFrame(projected_data, columns=[f'PC{i+1}' for i in range(num_components)])

#cov_dot_select_eigen = np.linalg.invt(selected_eigenvectors) @ cov_matrix @ selected_eigenvectors

#print(f'cov*select_eigen: {cov_dot_select_eigen}')


# Print the explained variance ratio
explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
for i,j in zip(sorted_eigenvalues,sorted_eigenvectors):
    print('*******************')
    print("Eigenvalues:", i)
    print("Eigenvectors:", j)

print("Explained Variance Ratio:", explained_variance_ratio)


# Feature importance calculation
feature_importance = np.abs(eigenvectors.T * explained_variance_ratio)

# Summing the importance scores for each feature
feature_importance_scores = feature_importance.sum(axis=1)

print(f'feature_importance_scores:{feature_importance_scores}')

# Plot the original data and principal components
fig, ax = plt.subplots(1, 3, figsize=(14, 6))

# Original data
ax[0].scatter(df['feature1'], df['feature2'], alpha=0.5, label='Feature 1 vs 2')
ax[0].scatter(df['feature1'], df['feature3'], alpha=0.5, label='Feature 1 vs 3')
ax[0].scatter(df['feature1'], df['feature4'], alpha=0.5, label='Feature 1 vs 4')
ax[1].scatter(df['feature2'], df['feature3'], alpha=0.5, label='Feature 2 vs 3')
ax[1].scatter(df['feature2'], df['feature4'], alpha=0.5, label='Feature 2 vs 4')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2 / 3 / 4')
ax[1].set_xlabel('Feature 2')
ax[1].set_ylabel('Feature 3 / 4')
ax[0].legend()
ax[0].set_title('Original Data')

# Principal components
ax[2].scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.5)
ax[2].set_xlabel('Principal Component 1')
ax[2].set_ylabel('Principal Component 2')
ax[2].set_title('PCA Result')

plt.show()


#eigen vectors
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.6)
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
# Plot the principal components
origin = np.zeros((2, 2))  # origin point for the arrows
ax[0].quiver(*origin, 0.6*eigenvalues[:2]*eigenvectors[0, :2], 0.6*eigenvalues[:2]*eigenvectors[1, :2], color=['r', 'g'], scale=3)


ax[1].scatter(df_pca['PC1'], df_pca['PC3'], alpha=0.6)
ax[1].set_xlabel('Principal Component 1')
ax[1].set_ylabel('Principal Component 3')
# Plot the principal components
origin = np.zeros((2, 2))  # origin point for the arrows
ax[1].quiver(*origin, 0.6*eigenvalues[::2]*eigenvectors[0, ::2], 0.6*eigenvalues[::2]*eigenvectors[1, ::2], color=['r', 'g'], scale=3)

plt.title('PCA - Principal Components')
plt.show()
