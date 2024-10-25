import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

n_samples=100

# Step 1: Create a Sample Dataset
x=np.random.normal(0, 1, n_samples)
data = {
    'Feature1': x,
    'Feature2': x+1*np.random.normal(0, 1, n_samples)
}
df = pd.DataFrame(data)

# Step 2: Standardize the Data
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)

# Step 3: Compute the Covariance Matrix
cov_matrix = np.cov(df_standardized.T)

# Step 4: Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Transform the Data Using PCA
pca_data = np.dot(df_standardized, eigenvectors)

# Step 6: Visualize the Results
plt.scatter(df_standardized[:, 0], df_standardized[:, 1], alpha=0.6, label='Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the principal components
origin = np.zeros((2, 2))  # origin point for the arrows
plt.quiver(*origin, 0.5*eigenvalues[:]*eigenvectors[0, :], 0.5*eigenvalues[:]*eigenvectors[1, :], color=['r', 'g'], scale=3)

plt.legend()
plt.title('PCA - Principal Components')
plt.show()

# Display Eigenvalues and Eigenvectors
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
