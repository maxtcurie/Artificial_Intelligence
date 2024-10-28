import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

# Generate a sample dataset with 10,000 data points using Student's t-distribution
np.random.seed(0)
data = {
    'A': np.random.standard_t(df=2, size=10000)  # t-distribution with 2 degrees of freedom
}
df = pd.DataFrame(data)

# Define different quantile settings
quantile_settings = [10, 50, 100, 1000]

# Apply quantile transformations with different quantiles
quantile_results = {}
for n_quantiles in quantile_settings:
    quantile_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
    transformed_data = pd.DataFrame(quantile_transformer.fit_transform(df), columns=df.columns)
    quantile_results[f'Quantile Transformation ({n_quantiles} quantiles)'] = transformed_data

# Plot histograms
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
for i, (method, result) in enumerate(quantile_results.items()):
    ax = axes[i // 2, i % 2]
    ax.hist(result['A'], bins=30, edgecolor='black')
    ax.set_title(f'{method}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

fig.tight_layout()
plt.show()
