import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

# Generate sample datasets with 10,000 data points each using Student's t-distribution
np.random.seed(0)
data1 = np.concatenate([np.random.standard_t(df=2, size=10000), 500 + np.random.standard_t(df=2, size=1000)])   # t-distribution with 2 degrees of freedom
data2 = np.concatenate([np.random.standard_t(df=3, size=10000), 1000 + np.random.standard_t(df=3, size=1000)])  # t-distribution with 3 degrees of freedom

# Create DataFrames
df1 = pd.DataFrame({'A': data1})
df2 = pd.DataFrame({'B': data2})

# Apply power transformation (Box-Cox)
power_transformer1 = PowerTransformer(method='box-cox')
# Box-Cox requires positive data
df1_positive = df1[df1['A'] > 0]
df1_transformed = pd.DataFrame(power_transformer1.fit_transform(df1_positive), columns=df1_positive.columns)

power_transformer2 = PowerTransformer(method='box-cox')
# Box-Cox requires positive data
df2_positive = df2[df2['B'] > 0]
df2_transformed = pd.DataFrame(power_transformer2.fit_transform(df2_positive), columns=df2_positive.columns)

# Merge the transformed datasets
merged_df = pd.concat([df1_transformed, df2_transformed], axis=1, join='outer')

# Plot histograms of the original and transformed data
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

# Original Data
axes[0, 0].hist(df1['A'], bins=30, edgecolor='black')
axes[0, 0].set_title('Original Data A')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(df2['B'], bins=30, edgecolor='black')
axes[0, 1].set_title('Original Data B')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')

# Transformed Data
axes[1, 0].hist(df1_transformed['A'], bins=30, edgecolor='black')
axes[1, 0].set_title('Transformed Data A')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(df2_transformed['B'], bins=30, edgecolor='black')
axes[1, 1].set_title('Transformed Data B')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')

fig.tight_layout()
plt.show()
