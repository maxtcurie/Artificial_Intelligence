import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer

# Generate a sample dataset with 10,000 data points using Student's t-distribution
np.random.seed(0)
data = {
    'A': np.concatenate([np.random.standard_t(df=2, size=10000), 500+np.random.standard_t(df=2, size=1000)])   # t-distribution with 2 degrees of freedom
}
df = pd.DataFrame(data)

# Apply normalization methods
min_max_scaler = MinMaxScaler()
df_min_max = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)

standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)

max_abs_scaler = MaxAbsScaler()
df_max_abs = pd.DataFrame(max_abs_scaler.fit_transform(df), columns=df.columns)

robust_scaler = RobustScaler()
df_robust = pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)

l2_normalizer = Normalizer(norm='l2')
df_l2 = pd.DataFrame(l2_normalizer.fit_transform(df), columns=df.columns)

quantile_transformer_uniform = QuantileTransformer(output_distribution='uniform')
df_quantile_uniform = pd.DataFrame(quantile_transformer_uniform.fit_transform(df), columns=df.columns)

quantile_transformer_normal = QuantileTransformer(output_distribution='normal')
df_quantile_normal = pd.DataFrame(quantile_transformer_normal.fit_transform(df), columns=df.columns)

power_transformer = PowerTransformer(method='box-cox')
df_power = pd.DataFrame(power_transformer.fit_transform(df[df > 0]), columns=df.columns)  # Box-Cox requires positive data

# Plot histograms
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
methods = {
    'Original Data': df,
    'Min-Max Scaling': df_min_max,
    'Standard Scaling': df_standard,
    'Max Absolute Scaling': df_max_abs,
    'Robust Scaling': df_robust,
    'L2 Normalization': df_l2,
    'Quantile Transformation (Uniform)': df_quantile_uniform,
    'Quantile Transformation (Normal)': df_quantile_normal,
    'Power Transformation (Box-Cox)': df_power
}

for i, (method, result) in enumerate(methods.items()):
    ax = axes[i // 5, i % 5]
    ax.hist(result['A'], bins=30, edgecolor='black')
    ax.set_title(f'{method}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

fig.tight_layout()
plt.show()

# Plot histograms with log scale for y-axis
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
methods = {
    'Original Data': df,
    'Min-Max Scaling': df_min_max,
    'Standard Scaling': df_standard,
    'Max Absolute Scaling': df_max_abs,
    'Robust Scaling': df_robust,
    'L2 Normalization': df_l2,
    'Quantile Transformation (Uniform)': df_quantile_uniform,
    'Quantile Transformation (Normal)': df_quantile_normal,
    'Power Transformation (Box-Cox)': df_power
}

for i, (method, result) in enumerate(methods.items()):
    ax = axes[i // 5, i % 5]
    ax.hist(result['A'], bins=30, edgecolor='black')
    ax.set_title(f'{method}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')

fig.tight_layout()
plt.show()
